import subprocess
import os
import tempfile
import networkx as nx
import torch
import logging
import argparse
import json
import pickle
import re
import signal
from typing import Optional
import threading

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("操作超时")

def extract_top_module(verilog_code):
    """从Verilog代码中提取顶层模块名称"""
    try:
        # 使用正则表达式匹配module声明
        module_pattern = r'module\s+(\w+)\s*\(?'
        matches = re.findall(module_pattern, verilog_code)
        
        if matches:
            # 返回第一个找到的模块名称
            top_module = matches[0]
            logging.info(f"检测到顶层模块: {top_module}")
            return top_module
        else:
            logging.warning("未找到模块声明，使用默认模块名'top'")
            return "top"
    except Exception as e:
        logging.warning(f"提取顶层模块名称时出错: {e}，使用默认模块名'top'")
        return "top"

def verilog_to_netlist(verilog_code, output_netlist, timeout=60):
    """使用Yosys将Verilog代码转换为网表，带超时机制"""
    try:
        # 首先将代码写入临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
            f.write(verilog_code)
            temp_verilog = f.name
        
        # 提取顶层模块名称
        top_module = extract_top_module(verilog_code)
        
        yosys_script = f"""
        read_verilog {temp_verilog}
        synth -top {top_module}
        write_blif {output_netlist}
        """
        
        yosys_command = ["yosys", "-p", yosys_script]
        
        # 设置超时
        try:
            result = subprocess.run(yosys_command, check=True, text=True, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            logging.error(f"Yosys处理超时 ({timeout}秒)")
            # 清理临时文件
            os.unlink(temp_verilog)
            if os.path.exists(output_netlist):
                os.unlink(output_netlist)
            return False
        
        # 清理临时文件
        os.unlink(temp_verilog)
        
        # 检查输出文件是否生成
        if not os.path.exists(output_netlist) or os.path.getsize(output_netlist) == 0:
            logging.error("Yosys未生成有效的网表文件")
            return False
            
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"运行Yosys时出错: {e.stderr}")
        # 清理临时文件
        if 'temp_verilog' in locals():
            try:
                os.unlink(temp_verilog)
            except:
                pass
        if os.path.exists(output_netlist):
            try:
                os.unlink(output_netlist)
            except:
                pass
        return False
    except FileNotFoundError:
        logging.error("未找到Yosys，请确保已安装Yosys")
        return False
    except Exception as e:
        logging.error(f"生成网表时发生未知错误: {e}")
        # 清理临时文件
        if 'temp_verilog' in locals():
            try:
                os.unlink(temp_verilog)
            except:
                pass
        if os.path.exists(output_netlist):
            try:
                os.unlink(output_netlist)
            except:
                pass
        return False

def parse_blif_to_graph(blif_file):
    """解析BLIF网表文件并转换为图表示"""
    try:
        graph = nx.DiGraph()
        
        with open(blif_file, 'r') as f:
            lines = f.readlines()
            
        current_model = None
        inputs = []
        outputs = []
        gates = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('.model'):
                current_model = line.split()[1]
                graph.add_node(current_model, type='model')
            elif line.startswith('.inputs'):
                inputs = line.split()[1:]
                for inp in inputs:
                    graph.add_node(inp, type='input')
                    if current_model:
                        graph.add_edge(inp, current_model)
            elif line.startswith('.outputs'):
                outputs = line.split()[1:]
                for out in outputs:
                    graph.add_node(out, type='output')
                    if current_model:
                        graph.add_edge(current_model, out)
            elif line.startswith('.names'):
                # 简单门表示
                parts = line.split()
                if len(parts) >= 2:
                    output = parts[-1]
                    input_list = parts[1:-1]
                    
                    graph.add_node(output, type='gate', gate_type='LUT')
                    for inp in input_list:
                        if inp not in graph:
                            graph.add_node(inp, type='wire')
                        graph.add_edge(inp, output)
                    gates.append(('LUT', len(input_list), 1))
            elif line.startswith('.gate'):
                # 复杂门实例化
                parts = line.split()
                gate_type = parts[1]
                gate_name = f"{gate_type}_{len([n for n in graph.nodes() if n.startswith(gate_type)])}"
                
                graph.add_node(gate_name, type='gate', gate_type=gate_type)
                gates.append((gate_type, 0, 0))
                
                for port in parts[2:]:
                    if '=' in port:
                        net, port_name = port.split('=')
                        if net not in graph:
                            graph.add_node(net, type='wire')
                        graph.add_edge(net, gate_name, port=port_name)
            elif line.startswith('.latch'):
                # 锁存器
                parts = line.split()
                if len(parts) >= 3:
                    input_net = parts[1]
                    output_net = parts[2]
                    graph.add_node(output_net, type='latch')
                    if input_net not in graph:
                        graph.add_node(input_net, type='wire')
                    graph.add_edge(input_net, output_net)
                    gates.append(('LATCH', 1, 1))
        
        return graph, gates
    except Exception as e:
        logging.error(f"解析BLIF文件 {blif_file} 时出错: {e}")
        return None, []

class NetlistFeatureExtractor:
    """
    处理Verilog网表特征提取的类
    """
    
    def __init__(self, timeout=60, log_level=logging.INFO):
        self.timeout = timeout
        logging.getLogger().setLevel(log_level)
        
        # 预定义特征维度 - 根据之前的成功案例设置为24维
        self.common_types = ['model', 'input', 'output', 'gate', 'wire', 'latch']
        self.common_gates = ['AND', 'OR', 'XOR', 'NAND', 'NOR', 'XNOR', 'BUF', 'NOT', 'LUT']
        
        # 计算特征向量总维度
        # 基本特征: 9个
        # 类型特征: len(common_types)个 = 6个
        # 门特征: len(common_gates)个 = 9个
        # 总维度: 9 + 6 + 9 = 24维
        self.feature_dim = 24
        logging.info(f"网表特征向量维度: {self.feature_dim}")

    def get_zero_features(self) -> torch.Tensor:
        """返回24维全零特征向量"""
        return torch.zeros(self.feature_dim, dtype=torch.float32)

    def extract_netlist_features_from_code(self, verilog_code):
        """从Verilog代码提取网表特征向量，带超时机制"""
        temp_blif = None
        try:
            logging.info("正在生成网表...")
            
            # 为网表创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.blif', delete=False) as f:
                temp_blif = f.name
            
            # 生成网表（带超时）
            success = verilog_to_netlist(verilog_code, temp_blif, timeout=self.timeout)
            if not success:
                logging.error("生成网表失败")
                if temp_blif and os.path.exists(temp_blif):
                    os.unlink(temp_blif)
                return self.get_zero_features()
            
            # 检查文件是否存在且非空
            if not os.path.exists(temp_blif) or os.path.getsize(temp_blif) == 0:
                logging.error("网表文件为空或不存在")
                if temp_blif and os.path.exists(temp_blif):
                    os.unlink(temp_blif)
                return self.get_zero_features()
            
            # 解析网表为图
            netlist_graph, gates = parse_blif_to_graph(temp_blif)
            
            # 清理临时文件
            if temp_blif and os.path.exists(temp_blif):
                os.unlink(temp_blif)
            
            if netlist_graph is None:
                logging.error("解析网表图失败")
                return self.get_zero_features()
            
            # 提取图特征
            num_nodes = netlist_graph.number_of_nodes()
            num_edges = netlist_graph.number_of_edges()
            density = nx.density(netlist_graph) if num_nodes > 1 else 0
            
            # 节点类型分布
            node_types = {}
            for node, attrs in netlist_graph.nodes(data=True):
                node_type = attrs.get('type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            type_features = [node_types.get(t, 0) for t in self.common_types]
            
            # 门类型分布
            gate_types = {}
            gate_input_counts = []
            gate_output_counts = []
            
            for node, attrs in netlist_graph.nodes(data=True):
                if attrs.get('type') == 'gate':
                    gate_type = attrs.get('gate_type', 'unknown')
                    gate_types[gate_type] = gate_types.get(gate_type, 0) + 1
                    
                    # 估算输入输出数量
                    in_degree = netlist_graph.in_degree(node)
                    out_degree = netlist_graph.out_degree(node)
                    gate_input_counts.append(in_degree)
                    gate_output_counts.append(out_degree)
            
            gate_features = [gate_types.get(gate, 0) for gate in self.common_gates]
            
            # 度统计
            degrees = [deg for _, deg in netlist_graph.degree()]
            avg_degree = sum(degrees) / len(degrees) if degrees else 0
            max_degree = max(degrees) if degrees else 0
            min_degree = min(degrees) if degrees else 0
            
            # 门输入输出统计
            avg_inputs = sum(gate_input_counts) / len(gate_input_counts) if gate_input_counts else 0
            avg_outputs = sum(gate_output_counts) / len(gate_output_counts) if gate_output_counts else 0
            
            # 组合所有特征
            feature_vector = [
                num_nodes, num_edges, density, 
                avg_degree, max_degree, min_degree,
                avg_inputs, avg_outputs,
                len(gates)  # 总门数
            ] + type_features + gate_features
            
            # 确保特征向量维度为24
            if len(feature_vector) > self.feature_dim:
                feature_vector = feature_vector[:self.feature_dim]
                logging.warning(f"特征向量被截断到 {self.feature_dim} 维")
            elif len(feature_vector) < self.feature_dim:
                padding = [0] * (self.feature_dim - len(feature_vector))
                feature_vector = feature_vector + padding
                logging.warning(f"特征向量被填充到 {self.feature_dim} 维")
            
            # 转换为张量
            features = torch.tensor(feature_vector, dtype=torch.float32)
            
            logging.info(f"已提取包含 {len(features)} 维的网表特征向量")
            return features
            
        except TimeoutException:
            logging.error(f"网表特征提取超时 ({self.timeout}秒)")
            if temp_blif and os.path.exists(temp_blif):
                try:
                    os.unlink(temp_blif)
                except:
                    pass
            return self.get_zero_features()
        except Exception as e:
            logging.error(f"计算网表特征时出错: {e}")
            if temp_blif and os.path.exists(temp_blif):
                try:
                    os.unlink(temp_blif)
                except:
                    pass
            return self.get_zero_features()

def process_jsonl_dataset(jsonl_file, output_file, timeout=60):
    """处理JSONL数据集并提取网表特征，保存为单个pkl文件"""
    extractor = NetlistFeatureExtractor(timeout=timeout)
    
    all_features = {}
    success_count = 0
    fail_count = 0
    total_count = 0
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            total_count += 1
            try:
                data = json.loads(line.strip())
                
                # 提取canonical_solution字段
                verilog_code = data.get('canonical_solution', '')
                if not verilog_code:
                    logging.warning(f"第 {i+1} 行没有找到canonical_solution字段")
                    # 为缺失代码的样本分配零特征
                    features = extractor.get_zero_features()
                    sample_id = f"sample_{i+1:05d}"
                    all_features[sample_id] = {
                        'features': features,
                        'shape': features.shape,
                        'verilog_code': "MISSING_CODE",
                        'status': 'missing_code'
                    }
                    fail_count += 1
                    continue
                
                logging.info(f"正在处理第 {i+1} 条数据...")
                
                # 提取网表特征
                features = extractor.extract_netlist_features_from_code(verilog_code)
                
                # 存储特征
                sample_id = f"sample_{i+1:05d}"
                all_features[sample_id] = {
                    'features': features,
                    'shape': features.shape,
                    'verilog_code': verilog_code[:200] + "..." if len(verilog_code) > 200 else verilog_code,
                    'status': 'success' if torch.any(features != 0) else 'zero_features'
                }
                
                if torch.any(features != 0):
                    success_count += 1
                    logging.info(f"第 {i+1} 条数据网表特征提取成功")
                else:
                    fail_count += 1
                    logging.warning(f"第 {i+1} 条数据网表特征为零向量")
                
            except json.JSONDecodeError as e:
                logging.error(f"第 {i+1} 行JSON解析错误: {e}")
                # JSON解析错误也分配零特征
                features = extractor.get_zero_features()
                sample_id = f"sample_{i+1:05d}"
                all_features[sample_id] = {
                    'features': features,
                    'shape': features.shape,
                    'verilog_code': "JSON_PARSE_ERROR",
                    'status': 'json_error'
                }
                fail_count += 1
            except Exception as e:
                logging.error(f"处理第 {i+1} 行时出错: {e}")
                # 其他错误也分配零特征
                features = extractor.get_zero_features()
                sample_id = f"sample_{i+1:05d}"
                all_features[sample_id] = {
                    'features': features,
                    'shape': features.shape,
                    'verilog_code': "PROCESS_ERROR",
                    'status': 'process_error'
                }
                fail_count += 1
            
            # 每处理10条数据保存一次进度
            if total_count % 10 == 0:
                logging.info(f"已处理 {total_count} 条数据，成功 {success_count} 条，失败 {fail_count} 条")
    
    # 保存所有特征到单个pkl文件
    with open(output_file, 'wb') as f:
        pickle.dump(all_features, f)
    
    logging.info(f"网表特征已保存到 {output_file}, 共 {len(all_features)} 个样本")
    logging.info(f"成功处理: {success_count} 个样本")
    logging.info(f"失败处理: {fail_count} 个样本 (已分配零特征)")
    logging.info(f"成功率: {success_count/total_count*100:.2f}%" if total_count > 0 else "成功率: 0%")
    
    # 打印统计信息
    if all_features:
        feature_shapes = [data['shape'] for data in all_features.values()]
        if feature_shapes:
            unique_shapes = set(str(shape) for shape in feature_shapes)
            logging.info(f"特征向量维度统计: {unique_shapes}")
        
        # 统计各种状态的样本数量
        status_count = {}
        for data in all_features.values():
            status = data.get('status', 'unknown')
            status_count[status] = status_count.get(status, 0) + 1
        
        logging.info(f"样本状态统计: {status_count}")

def main():
    parser = argparse.ArgumentParser(description="从JSONL数据集提取网表特征并保存为单个pkl文件")
    parser.add_argument("--jsonl_file", default='/home/syh123/workspace/Data_Filter/data/expanded_origen_120k.jsonl', help="JSONL数据集文件路径")
    parser.add_argument("--output_file", type=str, default="/home/syh123/workspace/Data_Filter/data/Origen/netlist_features.pkl", help="输出pkl文件路径")
    parser.add_argument("--timeout", type=int, default=3, help="Yosys处理超时时间（秒）")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="设置日志级别")
    
    args = parser.parse_args()
    
    # 应用日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 处理JSONL数据集
    process_jsonl_dataset(args.jsonl_file, args.output_file, args.timeout)
    
    logging.info("网表特征提取完成")

if __name__ == "__main__":
    main()