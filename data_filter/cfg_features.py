import networkx as nx
from pyverilog.vparser.parser import parse
from grakel import GraphKernel
import argparse
import logging
import torch
import numpy as np
import os
import json
import pickle
import tempfile
from typing import Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def nx_to_grakel(cfg):
    """将NetworkX图转换为GraKel格式"""
    if cfg.number_of_nodes() == 0:
        return ([], {})
    
    mapping = {old_id: new_id for new_id, old_id in enumerate(cfg.nodes())}
    cfg = nx.relabel_nodes(cfg, mapping)
    edges = list(cfg.edges())
    node_labels = {node: cfg.nodes[node].get('label', 'default') for node in cfg.nodes()}
    return (edges, node_labels)

class VerilogCFGFeatureExtractor:
    """
    处理Verilog控制流图(CFG)生成和特征提取的类
    """

    def __init__(self, n_iter: int = 5, log_level=logging.INFO):
        self.n_iter = n_iter
        logging.getLogger().setLevel(log_level)
        
        # 预定义特征维度
        self.common_labels = ['Source', 'ModuleDef', 'Decl', 'Input', 'Output', 'Reg', 'Wire',
                           'Assign', 'Always', 'Block', 'IfStatement', 'CaseStatement', 
                           'Identifier', 'IntConst', 'Plus', 'Minus', 'Times']
        
        # 计算特征向量总维度
        # 基本特征: 10个
        # 标签特征: len(common_labels)个
        # 核特征: n_iter + 1个
        # self.feature_dim = 10 + len(self.common_labels) + (self.n_iter + 1)
        self.feature_dim = 28

    def get_zero_features(self) -> torch.Tensor:
        """返回全零特征向量"""
        return torch.zeros(self.feature_dim, dtype=torch.float32)

    def create_cfg_from_code(self, code: str) -> Optional[nx.DiGraph]:
        """从Verilog代码创建控制流图"""
        try:
            # 将代码写入临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            ast, _ = parse([temp_file])
            cfg = nx.DiGraph()

            def traverse(node, parent=None):
                """递归遍历AST节点构建CFG"""
                if node is not None:
                    node_id = id(node)
                    cfg.add_node(node_id, label=type(node).__name__)
                    if parent is not None:
                        cfg.add_edge(parent, node_id)
                    # 对子节点排序以确保一致性
                    for child in sorted(node.children(), key=lambda x: str(type(x).__name__)):
                        traverse(child, node_id)

            logging.info("正在从AST构建CFG...")
            for child in ast.children():
                traverse(child)

            logging.info("CFG构建完成。")
            
            # 清理临时文件
            os.unlink(temp_file)
            
            return cfg
        except Exception as e:
            logging.error(f"从代码生成CFG时出错。异常: {e}")
            # 清理临时文件
            if 'temp_file' in locals():
                try:
                    os.unlink(temp_file)
                except:
                    pass
            return None

    def extract_cfg_features(self, cfg: nx.DiGraph) -> torch.Tensor:
        """
        使用图核方法和图统计提取CFG特征向量
        """
        try:
            logging.info("正在提取CFG特征向量...")
            
            # 基本图统计特征
            num_nodes = cfg.number_of_nodes()
            num_edges = cfg.number_of_edges()
            density = nx.density(cfg) if num_nodes > 1 else 0
            
            # 节点度统计
            degrees = [deg for _, deg in cfg.degree()]
            avg_degree = np.mean(degrees) if degrees else 0
            max_degree = np.max(degrees) if degrees else 0
            min_degree = np.min(degrees) if degrees else 0
            std_degree = np.std(degrees) if degrees else 0
            
            # 图直径和半径
            try:
                undirected_cfg = cfg.to_undirected()
                if nx.is_connected(undirected_cfg):
                    diameter = nx.diameter(undirected_cfg)
                    radius = nx.radius(undirected_cfg)
                else:
                    diameter = 0
                    radius = 0
            except:
                diameter = 0
                radius = 0
            
            # 聚类系数
            try:
                avg_clustering = nx.average_clustering(undirected_cfg)
            except:
                avg_clustering = 0
            
            # 节点类型分布
            node_labels = nx.get_node_attributes(cfg, 'label')
            label_counts = {}
            for label in node_labels.values():
                label_counts[label] = label_counts.get(label, 0) + 1
            
            label_features = [label_counts.get(label, 0) for label in self.common_labels]
            
            # 图核特征 (Weisfeiler-Lehman)
            kernel_features = []
            try:
                logging.info(f"正在计算Weisfeiler-Lehman核特征 (n_iter={self.n_iter})...")
                grakel_graph = nx_to_grakel(cfg)
                if grakel_graph[0]:  # 检查是否有边
                    gk = GraphKernel(kernel={"name": "weisfeiler_lehman", "n_iter": self.n_iter}, normalize=True)
                    graphs = [grakel_graph]
                    K = gk.fit_transform(graphs)
                    kernel_features = K[0].tolist()  # 核特征向量
                else:
                    kernel_features = [0] * (self.n_iter + 1)  # 空图的默认特征
            except Exception as e:
                logging.warning(f"图核特征计算失败: {e}")
                kernel_features = [0] * (self.n_iter + 1)
            
            # 组合所有特征
            basic_features = [
                num_nodes, num_edges, density, avg_degree, max_degree, 
                min_degree, std_degree, diameter, radius, avg_clustering
            ]
            
            feature_vector = basic_features + label_features + kernel_features
            
            # 转换为张量
            features = torch.tensor(feature_vector, dtype=torch.float32)
            
            logging.info(f"已提取包含 {len(features)} 维的CFG特征向量")
            return features
            
        except Exception as e:
            logging.error(f"提取CFG特征时出错。异常: {e}")
            # 返回零向量作为后备
            return self.get_zero_features()

def process_jsonl_dataset(jsonl_file, output_file, n_iter=5):
    """处理JSONL数据集并提取CFG特征，保存为单个pkl文件"""
    extractor = VerilogCFGFeatureExtractor(n_iter=n_iter)
    
    all_features = {}
    success_count = 0
    fail_count = 0
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                
                # 提取canonical_solution字段
                verilog_code = data.get('canonical_solution', '')
                if not verilog_code:
                    logging.warning(f"第 {i+1} 行没有找到canonical_solution字段")
                    # 为缺失代码的样本分配零特征
                    features = extractor.get_zero_features()
                    logging.info(f"features:\n{features}")
                    sample_id = f"sample_{i+1:05d}"
                    all_features[sample_id] = {
                        'features': features,
                        'shape': features.shape,
                        'verilog_code': "MISSING_CODE",
                        'cfg_info': {
                            'num_nodes': 0,
                            'num_edges': 0
                        },
                        'status': 'missing_code'
                    }
                    fail_count += 1
                    continue
                
                logging.info(f"正在处理第 {i+1} 条数据...")
                
                # 生成CFG
                cfg = extractor.create_cfg_from_code(verilog_code)
                if cfg is None:
                    logging.error(f"第 {i+1} 条数据CFG生成失败，分配零特征")
                    # CFG生成失败时分配零特征
                    features = extractor.get_zero_features()
                    logging.info(f"features:\n{features}")
                    sample_id = f"sample_{i+1:05d}"
                    all_features[sample_id] = {
                        'features': features,
                        'shape': features.shape,
                        'verilog_code': verilog_code[:200] + "..." if len(verilog_code) > 200 else verilog_code,
                        'cfg_info': {
                            'num_nodes': 0,
                            'num_edges': 0
                        },
                        'status': 'cfg_generation_failed'
                    }
                    fail_count += 1
                    continue
                
                # 提取特征
                features = extractor.extract_cfg_features(cfg)
                
                # 存储特征
                sample_id = f"sample_{i+1:05d}"
                all_features[sample_id] = {
                    'features': features,
                    'shape': features.shape,
                    'verilog_code': verilog_code[:200] + "..." if len(verilog_code) > 200 else verilog_code,
                    'cfg_info': {
                        'num_nodes': cfg.number_of_nodes(),
                        'num_edges': cfg.number_of_edges()
                    },
                    'status': 'success'
                }
                success_count += 1
                
            except json.JSONDecodeError as e:
                logging.error(f"第 {i+1} 行JSON解析错误: {e}")
                # JSON解析错误也分配零特征
                features = extractor.get_zero_features()
                logging.info(f"features:\n{features}")
                sample_id = f"sample_{i+1:05d}"
                all_features[sample_id] = {
                    'features': features,
                    'shape': features.shape,
                    'verilog_code': "JSON_PARSE_ERROR",
                    'cfg_info': {
                        'num_nodes': 0,
                        'num_edges': 0
                    },
                    'status': 'json_error'
                }
                fail_count += 1
            except Exception as e:
                logging.error(f"处理第 {i+1} 行时出错: {e}")
                # 其他错误也分配零特征
                features = extractor.get_zero_features()
                logging.info(f"features:\n{features}")
                sample_id = f"sample_{i+1:05d}"
                all_features[sample_id] = {
                    'features': features,
                    'shape': features.shape,
                    'verilog_code': "PROCESS_ERROR",
                    'cfg_info': {
                        'num_nodes': 0,
                        'num_edges': 0
                    },
                    'status': 'process_error'
                }
                fail_count += 1
    
    # 保存所有特征到单个pkl文件
    with open(output_file, 'wb') as f:
        pickle.dump(all_features, f)
    
    logging.info(f"CFG特征已保存到 {output_file}, 共 {len(all_features)} 个样本")
    logging.info(f"成功处理: {success_count} 个样本")
    logging.info(f"失败处理: {fail_count} 个样本 (已分配零特征)")
    
    # 打印统计信息
    if all_features:
        feature_shapes = [data['shape'] for data in all_features.values()]
        logging.info(f"特征向量维度: {feature_shapes[0]}")
        
        # 统计各种状态的样本数量
        status_count = {}
        for data in all_features.values():
            status = data.get('status', 'unknown')
            status_count[status] = status_count.get(status, 0) + 1
        
        logging.info(f"样本状态统计: {status_count}")

def main():
    parser = argparse.ArgumentParser(description="从JSONL数据集提取CFG特征并保存为单个pkl文件")
    parser.add_argument("--jsonl_file", default='/home/syh123/workspace/Data_Filter/data/expanded_origen_120k.jsonl', help="JSONL数据集文件路径")
    parser.add_argument("--n_iter", type=int, default=5, help="Weisfeiler-Lehman核迭代次数")
    parser.add_argument("--output_file", type=str, default="/home/syh123/workspace/Data_Filter/data/Origen/cfg_features.pkl", help="输出pkl文件路径")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="设置日志级别")
    
    args = parser.parse_args()
    
    # 应用日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 处理JSONL数据集
    process_jsonl_dataset(args.jsonl_file, args.output_file, args.n_iter)
    
    logging.info("CFG特征提取完成")

if __name__ == "__main__":
    main()