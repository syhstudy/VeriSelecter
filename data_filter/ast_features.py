import pyverilog.vparser.parser as pvparser
import pyverilog.vparser.ast as vast
import logging
import hashlib
import random
import numpy as np
import torch
import argparse
import os
import json
import pickle
import tempfile
from typing import List, Dict, Any, Optional

# 设置固定随机种子以确保确定性行为
random.seed(42)
np.random.seed(42)

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class VerilogASTFeatureExtractor:
    """
    处理Verilog AST解析和特征提取的类
    """

    def __init__(self, depth: int = 2, log_level=logging.INFO):
        self.depth = depth
        logging.getLogger().setLevel(log_level)
        
        # 定义特征向量维度
        self.common_node_types = [
            'Source', 'ModuleDef', 'Decl', 'Input', 'Output', 'Reg', 'Wire',
            'Assign', 'Always', 'Block', 'IfStatement', 'CaseStatement',
            'Case', 'NonblockingSubstitution', 'BlockingSubstitution',
            'Identifier', 'IntConst', 'Partselect', 'Pointer', 'Lconcat',
            'Plus', 'Minus', 'Times', 'Divide', 'Mod', 'Power', 'Ulnot',
            'Unot', 'Uand', 'Unand', 'Uor', 'Unor', 'Uxor', 'Uxnor',
            'Sll', 'Srl', 'Sra', 'LessThan', 'GreaterThan', 'LessEq',
            'GreaterEq', 'Eq', 'NotEq', 'Eql', 'NotEql', 'And', 'Xor',
            'Xnor', 'Or', 'Land', 'Lor', 'Lnot'
        ]
        
        # 计算特征向量总维度
        self.feature_dim = len(self.common_node_types) + 2 + (self.depth + 1)  # 节点类型 + 深度参数 + 总节点数 + 各深度节点数

    def get_zero_features(self) -> torch.Tensor:
        """返回全零特征向量"""
        return torch.zeros(self.feature_dim, dtype=torch.float32)

    def parse_verilog_code(self, code: str) -> Optional[vast.Source]:
        """解析Verilog代码并返回AST"""
        try:
            # 将代码写入临时文件进行解析
            with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            ast, _ = pvparser.parse([temp_file])
            
            # 清理临时文件
            os.unlink(temp_file)
            
            return ast
        except Exception as e:
            logging.info(f"Verilog代码:\n{code}")
            logging.error(f"解析Verilog代码时出错。异常: {e}")
            return None

    def normalize_ast(self, ast: vast.Source) -> vast.Source:
        """对AST进行归一化处理，重命名标识符"""
        def normalize_node(node: Any, name_map: Dict[str, str]) -> None:
            """递归归一化AST节点"""
            if isinstance(node, vast.Identifier):
                if node.name not in name_map:
                    name_map[node.name] = f"var_{len(name_map)}"
                node.name = name_map[node.name]

            # 对子节点进行排序以确保一致性
            for child in sorted(node.children(), key=lambda x: hashlib.sha256(str(x).encode()).hexdigest()):
                normalize_node(child, name_map)

        logging.info("开始AST归一化...")
        name_map = {}
        normalize_node(ast, name_map)
        logging.info("AST归一化完成。")
        return ast

    def extract_ast_features(self, ast: vast.Source) -> torch.Tensor:
        """
        基于节点类型和结构提取AST特征向量
        """
        def extract_node_features(node: Any, feature_dict: Dict[str, int], current_depth: int):
            """递归提取节点类型频率作为特征"""
            if current_depth < 0:
                return
            
            node_type = type(node).__name__
            feature_dict[node_type] = feature_dict.get(node_type, 0) + 1
            
            # 如果深度允许，继续处理子节点
            if current_depth > 0:
                for child in node.children():
                    extract_node_features(child, feature_dict, current_depth - 1)

        logging.info(f"正在提取深度 {self.depth} 的AST特征...")
        
        # 提取节点类型频率
        feature_dict = {}
        extract_node_features(ast, feature_dict, self.depth)
        
        # 创建特征向量
        feature_vector = [feature_dict.get(node_type, 0) for node_type in self.common_node_types]
        
        # 添加结构特征
        feature_vector.append(self.depth)  # 深度参数
        feature_vector.append(sum(feature_dict.values()))  # 总节点数
        
        # 添加按深度的节点计数
        for d in range(self.depth + 1):
            depth_dict = {}
            def count_depth_nodes(node, current_depth):
                if current_depth == d:
                    node_type = type(node).__name__
                    depth_dict[node_type] = depth_dict.get(node_type, 0) + 1
                if current_depth < d:
                    for child in node.children():
                        count_depth_nodes(child, current_depth + 1)
            
            count_depth_nodes(ast, 0)
            feature_vector.append(sum(depth_dict.values()))  # 该深度的总节点数
        
        # 转换为张量
        features = torch.tensor(feature_vector, dtype=torch.float32)
        
        logging.info(f"已提取包含 {len(features)} 维的AST特征向量")
        logging.info(f"features:\n{features}")
        return features

def process_jsonl_dataset(jsonl_file, output_file, depth=2):
    """处理JSONL数据集并提取AST特征，保存为单个pkl文件"""
    extractor = VerilogASTFeatureExtractor(depth=depth)
    
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
                        'status': 'missing_code'
                    }
                    fail_count += 1
                    continue
                
                logging.info(f"正在处理第 {i+1} 条数据...")
                
                # 解析Verilog代码
                ast = extractor.parse_verilog_code(verilog_code)
                if ast is None:
                    logging.error(f"第 {i+1} 条数据解析失败，分配零特征")
                    # 解析失败时分配零特征
                    features = extractor.get_zero_features()
                    logging.info(f"features:\n{features}")
                    sample_id = f"sample_{i+1:05d}"
                    all_features[sample_id] = {
                        'features': features,
                        'shape': features.shape,
                        'verilog_code': verilog_code[:200] + "..." if len(verilog_code) > 200 else verilog_code,
                        'status': 'parse_failed'
                    }
                    fail_count += 1
                    continue
                
                # 归一化AST
                norm_ast = extractor.normalize_ast(ast)
                
                # 提取特征
                features = extractor.extract_ast_features(norm_ast)
                
                # 存储特征
                sample_id = f"sample_{i+1:05d}"
                all_features[sample_id] = {
                    'features': features,
                    'shape': features.shape,
                    'verilog_code': verilog_code[:200] + "..." if len(verilog_code) > 200 else verilog_code,
                    'status': 'success'
                }
                success_count += 1
                
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
    
    # 保存所有特征到单个pkl文件
    with open(output_file, 'wb') as f:
        pickle.dump(all_features, f)
    
    logging.info(f"AST特征已保存到 {output_file}, 共 {len(all_features)} 个样本")
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
    parser = argparse.ArgumentParser(description="从JSONL数据集提取AST特征并保存为单个pkl文件")
    parser.add_argument("--jsonl_file", default='/home/syh123/workspace/Data_Filter/data/expanded_origen_120k.jsonl', help="JSONL数据集文件路径")
    parser.add_argument("--depth", type=int, default=2, help="AST特征提取深度")
    parser.add_argument("--output_file", type=str, default="/home/syh123/workspace/Data_Filter/data_filter/intermediate_data/origen_ast_features.pkl", help="输出pkl文件路径")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="设置日志级别")
    
    args = parser.parse_args()
    
    # 应用日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 处理JSONL数据集
    process_jsonl_dataset(args.jsonl_file, args.output_file, args.depth)
    
    logging.info("AST特征提取完成")

if __name__ == "__main__":
    main()