import argparse
import pickle
import torch
import numpy as np
import logging
from typing import Dict, Any, List
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# 全局变量存储特征维度
ast_features_dim = 0
cfg_features_dim = 0
netlist_features_dim = 0

def load_pkl_features(pkl_path: str) -> Dict[str, Any]:
    """加载pkl特征文件"""
    try:
        with open(pkl_path, 'rb') as f:
            features = pickle.load(f)
        logging.info(f"成功加载 {pkl_path}, 包含 {len(features)} 个样本")
        return features
    except Exception as e:
        logging.error(f"加载 {pkl_path} 失败: {e}")
        return {}

def load_pt_data(pt_path: str) -> Dict[str, Any]:
    """加载pt数据文件"""
    try:
        data = torch.load(pt_path, weights_only=False, map_location=torch.device('cpu'))
        logging.info(f"成功加载 {pt_path}")
        return data
    except Exception as e:
        logging.error(f"加载 {pt_path} 失败: {e}")
        return {}

def save_pt_data(data: Dict[str, Any], output_path: str):
    """保存pt数据文件"""
    try:
        torch.save(data, output_path)
        logging.info(f"成功保存到 {output_path}")
    except Exception as e:
        logging.error(f"保存到 {output_path} 失败: {e}")

def extract_feature_vectors_by_ids(features_dict: Dict[str, Any], sample_ids: List[str]) -> np.ndarray:
    """根据样本ID列表从特征字典中提取特征向量"""
    feature_vectors = []
    missing_count = 0
    
    for sample_id in sample_ids:
        if sample_id in features_dict:
            sample_data = features_dict[sample_id]
            if 'features' in sample_data:
                if isinstance(sample_data['features'], torch.Tensor):
                    feature_vector = sample_data['features'].numpy()
                else:
                    feature_vector = np.array(sample_data['features'])
                feature_vectors.append(feature_vector)
            else:
                logging.warning(f"样本 {sample_id} 中没有找到features字段")
                feature_vectors.append(None)
                missing_count += 1
        else:
            logging.warning(f"未找到样本 {sample_id} 的特征")
            feature_vectors.append(None)
            missing_count += 1
    
    if missing_count > 0:
        logging.warning(f"共有 {missing_count} 个样本缺少特征，将使用零向量填充")
    
    return np.array(feature_vectors) if feature_vectors else np.array([])

def norm(vec: np.ndarray) -> np.ndarray:
    """对向量进行L2归一化"""
    norm_val = np.linalg.norm(vec)
    if norm_val == 0:
        return vec
    return vec / norm_val

def merge_features(ast_features: Dict[str, Any], 
                   cfg_features: Dict[str, Any], 
                   netlist_features: Dict[str, Any], 
                   pt_data: Dict[str, Any]) -> Dict[str, Any]:
    """根据样本ID合并所有特征到pt数据中"""
    
    # 获取样本ID列表
    sample_ids = pt_data.get('sample_ids', [])
    if not sample_ids:
        logging.error("pt_data中没有找到sample_ids字段")
        return pt_data
    
    # 提取各特征文件的特征向量（按样本ID顺序）
    logging.info("正在提取AST特征向量...")
    ast_vectors = extract_feature_vectors_by_ids(ast_features, sample_ids)
    
    logging.info("正在提取CFG特征向量...")
    cfg_vectors = extract_feature_vectors_by_ids(cfg_features, sample_ids)
    
    logging.info("正在提取网表特征向量...")
    netlist_vectors = extract_feature_vectors_by_ids(netlist_features, sample_ids)
    
    # 获取原始高维向量
    original_vectors = pt_data.get('high_dim_vectors', np.array([]))
    
    if original_vectors.size == 0:
        logging.error("原始pt文件中没有找到high_dim_vectors字段")
        return pt_data
    
    # 检查样本数量是否一致
    n_samples = len(original_vectors)
    if len(ast_vectors) != n_samples or len(cfg_vectors) != n_samples or len(netlist_vectors) != n_samples:
        logging.error(f"样本数量不一致! 原始={n_samples}, AST={len(ast_vectors)}, CFG={len(cfg_vectors)}, 网表={len(netlist_vectors)}")
        return pt_data
    
    logging.info(f"将处理 {n_samples} 个样本")
    
    # 合并特征向量
    merged_vectors = []
    zero_vec_count = 0
    
    for i in range(n_samples):
        original_vec = original_vectors[i]
        ast_vec = ast_vectors[i] if i < len(ast_vectors) and ast_vectors[i] is not None else np.array([])
        cfg_vec = cfg_vectors[i] if i < len(cfg_vectors) and cfg_vectors[i] is not None else np.array([])
        netlist_vec = netlist_vectors[i] if i < len(netlist_vectors) and netlist_vectors[i] is not None else np.array([])
        
        # 处理缺失的特征 - 使用零向量
        if ast_vec.size == 0:
            ast_vec = np.zeros(ast_features_dim)
            zero_vec_count += 1
        if cfg_vec.size == 0:
            cfg_vec = np.zeros(cfg_features_dim)
            zero_vec_count += 1
        if netlist_vec.size == 0:
            netlist_vec = np.zeros(netlist_features_dim)
            zero_vec_count += 1
        
        # 拼接所有特征并进行归一化
        model_norm_vec = norm(original_vec)
        diy_norm_vec = norm(np.concatenate([ast_vec, cfg_vec, netlist_vec])) # 原始的三特征拼接
        # diy_norm_vec = norm(np.concatenate([ast_vec, cfg_vec])) # 去掉网表特征后的拼接
        # diy_norm_vec = norm(np.concatenate([cfg_vec, netlist_vec])) # 只保留CFG和网表特征的拼接
        # diy_norm_vec = norm(np.concatenate([ast_vec, netlist_vec])) # 只保留AST和网表特征的拼接
        merged_vec = np.concatenate([model_norm_vec, diy_norm_vec])
        merged_vectors.append(merged_vec)
    
    if zero_vec_count > 0:
        logging.warning(f"有 {zero_vec_count} 个样本使用了零向量填充缺失特征")
    
    # 更新pt数据
    pt_data['high_dim_vectors'] = np.array(merged_vectors)
    
    # 记录特征维度信息
    original_dim = original_vectors[0].shape[0] if original_vectors.size > 0 else 0
    ast_dim = ast_vectors[0].shape[0] if ast_vectors.size > 0 and len(ast_vectors[0].shape) > 0 else 0
    cfg_dim = cfg_vectors[0].shape[0] if cfg_vectors.size > 0 and len(cfg_vectors[0].shape) > 0 else 0
    netlist_dim = netlist_vectors[0].shape[0] if netlist_vectors.size > 0 and len(netlist_vectors[0].shape) > 0 else 0
    merged_dim = pt_data['high_dim_vectors'][0].shape[0]
    
    logging.info(f"特征维度: 原始={original_dim}, AST={ast_dim}, CFG={cfg_dim}, 网表={netlist_dim}, 合并后={merged_dim}")
    
    return pt_data

def main():
    parser = argparse.ArgumentParser(description="合并AST、CFG和网表特征到中间数据文件中")
    parser.add_argument("--ast_pkl", type=str, default='/home/syh123/workspace/Data_Filter/data_filter/intermediate_data/rtlcoder_ast_features.pkl', help='AST特征pkl文件路径')
    parser.add_argument("--cfg_pkl", type=str, default='/home/syh123/workspace/Data_Filter/data_filter/intermediate_data/rtlcoder_cfg_features.pkl', help='CFG特征pkl文件路径')
    parser.add_argument("--netlist_pkl", type=str, default='/home/syh123/workspace/Data_Filter/data_filter/intermediate_data/rtlcoder_netlist_features.pkl', help='网表特征pkl文件路径')
    parser.add_argument("--pt_file", type=str, default='/home/syh123/workspace/Data_Filter/data_filter/intermediate_data/qwen7/rtlcoder_step1_feature_extract_pre.pt', help='输入pt文件路径')
    parser.add_argument("--output_pt", type=str, default='/home/syh123/workspace/Data_Filter/data_filter/intermediate_data/qwen7/rtlcoder_step2_feature_merge_pre_woNetlist.pt', help='输出pt文件路径')
    
    args = parser.parse_args()
    
    logging.info("开始合并特征...")
    
    # 加载所有特征文件
    ast_features = load_pkl_features(args.ast_pkl)
    cfg_features = load_pkl_features(args.cfg_pkl)
    netlist_features = load_pkl_features(args.netlist_pkl)
    pt_data = load_pt_data(args.pt_file)
    
    # 检查是否成功加载所有文件
    if not ast_features or not cfg_features or not netlist_features or not pt_data:
        logging.error("未能成功加载所有必需的文件，退出程序")
        return
    
    # 从第一个特征文件获取维度信息
    global ast_features_dim, cfg_features_dim, netlist_features_dim
    
    ast_sample = next(iter(ast_features.values())) if ast_features else {}
    cfg_sample = next(iter(cfg_features.values())) if cfg_features else {}
    netlist_sample = next(iter(netlist_features.values())) if netlist_features else {}
    
    ast_features_dim = ast_sample.get('features', torch.zeros(0)).shape[0] if 'features' in ast_sample else 0
    cfg_features_dim = cfg_sample.get('features', torch.zeros(0)).shape[0] if 'features' in cfg_sample else 0
    netlist_features_dim = netlist_sample.get('features', torch.zeros(0)).shape[0] if 'features' in netlist_sample else 0
    
    logging.info(f"特征维度: AST={ast_features_dim}, CFG={cfg_features_dim}, 网表={netlist_features_dim}")
    
    # 合并特征
    merged_data = merge_features(ast_features, cfg_features, netlist_features, pt_data)
    
    # 保存结果
    save_pt_data(merged_data, args.output_pt)
    
    logging.info("特征合并完成!")

if __name__ == "__main__":
    main()