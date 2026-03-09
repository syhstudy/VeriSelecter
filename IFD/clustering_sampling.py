import os
import json
import torch
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

def parse_args():
    parser = argparse.ArgumentParser()
    # 数据路径参数
    parser.add_argument("--pt_data_path", type=str, default='/home/syh123/workspace/Data_Filter/data/RTLCoder12k/qwen2.5_coder_1.5b/intermediate_data.pt', help='中间数据文件路径')
    parser.add_argument("--json_save_path", type=str, default='/home/syh123/workspace/Data_Filter/data/RTLCoder12k/qwen2.5_coder_1.5b/cherry_merge_test.jsonl', help='保存结果路径')
    
    # 聚类参数
    parser.add_argument("--cluster_method", type=str, default='kmeans')
    parser.add_argument("--kmeans_num_clusters", type=int, default=150)
    
    # 采样参数
    parser.add_argument("--top_k", type=int, default=5, help='每个簇中采样的样本数量')
    
    args = parser.parse_args()
    return args

def do_clustering(args, high_dim_vectors):
    """执行聚类"""
    clustering_algorithm = args.cluster_method
    if clustering_algorithm == 'kmeans':
        print(f"使用K-means聚类，簇数量: {args.kmeans_num_clusters}")
        clustering = KMeans(n_clusters=args.kmeans_num_clusters, random_state=0).fit(high_dim_vectors)
    
    return clustering

def sample_topk_by_mean_rate(cluster_labels, mean_rate_list, top_k):
    """在每个簇中采样mean_rate排名靠前的top_k个样本"""
    num_clusters = len(np.unique(cluster_labels))
    
    # 获取每个簇的样本索引
    cluster_indices = {i: [] for i in range(num_clusters)}
    
    # 将样本按簇分组
    for mean_rate, idx in mean_rate_list:
        cluster_id = cluster_labels[idx]
        cluster_indices[cluster_id].append((mean_rate, idx))
    
    # 在每个簇中选择top_k个mean_rate最大的样本
    selected_samples = {}
    for cluster_id in range(num_clusters):
        cluster_samples = cluster_indices[cluster_id]
        if cluster_samples:  # 只处理有样本的簇
            # 按mean_rate排序（从大到小）
            cluster_samples.sort()
            # 选择前top_k个
            topk_samples = cluster_samples[:top_k]
            selected_samples[cluster_id] = [idx for _, idx in topk_samples]
        else:
            selected_samples[cluster_id] = []
    
    return selected_samples

def main():
    args = parse_args()
    print("聚类和采样参数:", args)
    
    # 检查中间文件是否存在
    if not os.path.exists(args.pt_data_path):
        print(f"错误: 中间文件 {args.pt_data_path} 不存在")
        print("请先运行 feature_extraction.py 生成中间数据")
        return
    
    # 加载中间数据
    print(f"加载中间数据从: {args.pt_data_path}")
    # intermediate_data = torch.load(args.pt_data_path)
    intermediate_data = torch.load(args.pt_data_path, weights_only=False, map_location=torch.device('cpu'))
    
    # 从中间数据中提取变量
    high_dim_vectors = intermediate_data['high_dim_vectors']
    mean_rate_list = intermediate_data['mean_rate_list']
    sampled_data = intermediate_data['sampled_data']
    original_info = intermediate_data.get('original_data_info', {})
    
    print(f"数据信息: {original_info}")
    print(f"特征维度: {high_dim_vectors.shape}")
    print(f"有效样本数量 (mean_rate <= 1): {len(mean_rate_list)}")
    
    # 步骤3: 执行聚类
    print(f"步骤3: 执行聚类...")
    clustering = do_clustering(args, high_dim_vectors)
    cluster_labels = clustering.labels_
    
    # 步骤4: 在每个簇中采样top_k个mean_rate最小的样本
    print(f"步骤4: 在每个簇中采样top_k个mean_rate最小的样本...")
    selected_samples = sample_topk_by_mean_rate(cluster_labels, mean_rate_list, args.top_k)
    
    # 收集所有选中的样本
    all_selected_indices = []
    for cluster_id, indices in selected_samples.items():
        all_selected_indices.extend(indices)
    
    all_selected_indices = sorted(set(all_selected_indices))  # 去重并排序
    
    # 生成新的数据集
    new_data = [sampled_data[idx] for idx in all_selected_indices]
    print(f'最终采样数据长度: {len(new_data)}')
    print(f'从 {len(np.unique(cluster_labels))} 个簇中采样')
    
    # 保存结果
    print(f"保存结果到: {args.json_save_path}")
    with open(args.json_save_path, "w", encoding='utf-8') as fw:
        for item in new_data:
            fw.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 输出统计信息
    print("\n详细统计信息:")
    total_samples = 0
    non_empty_clusters = 0
    for cluster_id in sorted(selected_samples.keys()):
        sample_count = len(selected_samples[cluster_id])
        total_samples += sample_count
        if sample_count > 0:
            non_empty_clusters += 1
            # print(f"簇 {cluster_id}: {sample_count} 个样本")
    
    # 输出mean_rate统计信息
    if mean_rate_list and all_selected_indices:
        selected_mean_rates = [mean_rate for mean_rate, idx in mean_rate_list if idx in all_selected_indices]
        print(f"\n选中样本的mean_rate统计:")
        print(f"最小 mean_rate: {min(selected_mean_rates):.4f}")
        print(f"最大 mean_rate: {max(selected_mean_rates):.4f}")
        print(f"平均 mean_rate: {np.mean(selected_mean_rates):.4f}")
        print(f"中位数 mean_rate: {np.median(selected_mean_rates):.4f}")
    
    print(f"\n总结:")
    print(f"有样本的簇数量: {non_empty_clusters}")
    print(f"总采样样本数: {total_samples}")
    print(f"采样完成!")

if __name__ == '__main__':
    main()