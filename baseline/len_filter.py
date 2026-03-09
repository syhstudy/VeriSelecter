import json
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("json", data_files="/home/syh123/workspace/Data_Filter/data/expanded_rtlcoder_10k.jsonl", split="train")

# 计算每个样本的总长度（Instruction + Response + canonical_solution + test）
def calculate_total_length(example):
    # 计算总字符数
    total_length = len(example['Instruction']) + len(example['canonical_solution'])
    return {'total_length': total_length}

# 添加长度信息到数据集
dataset_with_length = dataset.map(calculate_total_length)

# 按长度排序
# sorted_dataset = dataset_with_length.sort('total_length')
sorted_dataset = dataset_with_length.sort('total_length', reverse=True)

# 抽取k%的数据 - 这里提供几种不同的策略
total_size = len(dataset)
num = 26532
sample_k = 0.2

# 策略1: 抽取中等长度的20%（去掉最短和最长的各40%）
def get_middle_20_percent(dataset):
    start_idx = int(num * (1-sample_k)/2)  # 去掉前40%
    end_idx = int(num * (1+sample_k)/2)    # 保留中间的20%
    return dataset.select(range(start_idx, end_idx))

# 策略2: 抽取最长的20%
def get_longest_20_percent(dataset):
    start_idx = int(num * (1-sample_k))  # 取最长的20%
    return dataset.select(range(start_idx, total_size))

# 策略3: 抽取最短的20%
def get_shortest_20_percent(dataset):
    end_idx = int(num * sample_k)    # 取最短的20%
    return dataset.select(range(0, end_idx))

# 选择一种策略
# sampled_dataset = get_middle_20_percent(sorted_dataset)  # 中等长度
# sampled_dataset = get_longest_20_percent(sorted_dataset)    # 最长
sampled_dataset = get_shortest_20_percent(sorted_dataset) # 最短

# 移除添加的长度字段
sampled_dataset = sampled_dataset.remove_columns(['total_length'])

# 将抽取的数据保存为JSON文件，保持与原数据集相同的格式
output_file = "/home/syh123/workspace/Data_Filter/data/RTLCoder12k/len_filter/len_long_20.json"
with open(output_file, 'w', encoding='utf-8') as f:
    # 遍历每个样本，保持JSONL格式（每行一个JSON对象）
    for item in sampled_dataset:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

print(f"已成功抽取 {len(sampled_dataset)} 条数据，保存到 {output_file}")
print(f"数据集总大小: {len(dataset)}")