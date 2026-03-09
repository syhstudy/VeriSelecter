from datasets import load_dataset
import json

# 加载数据集
dataset = load_dataset("json", data_files="/home/syh123/workspace/Data_Filter/data/expanded_rtlcoder_10k.jsonl", split="train")

# 随机抽取k%的数据
sampled_dataset = dataset.train_test_split(test_size=0.3222, seed=0)['test']

# 将抽取的数据保存为JSON文件，保持与原数据集相同的格式
output_file = "/home/syh123/workspace/Data_Filter/data/RTLCoder12k/random/random_data_0_15.json"
with open(output_file, 'w', encoding='utf-8') as f:
    # 遍历每个样本，保持JSONL格式（每行一个JSON对象）
    for item in sampled_dataset:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')
# print(sampled_dataset[0])
print(f"已成功抽取 {len(sampled_dataset)} 条数据")