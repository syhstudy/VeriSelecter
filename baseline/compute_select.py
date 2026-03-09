from datasets import load_dataset
from code_grand_score import compute_grand_scores_and_save, sample_from_scored_dataset
from code_el2n_score import compute_el2n_scores_and_save
import json

# 加载数据集
dataset = load_dataset("json", data_files="/home/syh123/workspace/Data_Filter/data/expanded_rtlcoder_10k.jsonl", split="train")

# 模型路径
model_name = "/home/syh123/workspace/models/Qwen2.5-Coder-1.5B"

# 计算EL2N分数并保存
el2n_scores_file = "/home/syh123/workspace/Data_Filter/baseline/temp/dataset_with_el2n_scores.json"
# el2n_scores = compute_el2n_scores_and_save(
#     model_name=model_name,
#     dataset=dataset,
#     output_file=el2n_scores_file,
#     batch_size=1
# )

# 加载带有EL2N分数的数据集
with open(el2n_scores_file, 'r', encoding='utf-8') as f:
    dataset_with_el2n_scores = json.load(f)

# 采样不同比例的数据
for percent in [15, 20, 25]:
    output_file = f"/home/syh123/workspace/Data_Filter/data/RTLCoder12k/other/el2n_{percent}.json"
    sampled_data, indices = sample_from_scored_dataset(
        dataset_with_el2n_scores,
        score_type="el2n",
        k_percent=percent,
        output_file=output_file
    )
    print(f"Sampled {percent}% data: {len(sampled_data)} examples")

# # 计算GraNd分数并保存
# grand_scores_file = "/home/syh123/workspace/Data_Filter/baseline/temp/dataset_with_grand_scores.json"
# grand_scores = compute_grand_scores_and_save(
#     model_name=model_name,
#     dataset=dataset,
#     output_file=grand_scores_file,
#     batch_size=2
# )

# # 同样处理GraNd分数
# with open(grand_scores_file, 'r', encoding='utf-8') as f:
#     dataset_with_grand_scores = json.load(f)

# for percent in [15, 20, 25]:
#     output_file = f"/home/syh123/workspace/Data_Filter/data/RTLCoder12k/other/grand_{percent}.json"
#     sampled_data, indices = sample_from_scored_dataset(
#         dataset_with_grand_scores,
#         score_type="grand",
#         k_percent=percent,
#         output_file=output_file
#     )
#     print(f"Sampled {percent}% data: {len(sampled_data)} examples")