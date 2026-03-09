import os
import json
import torch
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn

# 初始化损失函数
log_softmax = nn.LogSoftmax(dim=-1)
nll_loss = nn.NLLLoss(reduction='none')

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def parse_args():
    parser = argparse.ArgumentParser()
    # 数据路径参数
    parser.add_argument("--json_data_path", type=str, default='/home/syh123/workspace/Data_Filter/data/expanded_rtlcoder_10k.jsonl', help='原始JSONL数据路径')
    parser.add_argument("--pt_data_path", type=str, default='/home/syh123/workspace/Data_Filter/data_filter/intermediate_data/sc8/rtlcoder_step1_feature_extract_pre.pt', help='中间数据文件路径')
    parser.add_argument("--model_path", type=str, default='/home/syh123/workspace/Data_Filter/save_models/save_model_sc8_rtlcoder_pre', help='模型路径')
    
    # 处理参数
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    
    args = parser.parse_args()
    return args

def load_verilog_data(data_path):
    """加载 Verilog 数据集"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data

def format_verilog_prompt(instruction, response=None, canonical_solution=None):
    """格式化 Verilog 提示 - 使用deepseek-coder的通用模板"""
    if canonical_solution:
        solution = canonical_solution
    elif response and len(response) > 0:
        solution = response[0]
    else:
        solution = ""
    
    # 构建系统提示
    system_prompt = "Please act as a professional Verilog designer and provide Verilog code based on the given instruction."
         
    # deepseek格式
    # conversation = f"System: {system_prompt}\n\nUser: {instruction}\n\nAssistant: ```verilog\n{solution}\n```"
    # prompt_only = f"System: {system_prompt}\n\nUser: {instruction}\n\nAssistant:```verilog\n"

    # codellama格式
    # conversation = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction.strip()} [/INST] ```verilog\n{solution}\n"
    # prompt_only = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction.strip()} [/INST] ```verilog\n"

    # seedcoder格式
    conversation = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{instruction.strip()}<|end|>\n<|assistant|>\n```verilog\n{solution}\n"
    prompt_only = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{instruction.strip()}<|end|>\n<|assistant|>\n```verilog\n"
    
    return conversation, prompt_only

def get_perplexity_and_embedding_whole_text(tokenizer, model, text, max_length):
    """获取整个文本的困惑度和嵌入（用于聚类）"""
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    with torch.no_grad(): 
        outputs = model(input_ids, labels=input_ids.contiguous())
    loss = outputs.loss
    perplexity = torch.exp(loss)

    hidden_states = outputs.hidden_states
    embeddings = hidden_states[-1]
    sentence_embedding = embeddings.mean(dim=1)

    return perplexity.to('cpu'), sentence_embedding.to('cpu')

def get_perplexity_and_embedding_part_text(tokenizer, model, text, target_span, max_length):
    """获取部分文本的困惑度和token-wise损失（用于计算mean_rate）"""
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    start_index = text.rfind(target_span)
    start_token = len(tokenizer.encode(text[:start_index]))
    end_token = input_ids.shape[1]

    labels = input_ids.clone()
    labels[0, :start_token] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)

    loss = outputs.loss
    perplexity = torch.exp(loss)

    losses = []
    logits = outputs.logits
    for i in range(1, end_token):
        log_prob_dist = log_softmax(logits[0, i-1])
        true_token = input_ids[0, i]
        token_loss = nll_loss(log_prob_dist.unsqueeze(0), true_token.unsqueeze(0))
        losses.append(token_loss.item())

    return perplexity.to('cpu'), 0, losses

def main():
    args = parse_args()
    print("特征提取参数:", args)
    
    # 加载模型和分词器
    print("Loading model and tokenizer...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            device_map="auto", 
            cache_dir='../cache', 
            output_hidden_states=True,
            torch_dtype=torch.float16
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir='../cache')
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir='../cache')
    
    # 设置 padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    
    # 加载数据
    print("Loading data...")
    json_data = load_verilog_data(args.json_data_path)
    
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx != -1 else len(json_data)
    sampled_data = json_data[start_idx:end_idx]
    
    print(f"Processing {len(sampled_data)} samples...")
    
    # 生成样本ID映射 - 使用全局唯一ID
    sample_ids = [f"sample_{start_idx + i + 1:05d}" for i in range(len(sampled_data))]
    
    # 步骤1: 提取特征进行聚类
    print("步骤1: 提取特征进行聚类...")
    emb_list = []
    ppl_list = []
    
    for i in tqdm(range(len(sampled_data))):
        data_i = sampled_data[i]
        
        # 从数据中提取字段
        instruction = data_i['Instruction']
        response = data_i.get('Response', [])
        canonical_solution = data_i.get('canonical_solution', '')
        
        # 格式化提示 - 只使用指令部分进行聚类
        _, instruct_i = format_verilog_prompt(
            instruction, response, canonical_solution
        )
        
        # 获取指令的困惑度和嵌入
        ppl_ins_alone, emb_ins_alone = get_perplexity_and_embedding_whole_text(
            tokenizer, model, instruct_i, args.max_length
        )
        
        emb_list.append(emb_ins_alone)
        ppl_list.append(ppl_ins_alone.item())
    
    high_dim_vectors = torch.cat(emb_list, 0).numpy()
    
    # 步骤2: 计算所有样本的mean_rate指标
    print("步骤2: 计算所有样本的mean_rate指标...")
    mean_rate_list = []  # 现在存储 (mean_rate, sample_id)
    
    for i in tqdm(range(len(sampled_data)), desc="Calculating mean rates"):
        data_i = sampled_data[i]
        sample_id = sample_ids[i]  # 获取对应的样本ID
        
        # 从数据中提取字段
        instruction = data_i['Instruction']
        response = data_i.get('Response', [])
        canonical_solution = data_i.get('canonical_solution', '')
        
        # 格式化提示
        whole_text, instruct_i = format_verilog_prompt(
            instruction, response, canonical_solution
        )
        
        # 获取输出文本
        if canonical_solution:
            output_i = canonical_solution
        elif response and len(response) > 0:
            output_i = response[0]
        else:
            output_i = ""
        
        direct_answer_text = f"Assistant: ```verilog\n{output_i}\n```"
        
        # 计算条件和非条件损失
        instruct_i_input_ids = tokenizer.encode(
            instruct_i, return_tensors="pt", truncation=True, max_length=args.max_length
        ).to(device)
        instruct_i_len = instruct_i_input_ids.shape[1] 
    
        _, _, loss_list_alone = get_perplexity_and_embedding_part_text(
            tokenizer, model, direct_answer_text, output_i, args.max_length - instruct_i_len + 4
        )
        _, _, loss_list_condition = get_perplexity_and_embedding_part_text(
            tokenizer, model, whole_text, output_i, args.max_length
        )
        
        # 计算mean_rate
        if len(loss_list_alone) > 0 and len(loss_list_condition) > 0:
            mean_1 = np.mean(loss_list_alone)  # 非条件损失
            mean_2 = np.mean(loss_list_condition)  # 条件损失
            
            if mean_1 > 0:  # 避免除零错误
                mean_rate = mean_2 / mean_1
                # 只保留mean_rate <= 1的样本
                if mean_rate <= 1:
                    mean_rate_list.append((mean_rate, sample_id))  # 存储sample_id而不是索引
    
    # 按mean_rate排序（从小到大）
    mean_rate_list.sort()
    
    # 保存中间结果
    intermediate_data = {
        'high_dim_vectors': high_dim_vectors,
        'ppl_list': ppl_list,
        'mean_rate_list': mean_rate_list,  # 现在包含sample_id
        'sample_ids': sample_ids,  # 所有样本的ID映射
        'sampled_data': sampled_data,
        'original_data_info': {
            'json_data_path': args.json_data_path,
            'start_idx': args.start_idx,
            'end_idx': end_idx,
            'total_samples': len(sampled_data)
        }
    }
    
    torch.save(intermediate_data, args.pt_data_path)
    print(f"中间数据已保存到: {args.pt_data_path}")
    print(f"特征维度: {high_dim_vectors.shape}")
    print(f"有效样本数量 (mean_rate <= 1): {len(mean_rate_list)}")
    print(f"样本ID范围: {sample_ids[0]} 到 {sample_ids[-1]}")
    print(f"特征提取完成!")

if __name__ == '__main__':
    main()