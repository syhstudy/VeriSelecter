import torch
import json
import numpy as np
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_data_path", type=str, default='/home/syh123/workspace/Data_Filter/data/qwen2.5_coder_1.5b/rtlcoder_10k_cherry.pt')
    parser.add_argument("--json_data_path", type=str, default='/home/syh123/workspace/Data_Filter/data/expanded_rtlcoder_10k.jsonl')
    parser.add_argument("--json_save_path", type=str, default='/home/syh123/workspace/Data_Filter/data/qwen2.5_coder_1.5b/rtlcoder_10k_cherry9.jsonl')
    parser.add_argument("--model_name_or_path", type=str, default='/home/syh123/workspace/Data_Filter/save_model_10k_pre')
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--sample_rate", type=float, default=0.09)
    parser.add_argument("--sample_number", type=int, default=0)
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

def format_verilog_prompt(tokenizer, instruction, response=None, canonical_solution=None):
    """格式化 Verilog 提示 - 使用固定模板"""
    # 优先使用 canonical_solution，如果没有则使用 response 的第一个
    if canonical_solution:
        solution = canonical_solution
    elif response and len(response) > 0:
        solution = response[0]
    else:
        solution = ""
    
    # 使用固定模板
    messages = [
        {"role": "system", "content": "Please act as a professional Verilog designer and provide Verilog code based on the given instruction."},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": f"```verilog\n{solution}\n```"}
    ]
    whole_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    prompt_text = whole_text[:whole_text.find("<|im_start|>assistant")].strip()    
    return whole_text, prompt_text

def main():
    args = parse_args()
    print(args)

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        print(f"成功加载tokenizer from: {args.model_name_or_path}")
        print(f"Tokenizer 类型: {type(tokenizer)}")
    except Exception as e:
        print(f"加载tokenizer失败: {e}")
        return

    # 如果tokenizer没有pad_token，设置一个
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("设置pad_token为eos_token")

    pt_data = torch.load(args.pt_data_path, map_location=torch.device('cpu'))
    
    # 加载 Verilog 数据
    json_data = load_verilog_data(args.json_data_path)

    mean_rate_list = []
    mean_list_1 = []
    mean_list_2 = []
    
    for i in tqdm(range(len(pt_data))):
        pt_data_i = pt_data[i]
        loss_1_list = pt_data_i['token_loss'][1]
        loss_2_list = pt_data_i['token_loss'][2]

        json_data_i = json_data[i]
        
        # 从 Verilog 数据中提取字段
        instruction = json_data_i['Instruction']
        response = json_data_i.get('Response', [])
        canonical_solution = json_data_i.get('canonical_solution', '')
        
        # 格式化提示 - 使用固定模板
        whole_text, instruct_i = format_verilog_prompt(
            tokenizer, instruction, response, canonical_solution
        )
        
        # 获取输出文本（Verilog 代码）
        if canonical_solution:
            output_i = canonical_solution
        elif response and len(response) > 0:
            output_i = response[0]
        else:
            output_i = ""
        
        direct_answer_text = output_i

        # Tokenize the input text
        instruct_i_input_ids = tokenizer.encode(instruct_i, return_tensors="pt", truncation=True, max_length=args.max_length).to('cpu')
        instruct_i_len = instruct_i_input_ids.shape[1] 

        def get_loss_part_text(tokenizer, text, target_span, max_length, loss_list_):
            input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to('cpu')
            start_index = text.rfind(target_span)
            text_temp = text[:start_index]
            token_id_temp = tokenizer.encode(text_temp)
            start_token = len(token_id_temp) 
            end_token_real = input_ids.shape[1]

            loss_list = loss_list_[start_token-1:end_token_real-1] 

            return end_token_real - start_token , input_ids[0][start_token:end_token_real], np.array(loss_list)
        
        if args.max_length - instruct_i_len > 0:
            len_1, token_ids_1, loss_list_1 = get_loss_part_text(tokenizer, direct_answer_text, output_i, args.max_length-instruct_i_len+4, loss_1_list)
            len_2, token_ids_2, loss_list_2 = get_loss_part_text(tokenizer, whole_text, output_i, args.max_length, loss_2_list)

            if len_1 <= 0 or len_2 <= 0:
                continue

            if instruct_i_len + len_1 > args.max_length:
                continue

            mean_1 = loss_list_1.mean()
            mean_2 = loss_list_2.mean()
            mean_rate = mean_2/mean_1
            if mean_rate > 1: 
                continue

            mean_rate_list.append((mean_rate, i))
            mean_list_1.append((mean_1, i))
            mean_list_2.append((mean_2, i))
        else:
            continue

    print('Do Rate')
    mean_rate_list = sorted(mean_rate_list)
    if args.sample_number == 0:
        args.sample_number = int(len(mean_rate_list) * args.sample_rate)
    
    mean_rate_list_id = [i for i in range(len(mean_rate_list))][-args.sample_number:]
    mean_rate_list_id_sample = [mean_rate_list[id][1] for id in mean_rate_list_id]
    mean_rate_list_id_sample = sorted(mean_rate_list_id_sample)

    new_data = [json_data[idx] for idx in mean_rate_list_id_sample]
    print('New data len:', len(new_data))
    
    # 保存为 JSONL 格式
    with open(args.json_save_path, "w", encoding='utf-8') as fw:
        for item in new_data:
            fw.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()