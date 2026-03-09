from unsloth import FastLanguageModel
import torch
max_seq_length = 4096 * 2
dtype = None
load_in_4bit = False

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/home/syh123/workspace/models/Seed-Coder-8B-Base",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
print("model init ...")

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)
print("lora init ...")

from datasets import load_dataset
dataset = load_dataset("json", data_files="/home/syh123/workspace/Data_Filter/data/RTLCoder12k/sc8/pre_ratio_10.jsonl", split="train")
print("dataset init ...")

def formatting_func(examples):
    texts = []
    
    # 统一处理指令字段
    if isinstance(examples["Instruction"], list):
        # 批量数据
        instructions = examples["Instruction"]
        canonical_solutions = examples.get("canonical_solution", [""] * len(instructions))
        responses = examples.get("Response", [[]] * len(instructions))
    else:
        # 单个样本
        instructions = [examples["Instruction"]]
        canonical_solutions = [examples.get("canonical_solution", "")]
        responses = [examples.get("Response", [])]
    
    for i in range(len(instructions)):
        instruction = instructions[i]
        canonical_solution = canonical_solutions[i] if i < len(canonical_solutions) else ""
        response = responses[i] if i < len(responses) else []
        
        # 确保 instruction 是字符串
        if isinstance(instruction, list):
            instruction = instruction[0] if instruction else ""
        
        # 确保 instruction 是字符串类型
        instruction = str(instruction) if instruction is not None else ""
        
        # 选择解决方案：优先使用 canonical_solution
        if canonical_solution and str(canonical_solution).strip():
            solution = canonical_solution
        elif response and isinstance(response, list) and len(response) > 0:
            solution = response[0] if response[0] is not None else ""
        else:
            solution = ""
        
        # 确保 solution 是字符串
        solution = str(solution) if solution is not None else ""
        
        # # qwen格式
        # messages = [
        #     {"role": "system", "content": "Please act as a professional Verilog designer and provide Verilog code based on the given instruction."},
        #     {"role": "user", "content": instruction.strip()},
        #     {"role": "assistant", "content": f"```verilog\n{solution.strip()}\n```"}
        # ]
        # text = tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=False
        # )

        system_prompt = "Please act as a professional Verilog designer and provide Verilog code based on the given instruction."

        # deepseek格式
        # text = f"System: {system_prompt}\n\nUser: {instruction.strip()}\n\nAssistant: ```verilog\n{solution.strip()}\n```"

        # codellama格式
        # text = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction.strip()} [/INST] ```verilog\n{solution.strip()}\n```</s>"

        # seedcoder格式
        text = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{instruction.strip()}<|end|>\n<|assistant|>\n```verilog\n{solution.strip()}\n```<|end|>"

        texts.append(text)
    
    return texts

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    formatting_func = formatting_func,  # 使用格式化函数
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = True,  # 对于长序列，先设置为 False 测试
    args = TrainingArguments(
        per_device_train_batch_size = 2,  # RTX 3090 24GB 可以适当调整
        gradient_accumulation_steps = 8,   # 有效批次大小 = 2 * 8 = 16
        warmup_ratio = 0.1,
        num_train_epochs = 3,
        save_strategy="epoch",
        output_dir="/home/syh123/workspace/Data_Filter/lora_model/results_sc8_rtlcoder_pre_ratio_10",
        learning_rate = 2e-4,  # 稍微提高学习率
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none",
        dataloader_pin_memory = False,
    ),
)

# 打印 GPU 信息
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

print("开始训练...")
trainer_stats = trainer.train()

print("保存模型...")
model.save_pretrained_merged("/home/syh123/workspace/Data_Filter/save_models/save_model_sc8_rtlcoder_pre_ratio_10", tokenizer, save_method = "merged_16bit",)
print("训练完成！")
