import torch
from unsloth import FastLanguageModel
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 配置参数
base_model_path = "/home/syh123/workspace/models/deepseek-coder-1.3b"
lora_model_path = "/home/syh123/workspace/Data_Filter_all/lora_model/results_ds13_Resyn27k_all/checkpoint-1659"
output_model_path = "/home/syh123/workspace/Data_Filter_all/save_models/save_model_ds13_Resyn27k_all"

def merge_lora_to_base_model():
    print("开始加载基础模型...")
    
    # 加载基础模型和tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    print("基础模型加载完成")
    
    # 加载LoRA适配器
    print("开始加载LoRA适配器...")
    model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        torch_dtype=torch.float16
    )
    
    print("LoRA适配器加载完成")
    
    # 合并LoRA权重到基础模型
    print("开始合并LoRA权重...")
    merged_model = model.merge_and_unload()
    
    print("LoRA权重合并完成")
    
    # 保存合并后的模型
    print("开始保存合并后的模型...")
    merged_model.save_pretrained(
        output_model_path,
        safe_serialization=True
    )
    
    # 保存tokenizer
    tokenizer.save_pretrained(output_model_path)
    
    print(f"模型已成功保存到: {output_model_path}")
    
    # 验证模型是否可以正常加载
    print("验证模型加载...")
    test_model = AutoModelForCausalLM.from_pretrained(
        output_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    test_tokenizer = AutoTokenizer.from_pretrained(
        output_model_path,
        trust_remote_code=True
    )
    
    print("模型验证成功！合并后的模型可以正常加载和使用。")

# 使用Unsloth的替代方法（如果上面的方法有问题）
def merge_lora_with_unsloth():
    print("使用Unsloth方法合并模型...")
    
    # 使用Unsloth加载基础模型
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=4096 * 2,
        dtype=None,
        load_in_4bit=False,
    )
    
    print("基础模型加载完成")
    
    # 加载LoRA权重
    model.load_adapter(lora_model_path)
    
    print("LoRA适配器加载完成")
    
    # 合并模型
    merged_model = model.merge_and_unload()
    
    print("模型合并完成")
    
    # 保存合并后的模型
    merged_model.save_pretrained(
        output_model_path,
        safe_serialization=True
    )
    
    tokenizer.save_pretrained(output_model_path)
    
    print(f"模型已成功保存到: {output_model_path}")

if __name__ == "__main__":
    try:
        # 首先尝试标准方法
        merge_lora_to_base_model()
    except Exception as e:
        print(f"标准方法失败: {e}")
        print("尝试使用Unsloth方法...")
        try:
            merge_lora_with_unsloth()
        except Exception as e2:
            print(f"Unsloth方法也失败: {e2}")
            
            # 最后尝试：手动合并
            print("尝试手动合并方法...")
            from peft import get_peft_model_state_dict, set_peft_model_state_dict
            
            # 加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=True
            )
            
            # 加载LoRA权重
            lora_weights = torch.load(f"{lora_model_path}/adapter_model.safetensors")
            
            # 手动将LoRA权重应用到基础模型
            for name, param in base_model.named_parameters():
                lora_name = name.replace("base_model.model.", "")
                if lora_name in lora_weights:
                    param.data += lora_weights[lora_name]
            
            # 保存合并后的模型
            base_model.save_pretrained(output_model_path, safe_serialization=True)
            tokenizer.save_pretrained(output_model_path)
            
            print(f"手动合并完成，模型保存到: {output_model_path}")