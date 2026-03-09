import json
import os
from transformers import AutoTokenizer
from typing import Dict, Any, List
from dataclasses import dataclass
from tqdm import tqdm
from vllm import LLM, SamplingParams


@dataclass
class Problem:
    prompt: str
    module_header: str
    module_name: str


class LocalVerilogGenerator:
    def __init__(self, model_path: str, max_tokens: int = 1024):
        """Initialize the generator with local LLM using vllm"""
        self.model = LLM(model=model_path, gpu_memory_utilization=0.75, max_model_len=2048, trust_remote_code=True)
        self.max_tokens = max_tokens

    def _create_prompt(self, problem: Problem, tokenizer) -> str:
#         return f"""### Instruct: Please act as a professional Verilog designer and provide Verilog code based on the given instruction. {problem.prompt}

# ### Response: ```verilog
# {problem.module_header}
# """
        # messages = [
        #     {"role": "system", "content": "Please act as a professional Verilog designer and provide Verilog code based on the given instruction."},
        #     {"role": "user", "content": problem.prompt.strip()},
        #     # {"role": "assistant", "content": f"```verilog\n{solution.strip()}\n```"}
        # ]
        # text = tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )
        # text = text + f"\n```verilog\n{problem.module_header}\n"

        system_prompt = "Please act as a professional Verilog designer and provide Verilog code based on the given instruction."
        
        # 构建deepseek风格
        # text = f"System: {system_prompt}\n\nUser: {problem.prompt.strip()}\n\nAssistant: ```verilog\n{problem.module_header}\n"
        
        # codellama格式
        # text = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{problem.prompt.strip()} [/INST] ```verilog\n{problem.module_header}\n"

        # seedcoder格式
        text = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{problem.prompt.strip()}<|end|>\n<|assistant|>\n```verilog\n{problem.module_header}\n"

        return text

    def generate_solutions(self, problems: List[Problem], k: int, tokenizer) -> List[Dict[str, Any]]:
        """Generate k solutions for multiple problems using local LLM with vllm"""
        prompts = [self._create_prompt(problem, tokenizer) for problem in problems]
        temperature = 0 if k == 1 else 0.6
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=self.max_tokens,
            n=k  # Generate k samples for each prompt
        )

        # Generate samples for all prompts at once
        outputs = self.model.generate(
            prompts=prompts,
            sampling_params=sampling_params
        )
        all_solutions = []
        # Process each generated output
        for i, output in enumerate(outputs):
            problem = problems[i]
            solutions = []
            for sample in output.outputs:
                generated_text = sample.text
                verilog_code = self._extract_verilog_code(generated_text)
                verilog_code = problem.module_header + '\n    ' + verilog_code
                solutions.append({"solution": verilog_code, "pass": ""})
            result = {
                "module_name": problem.module_name,
                "solutions": solutions
            }
            all_solutions.append(result)

        return all_solutions

    def _extract_verilog_code(self, content: str) -> str:
        if 'endmodule' in content:
            content = content.split('endmodule')[0].strip() + '\nendmodule'
        if '```' in content:
            result = content.split('```')[0].strip()
        else:
            result = content
        return result

def generate_solutions(config):
    # Initialize the local model generator
    generator = LocalVerilogGenerator(model_path=config["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])

    all_problems = []
    # 加载问题数据
    with open(config["prompt_file"], "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            problem = Problem(
                prompt=item.get("prompt", ""),
                module_header=item.get("module_header", ""),
                module_name=item.get("module_name", ""),
            )
            all_problems.append(problem)

    print(f"Generating {config['k']} solutions for {len(all_problems)} problems...")

    # Generate solutions
    all_solutions = generator.generate_solutions(all_problems, config["k"], tokenizer)

    # 保存结果
    with open(config["output_file"], "w", encoding="utf-8") as f:
        json.dump(all_solutions, f, ensure_ascii=False, indent=4)

    print(f"All solutions generated and saved to {config['output_file']}")


if __name__ == "__main__":
    # # qwen15 Configuration
    # config = {
    #     "model_path": "/home/syh123/workspace/Data_Filter/save_models/save_model_qwen15_rtlcoder_el2n_15",  # Path to your local HF model
    #     "model_name": "qwen2.5_coder_1.5b",  # Name to use in the JSON output
    #     "prompt_file": "/home/syh123/workspace/Data_Filter/eval/verilogeval_v2/problems_verilogeval_v2.jsonl",
    #     "output_file": "/home/syh123/workspace/Data_Filter/eval/verilogeval_v2/eval_file/qwen15/eval_rtlcoder_el2n_15.json",
    #     "k": 5  # Number of solutions to generate per problem
    # }

    # # qwen7 Configuration
    # config = {
    #     "model_path": "/home/syh123/workspace/Data_Filter/save_models/save_model_qwen7_rtlcoder_el2n_20",  # Path to your local HF model
    #     "model_name": "qwen2.5_coder_7b",  # Name to use in the JSON output
    #     "prompt_file": "/home/syh123/workspace/Data_Filter/eval/verilogeval_v2/problems_verilogeval_v2.jsonl",
    #     "output_file": "/home/syh123/workspace/Data_Filter/eval/verilogeval_v2/eval_file/qwen7/eval_rtlcoder_el2n_20.json",
    #     "k": 1  # Number of solutions to generate per problem
    # }

    # # ds13 Configuration
    # config = {
    #     "model_path": "/home/syh123/workspace/Data_Filter/save_models/save_model_ds13_rtlcoder_pre_ratio_10",  # Path to your local HF model
    #     "model_name": "deepseek_coder_1.3b",  # Name to use in the JSON output
    #     "prompt_file": "/home/syh123/workspace/Data_Filter/eval/verilogeval_v2/problems_verilogeval_v2.jsonl",
    #     "output_file": "/home/syh123/workspace/Data_Filter/eval/verilogeval_v2/eval_file/ds13/eval_rtlcoder_pre_ratio_10.json",
    #     "k": 5  # Number of solutions to generate per problem
    # }

    # # ds67 Configuration
    # config = {
    #     "model_path": "/home/syh123/workspace/Data_Filter/save_models/save_model_ds67_rtlcoder_pre_ratio_25",  # Path to your local HF model
    #     "model_name": "deepseek_coder_6.7b",  # Name to use in the JSON output
    #     "prompt_file": "/home/syh123/workspace/Data_Filter/eval/verilogeval_v2/problems_verilogeval_v2.jsonl",
    #     "output_file": "/home/syh123/workspace/Data_Filter/eval/verilogeval_v2/eval_file/ds67/eval_rtlcoder_pre_ratio_25.json",
    #     "k": 1  # Number of solutions to generate per problem
    # }

    # # cl7 Configuration
    # config = {
    #     "model_path": "/home/syh123/workspace/Data_Filter/save_models/save_model_cl7_rtlcoder_el2n_25",  # Path to your local HF model
    #     "model_name": "CodeLlama_7b",  # Name to use in the JSON output
    #     "prompt_file": "/home/syh123/workspace/Data_Filter/eval/verilogeval_v2/problems_verilogeval_v2.jsonl",
    #     "output_file": "/home/syh123/workspace/Data_Filter/eval/verilogeval_v2/eval_file/cl7/eval_rtlcoder_el2n_25.json",
    #     "k": 5  # Number of solutions to generate per problem
    # }

    # sc8 Configuration
    config = {
        "model_path": "/home/syh123/workspace/Data_Filter/save_models/save_model_sc8_rtlcoder_el2n_25",  # Path to your local HF model
        "model_name": "Seed-Coder_8B",  # Name to use in the JSON output
        "prompt_file": "/home/syh123/workspace/Data_Filter/eval/verilogeval_v2/problems_verilogeval_v2.jsonl",
        "output_file": "/home/syh123/workspace/Data_Filter/eval/verilogeval_v2/eval_file/sc8/eval_rtlcoder_el2n_25.json",
        "k": 10  # Number of solutions to generate per problem
    }

    # Run only the generation part
    generate_solutions(config)