from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained('/home/syh123/workspace/models/Qwen2.5-Coder-1.5B')
tokenizer = AutoTokenizer.from_pretrained('/home/syh123/workspace/models/deepseek-coder-1.3b-base')
# tokenizer = AutoTokenizer.from_pretrained('/home/syh123/workspace/models/Seed-Coder-8B-Instruct')

prompt = "You are tasked with designing a verilog module for a 4-bit binary comparator. The module should take two 4-bit inputs, A and B, and output a 2-bit signal indicating their relationship. The output signal should have the following values:\n- 00 if A = B\n- 01 if A > B\n- 10 if A < B\n\nDesign and implement the verilog module for this comparator. \n\n"
code = "module comparator(A, B, result);\n    input [3:0] A, B;      // 4-bit inputs A and B\n    output [1:0] result;   // 2-bit output result indicating comparison\n    \n    // Internal signal declarations\n    wire A_eq_B, A_gt_B, A_lt_B;\n    \n    // Compare A and B\n    assign A_eq_B = (A == B); // A is equal to B\n    assign A_gt_B = (A > B);  // A is greater than B\n    assign A_lt_B = (A < B);  // A is less than B\n    \n    // Output results based on comparison\n    assign result = (A_eq_B) ? 2'b00 :\n                    (A_gt_B) ? 2'b01 :\n                    2'b10; // A_lt_B is the last case\nendmodule\n"

# qwen/seed-coder
# messages = [
#     {"role": "system", "content": "Please act as a professional Verilog designer and provide Verilog code based on the given instruction."},
#     {"role": "user", "content": prompt},
#     {"role": "assistant", "content": f"```verilog\n{code}\n```"}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=False
# )

# print(text)

# # deepseek
# system_prompt = "Please act as a professional Verilog designer and provide Verilog code based on the given instruction."
# text = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant: ```verilog\n{code}\n```"
# print(text)

# codellama
system_prompt = "Please act as a professional Verilog designer and provide Verilog code based on the given instruction."
# 格式: <s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction} [/INST] {response}</s>
text = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST] ```verilog\n{code}\n```</s>"
print(text)

# 构建 Seed-Coder 格式的训练文本
# 格式: <|system|>\n{system_prompt}<|end|>\n<|user|>\n{instruction}<|end|>\n<|assistant|>\n{response}<|end|>
# text = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{instruction.strip()}<|end|>\n<|assistant|>\n```verilog\n{solution.strip()}\n```<|end|>"

# <|im_start|>system
# {system_message}<|im_end|>
# <|im_start|>user
# {user_message}<|im_end|>
# <|im_start|>assistant
# {assistant_message}<|im_end|>


# 格式: <|system|>\n{system_prompt}<|end|>\n<|user|>\n{instruction}<|end|>\n<|assistant|>\n{response}<|end|>
# <s>System:
# {system_message}</s>
# <s>User:
# {user_message}</s>
# <s>Assistant:
# {assistant_message}</s>

# <s>System:
# Please act as a professional Verilog designer and provide Verilog code based on the given instruction.</s>
# <s>User:
# {instruction}</s>
# <s>Assistant:
# ```verilog
# {solution}
# ```</s>