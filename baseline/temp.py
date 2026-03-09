import json

# 文件路径
input_file_path = "/home/syh123/workspace/Data_Filter/baseline/temp/dataset_with_el2n_scores.json"
output_file_path = "/home/syh123/workspace/Data_Filter/baseline/temp/dataset_with_el2n_scores1.json"

try:
    # 读取原始数据文件
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"成功读取文件: {input_file_path}")
    print(f"数据条数: {len(data)}")
    
    # 显示一些统计信息
    if len(data) > 0:
        el2n_scores = [item['el2n_score'] for item in data]
        print(f"EL2N分数范围: {min(el2n_scores):.6f} - {max(el2n_scores):.6f}")
        print(f"EL2N分数平均值: {sum(el2n_scores)/len(el2n_scores):.6f}")
        
        # 显示前几条数据的结构
        print("\n第一条数据的结构:")
        print(f"Instruction长度: {len(data[0]['Instruction'])}")
        print(f"Response类型: {type(data[0]['Response'])}")
        print(f"canonical_solution长度: {len(data[0]['canonical_solution'])}")
        print(f"test长度: {len(data[0]['test'])}")
        print(f"el2n_score: {data[0]['el2n_score']}")
    
    # 输出为JSON Lines格式（每行一个JSON对象）
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for i, item in enumerate(data):
            # 将每个对象单独写入一行
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')  # 每行结束后换行
    
    print(f"\n数据已成功输出到: {output_file_path}")
    print(f"输出文件包含 {len(data)} 条数据，格式为JSON Lines")
    
    # 验证输出文件的前几行
    print("\n输出文件前2行预览:")
    with open(output_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 2:  # 只显示前2行
                print(f"第{i+1}行: {line[:200]}...")  # 显示每行的前200个字符
            else:
                break

except FileNotFoundError:
    print(f"错误: 找不到输入文件 {input_file_path}")
except json.JSONDecodeError as e:
    print(f"错误: JSON解析失败 - {e}")
except Exception as e:
    print(f"错误: {e}")