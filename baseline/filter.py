import json

def filter_lowest_el2n_scores(input_file_path, output_file_path, k):
    """
    从JSON Lines格式文件中筛选EL2N分数最小的前k条数据，
    移除el2n_score字段，并保持JSON Lines格式输出
    
    参数:
        input_file_path: 输入文件路径
        output_file_path: 输出文件路径
        k: 要筛选的数据条数
    """
    try:
        # 读取JSON Lines格式文件
        data = []
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    item = json.loads(line)
                    data.append(item)
        
        print(f"成功读取文件: {input_file_path}")
        print(f"原始数据条数: {len(data)}")
        
        # 按EL2N分数升序排序（最小在前）
        data_sorted = sorted(data, key=lambda x: x['el2n_score'])
        
        # 选择前k条数据
        if k > len(data_sorted):
            k = len(data_sorted)
            print(f"警告: k值超过数据总数，将使用全部 {k} 条数据")
        
        filtered_data = data_sorted[:k]
        
        # 显示筛选前后的EL2N分数范围
        if len(data) > 0:
            all_scores = [item['el2n_score'] for item in data]
            filtered_scores = [item['el2n_score'] for item in filtered_data]
            print(f"原始EL2N分数范围: {min(all_scores):.6f} - {max(all_scores):.6f}")
            print(f"筛选后EL2N分数范围: {min(filtered_scores):.6f} - {max(filtered_scores):.6f}")
        
        print(f"筛选后数据条数: {len(filtered_data)}")
        
        # 输出为JSON Lines格式，同时移除el2n_score字段
        count = 0
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for item in filtered_data:
                # 创建新对象，不包含el2n_score字段
                new_item = {
                    "Instruction": item["Instruction"],
                    "Response": item["Response"],
                    "canonical_solution": item["canonical_solution"],
                    "test": item["test"]
                }
                # 写入JSON Lines格式
                json.dump(new_item, f, ensure_ascii=False)
                f.write('\n')
                count += 1
        
        print(f"\n数据已成功输出到: {output_file_path}")
        print(f"输出文件包含 {count} 条数据，格式为JSON Lines")
        
        # 验证输出文件的前几行
        print("\n输出文件前2行预览:")
        with open(output_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < 2:  # 只显示前2行
                    print(f"第{i+1}行: {line[:200]}...")  # 显示每行的前200个字符
                else:
                    break
                    
        return True
        
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_file_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"错误: JSON解析失败 - {e}")
        return False
    except Exception as e:
        print(f"错误: {e}")
        return False

if __name__ == "__main__":
    # 文件路径
    input_file_path = "/home/syh123/workspace/Data_Filter/baseline/temp/dataset_with_el2n_scores1.json"
    output_file_path = "/home/syh123/workspace/Data_Filter/baseline/temp/filtered_el2n_25.json"
    
    # 设置要筛选的数据条数
    k = 6633  # 3980, 5306, 6633
    
    # 执行筛选
    success = filter_lowest_el2n_scores(input_file_path, output_file_path, k)
    
    if success:
        print("\n筛选完成!")
    else:
        print("\n筛选失败!")