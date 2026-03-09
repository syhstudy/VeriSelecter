import pyverilog.vparser.parser as pvparser
import pyverilog.vparser.ast as vast
import logging
import json
import argparse
import os
import tempfile
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class VerilogASTFrequencyAnalyzer:
    """
    独立分析Verilog AST节点类型频率的类
    """
    
    def __init__(self, log_level=logging.INFO):
        logging.getLogger().setLevel(log_level)
        self.all_node_types = set()
        self.node_frequencies = Counter()
        self.total_samples = 0
        self.processed_samples = 0
    
    def parse_verilog_code(self, code: str) -> Optional[vast.Source]:
        """解析Verilog代码并返回AST"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            ast, _ = pvparser.parse([temp_file])
            os.unlink(temp_file)
            return ast
        except Exception as e:
            logging.debug(f"解析Verilog代码时出错: {e}")
            return None
    
    def analyze_ast_nodes(self, ast: vast.Source) -> Dict[str, int]:
        """分析单个AST的节点类型频率"""
        node_counter = Counter()
        
        def count_nodes(node: Any):
            """递归统计节点类型"""
            node_type = type(node).__name__
            node_counter[node_type] += 1
            self.all_node_types.add(node_type)
            
            for child in node.children():
                count_nodes(child)
        
        count_nodes(ast)
        return node_counter
    
    def analyze_dataset(self, jsonl_file: str, sample_limit: int = None, 
                       use_canonical_solution: bool = True) -> Tuple[Counter, set]:
        """
        分析整个数据集的节点类型频率
        
        Args:
            jsonl_file: JSONL文件路径
            sample_limit: 限制分析的样本数量（None表示分析所有样本）
            use_canonical_solution: 是否使用canonical_solution字段
        
        Returns:
            node_frequencies: 节点类型频率计数器
            all_node_types: 所有出现的节点类型集合
        """
        logging.info(f"开始分析数据集: {jsonl_file}")
        if sample_limit:
            logging.info(f"样本限制: {sample_limit}")
        
        self.node_frequencies = Counter()
        self.all_node_types = set()
        self.total_samples = 0
        self.processed_samples = 0
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if sample_limit and i >= sample_limit:
                    break
                    
                self.total_samples += 1
                
                try:
                    data = json.loads(line.strip())
                    
                    # 支持多种可能的代码字段
                    if use_canonical_solution:
                        verilog_code = data.get('canonical_solution', '')
                    else:
                        # 尝试其他可能的字段
                        verilog_code = data.get('verilog_code', 
                                               data.get('code', 
                                                       data.get('solution', '')))
                    
                    if not verilog_code:
                        if self.total_samples % 1000 == 0:
                            logging.debug(f"第 {self.total_samples} 个样本没有找到代码字段")
                        continue
                    
                    ast = self.parse_verilog_code(verilog_code)
                    if ast is None:
                        continue
                    
                    sample_freq = self.analyze_ast_nodes(ast)
                    self.node_frequencies.update(sample_freq)
                    self.processed_samples += 1
                    
                    if self.processed_samples % 1000 == 0:
                        logging.info(f"已处理 {self.processed_samples} 个样本")
                        
                except Exception as e:
                    logging.debug(f"处理第 {i+1} 行时出错: {e}")
                    continue
        
        logging.info(f"分析完成: 总共 {self.total_samples} 个样本, 成功处理 {self.processed_samples} 个样本")
        logging.info(f"发现 {len(self.all_node_types)} 种不同的节点类型")
        
        return self.node_frequencies, self.all_node_types
    
    def get_top_k_nodes(self, k: int = 30, min_frequency: float = 0.001) -> List[str]:
        """
        获取前k个最常见的节点类型
        
        Args:
            k: 返回的节点类型数量
            min_frequency: 最小频率阈值（相对于总节点数）
        
        Returns:
            前k个节点类型列表
        """
        if not self.node_frequencies:
            logging.warning("尚未分析数据集，请先调用analyze_dataset")
            return []
        
        total_nodes = sum(self.node_frequencies.values())
        logging.info(f"总节点数: {total_nodes}")
        
        # 计算每个节点类型的频率
        sorted_nodes = self.node_frequencies.most_common()
        
        # 过滤低频节点
        filtered_nodes = []
        for node_type, count in sorted_nodes:
            frequency = count / total_nodes
            if frequency >= min_frequency:
                filtered_nodes.append((node_type, count, frequency))
        
        # 取前k个
        top_k = filtered_nodes[:k]
        
        logging.info(f"前{len(top_k)}个节点类型 (频率 >= {min_frequency}):")
        for i, (node_type, count, freq) in enumerate(top_k, 1):
            logging.info(f"  {i:2d}. {node_type:25s} {count:8d} ({freq:.4f})")
        
        return [node_type for node_type, _, _ in top_k]
    
    def get_detailed_statistics(self, top_k: int = 50) -> Dict[str, Any]:
        """
        获取详细的统计信息
        
        Returns:
            包含详细统计信息的字典
        """
        if not self.node_frequencies:
            return {}
        
        total_nodes = sum(self.node_frequencies.values())
        
        # 计算累积频率
        sorted_nodes = self.node_frequencies.most_common()
        cumulative_percentage = 0
        stats = []
        
        for i, (node_type, count) in enumerate(sorted_nodes[:top_k]):
            percentage = count / total_nodes * 100
            cumulative_percentage += percentage
            stats.append({
                'rank': i + 1,
                'node_type': node_type,
                'count': count,
                'percentage': round(percentage, 4),
                'cumulative_percentage': round(cumulative_percentage, 4)
            })
        
        # 找到覆盖80%和90%的节点类型数量
        coverage_80 = 0
        coverage_90 = 0
        cumulative = 0
        
        for i, (node_type, count) in enumerate(sorted_nodes):
            cumulative += count / total_nodes
            if cumulative >= 0.8 and coverage_80 == 0:
                coverage_80 = i + 1
            if cumulative >= 0.9 and coverage_90 == 0:
                coverage_90 = i + 1
                break
        
        return {
            'total_samples': self.total_samples,
            'processed_samples': self.processed_samples,
            'total_nodes': total_nodes,
            'unique_node_types': len(self.all_node_types),
            'coverage_80_percent': coverage_80,
            'coverage_90_percent': coverage_90,
            'top_nodes': stats
        }
    
    def plot_node_frequencies(self, top_k: int = 30, save_path: str = None, 
                             figsize: Tuple[int, int] = (12, 8)):
        """绘制节点类型频率分布图"""
        if not self.node_frequencies:
            logging.warning("尚未分析数据集")
            return
        
        total_nodes = sum(self.node_frequencies.values())
        top_nodes = self.node_frequencies.most_common(top_k)
        
        node_types = [node[0] for node in top_nodes]
        frequencies = [node[1] / total_nodes for node in top_nodes]
        
        plt.figure(figsize=figsize)
        bars = plt.barh(node_types, frequencies)
        plt.xlabel('Frequency')
        plt.title(f'Top {top_k} AST Node Type Frequencies (Total: {total_nodes:,} nodes)')
        plt.gca().invert_yaxis()
        
        # 在条形上添加频率标签
        for bar, freq in zip(bars, frequencies):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{freq:.4f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"频率分布图已保存到: {save_path}")
        
        plt.show()
    
    def plot_cumulative_frequency(self, save_path: str = None):
        """绘制累积频率分布图"""
        if not self.node_frequencies:
            logging.warning("尚未分析数据集")
            return
        
        total_nodes = sum(self.node_frequencies.values())
        sorted_nodes = self.node_frequencies.most_common()
        
        cumulative_percentages = []
        cumulative_count = 0
        
        for _, count in sorted_nodes:
            cumulative_count += count
            cumulative_percentages.append(cumulative_count / total_nodes * 100)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_percentages) + 1), cumulative_percentages)
        plt.xlabel('Number of Node Types')
        plt.ylabel('Cumulative Percentage (%)')
        plt.title('Cumulative Frequency Distribution of AST Node Types')
        plt.grid(True, alpha=0.3)
        
        # 标记80%和90%覆盖点
        for target in [80, 90]:
            for i, percentage in enumerate(cumulative_percentages):
                if percentage >= target:
                    plt.axvline(x=i+1, color='red', linestyle='--', alpha=0.7)
                    plt.text(i+1, target, f' {target}%: {i+1} types', 
                            verticalalignment='bottom')
                    break
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"累积频率图已保存到: {save_path}")
        
        plt.show()
    
    def save_analysis_results(self, output_file: str, top_k_nodes: List[str] = None):
        """保存分析结果"""
        detailed_stats = self.get_detailed_statistics()
        
        results = {
            'analysis_summary': {
                'total_samples': self.total_samples,
                'processed_samples': self.processed_samples,
                'total_nodes': sum(self.node_frequencies.values()),
                'unique_node_types': len(self.all_node_types),
                'coverage_80_percent': detailed_stats.get('coverage_80_percent', 0),
                'coverage_90_percent': detailed_stats.get('coverage_90_percent', 0)
            },
            'node_frequencies': dict(self.node_frequencies),
            'top_nodes': detailed_stats.get('top_nodes', [])
        }
        
        if top_k_nodes:
            results['selected_top_k_nodes'] = top_k_nodes
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logging.info(f"分析结果已保存到: {output_file}")
        
        # 同时保存简化的节点列表（便于其他脚本使用）
        if top_k_nodes:
            list_file = output_file.replace('.json', '_nodes.txt')
            with open(list_file, 'w', encoding='utf-8') as f:
                for node in top_k_nodes:
                    f.write(f"'{node}',\n")
            logging.info(f"节点列表已保存到: {list_file}")

def main():
    parser = argparse.ArgumentParser(description="分析Verilog AST节点类型频率")
    parser.add_argument("--jsonl_file", default='/home/syh123/workspace/Data_Filter/data/expanded_rtlcoder_10k.jsonl', help="JSONL数据集文件路径")
    parser.add_argument("--output_dir", default='/home/syh123/workspace/structural_features', help="输出目录")
    parser.add_argument("--k", type=int, default=50, help="选择的前k个节点类型数量")
    parser.add_argument("--sample_limit", type=int, default=3000, 
                       help="分析时使用的样本数量限制（None表示使用所有样本）")
    parser.add_argument("--min_frequency", type=float, default=0.001, 
                       help="节点类型最小频率阈值")
    parser.add_argument("--plot_top_k", type=int, default=50, 
                       help="绘图中显示的前k个节点类型数量")
    parser.add_argument("--no_canonical_solution", action="store_true",
                       help="不使用canonical_solution字段，尝试其他代码字段")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="设置日志级别")
    
    args = parser.parse_args()
    
    # 应用日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 分析节点频率
    analyzer = VerilogASTFrequencyAnalyzer(log_level=args.log_level)
    node_frequencies, all_node_types = analyzer.analyze_dataset(
        args.jsonl_file, 
        sample_limit=args.sample_limit,
        use_canonical_solution=not args.no_canonical_solution
    )
    
    # 获取前k个节点类型
    top_k_nodes = analyzer.get_top_k_nodes(k=args.k, min_frequency=args.min_frequency)
    
    # 保存分析结果
    output_json = os.path.join(args.output_dir, "node_frequency_analysis.json")
    analyzer.save_analysis_results(output_json, top_k_nodes)
    
    # 生成可视化图表
    plot_file = os.path.join(args.output_dir, "node_frequency_plot.png")
    analyzer.plot_node_frequencies(top_k=args.plot_top_k, save_path=plot_file)
    
    cumulative_plot_file = os.path.join(args.output_dir, "cumulative_frequency_plot.png")
    analyzer.plot_cumulative_frequency(save_path=cumulative_plot_file)
    
    # 打印关键统计信息
    stats = analyzer.get_detailed_statistics()
    logging.info("\n" + "="*50)
    logging.info("关键统计信息:")
    logging.info(f"总样本数: {stats['total_samples']}")
    logging.info(f"成功处理: {stats['processed_samples']}")
    logging.info(f"总节点数: {stats['total_nodes']:,}")
    logging.info(f"唯一节点类型: {stats['unique_node_types']}")
    logging.info(f"覆盖80%节点所需类型数: {stats['coverage_80_percent']}")
    logging.info(f"覆盖90%节点所需类型数: {stats['coverage_90_percent']}")
    logging.info("="*50)

if __name__ == "__main__":
    main()