import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import List, Dict, Any, Tuple
import json
import os


class CodeEL2NScorer:
    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Initialize EL2N scorer for code generation tasks.
        
        Args:
            model_name: Name or path of the Qwen model
            device: Device to run computation on
        """
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_code_data(self, dataset_batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Prepare code generation data for model input.
        
        Args:
            dataset_batch: Batch from dataset containing 'Instruction' and 'canonical_solution'
            
        Returns:
            Tokenized inputs with labels
        """
        instructions = dataset_batch['Instruction']
        solutions = dataset_batch['canonical_solution']
        
        # Create prompts for code generation
        prompts = []
        labels = []
        
        for instr, sol in zip(instructions, solutions):
            # Format: Instruction + Response prefix
            prompt = f"Instruction: {instr}\nResponse: "
            prompts.append(prompt)
            labels.append(sol)
        
        # Tokenize inputs and labels
        inputs = self.tokenizer(
            prompts, 
            padding=True, 
            truncation=True, 
            max_length=2048,
            return_tensors="pt"
        )
        
        # Tokenize labels separately
        labels_encoded = self.tokenizer(
            labels,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        labels = labels_encoded['input_ids'].to(self.device)
        labels_attention_mask = labels_encoded['attention_mask'].to(self.device)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'labels_attention_mask': labels_attention_mask,
            'prompts': prompts,
            'solutions': solutions
        }
    
    def compute_el2n_scores_batch(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        Compute EL2N scores for a batch of code examples.
        
        Args:
            batch: Batch containing input data
            
        Returns:
            Array of EL2N scores for the batch
        """
        self.model.eval()
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        labels_mask = batch['labels_attention_mask']
        
        with torch.no_grad():
            # Get model predictions for the input (prompt + response)
            # We need to concatenate the prompt and response for proper forward pass
            batch_size = input_ids.shape[0]
            el2n_scores = []
            
            for i in range(batch_size):
                # Get the prompt part
                prompt_input_ids = input_ids[i].unsqueeze(0)
                prompt_attention_mask = attention_mask[i].unsqueeze(0)
                
                # Get the response part
                response_input_ids = labels[i].unsqueeze(0)
                response_attention_mask = labels_mask[i].unsqueeze(0)
                
                # Remove padding from response
                response_length = response_attention_mask.sum().item()
                if response_length > 0:
                    response_input_ids = response_input_ids[:, :response_length]
                    response_attention_mask = response_attention_mask[:, :response_length]
                
                # Concatenate prompt and response
                full_input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1)
                full_attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
                
                # Forward pass with the full sequence
                outputs = self.model(
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask
                )
                
                # Get logits for the response part only
                prompt_length = prompt_attention_mask.sum().item()
                response_logits = outputs.logits[:, prompt_length-1:prompt_length-1 + response_length, :]
                
                # Get probabilities using softmax
                probabilities = F.softmax(response_logits, dim=-1)
                
                # Create one-hot encoding for labels
                one_hot_labels = F.one_hot(response_input_ids, num_classes=self.model.config.vocab_size)
                
                # Compute error vectors (prediction - ground truth)
                error_vectors = probabilities - one_hot_labels
                
                # Compute L2 norm of error vectors for each token
                error_norms = torch.norm(error_vectors, p=2, dim=-1)
                
                # Average error norm across valid tokens
                if response_length > 0:
                    avg_error_norm = error_norms.mean().item()
                else:
                    avg_error_norm = 0.0
                
                el2n_scores.append(avg_error_norm)
            
            return np.array(el2n_scores)
    
    def compute_el2n_scores(self, dataset, batch_size: int = 8) -> np.ndarray:
        """
        Compute EL2N scores for entire dataset.
        
        Args:
            dataset: Hugging Face dataset
            batch_size: Batch size for computation
            
        Returns:
            Array of EL2N scores for all examples
        """
        all_scores = []
        
        for i in range(0, len(dataset), batch_size):
            end_idx = min(i + batch_size, len(dataset))
            batch_data = dataset[i:end_idx]
            
            prepared_batch = self.prepare_code_data(batch_data)
            batch_scores = self.compute_el2n_scores_batch(prepared_batch)
            
            all_scores.extend(batch_scores)
            
            if (i // batch_size) % 10 == 0:
                print(f"Processed {i}/{len(dataset)} examples...")
        
        return np.array(all_scores)


def save_dataset_with_scores(dataset, scores, score_type: str, output_file: str) -> List[Dict]:
    """
    Save dataset with scores as JSON file with original format.
    
    Args:
        dataset: Dataset to save
        scores: Array of scores
        score_type: Type of score ('el2n' or 'grand')
        output_file: Path to output JSON file
        
    Returns:
        Dataset with scores as list of dictionaries
    """
    # Convert dataset to list of dictionaries with scores
    data_list = []
    for i, item in enumerate(dataset):
        # 确保处理Response字段是列表的情况
        response = item["Response"]
        if isinstance(response, list) and len(response) > 0:
            response = response[0]  # 取列表中的第一个元素
        
        data_item = {
            "Instruction": item["Instruction"],
            "Response": response,
            "canonical_solution": item["canonical_solution"],
            "test": item["test"],
            f"{score_type}_score": float(scores[i])
        }
        data_list.append(data_item)
    
    # Save as JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset with {score_type} scores saved to {output_file} with {len(data_list)} examples")
    return data_list


def load_dataset_with_scores(file_path: str) -> List[Dict]:
    """
    Load dataset with precomputed scores from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of data items with scores
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 验证数据集结构
    if len(dataset) > 0:
        first_item = dataset[0]
        print(f"Dataset loaded with {len(dataset)} examples")
        print(f"Available keys: {list(first_item.keys())}")
        
        # 检查是否有el2n_score字段
        if 'el2n_score' in first_item:
            print(f"Found el2n_score field with value: {first_item['el2n_score']}")
        else:
            print("Warning: el2n_score field not found in dataset")
    
    return dataset


def sample_from_scored_dataset(dataset_with_scores, score_type: str, k_percent: float, output_file: str = None):
    """
    Sample top k% of data from a dataset that already has scores.
    
    Args:
        dataset_with_scores: Dataset with precomputed scores
        score_type: Type of score ('el2n' or 'grand')
        k_percent: Percentage of data to keep (0-100)
        output_file: Path to save sampled dataset (optional)
        
    Returns:
        Sampled dataset and selected indices
    """
    # 检查数据集是否包含分数字段
    score_field = f"{score_type}_score"
    
    print(f"Dataset length: {len(dataset_with_scores)}")
    
    # 验证所有条目都有分数字段
    missing_scores = [i for i, item in enumerate(dataset_with_scores) if score_field not in item]
    if missing_scores:
        print(f"Warning: {len(missing_scores)} examples missing {score_field}")
        print(f"Indices with missing scores: {missing_scores[:10]}")  # 只显示前10个
        
        # 显示第一个缺失分数的样本的键
        if missing_scores:
            first_missing = dataset_with_scores[missing_scores[0]]
            print(f"Keys in first missing sample: {list(first_missing.keys())}")
        
        # 只保留有分数的条目
        dataset_with_scores = [item for item in dataset_with_scores if score_field in item]
    
    if len(dataset_with_scores) == 0:
        raise ValueError(f"No examples found with {score_field}")
    
    print(f"Using {len(dataset_with_scores)} examples with scores")
    
    # Extract scores
    scores = [item[score_field] for item in dataset_with_scores]
    
    # Calculate number of samples to keep
    k = int(len(dataset_with_scores) * k_percent / 100)
    
    print(f"Selecting top {k_percent}% ({k} examples) from {len(dataset_with_scores)} total examples")
    
    # Get indices of examples with highest scores
    selected_indices = np.argsort(scores)[-k:]
    
    # Create sampled dataset
    sampled_data = [dataset_with_scores[i] for i in selected_indices]
    
    # Remove score field if desired (optional)
    for item in sampled_data:
        if score_field in item:
            del item[score_field]
    
    # Save sampled dataset if output file is specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sampled_data, f, indent=2, ensure_ascii=False)
        print(f"Sampled dataset saved to {output_file} with {len(sampled_data)} examples")
    
    return sampled_data, selected_indices


def compute_el2n_scores_and_save(
    model_name: str,
    dataset,
    output_file: str,
    batch_size: int = 8,
    device: str = "cuda"
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Compute EL2N scores and save them with the dataset.
    
    Args:
        model_name: Qwen model name/path
        dataset: Training dataset
        output_file: Path to save dataset with scores
        batch_size: Batch size for computation
        device: Device to use
        
    Returns:
        Tuple of (EL2N scores array, dataset with scores)
    """
    print(f"Initializing EL2N scorer with model: {model_name}")
    scorer = CodeEL2NScorer(model_name, device)
    
    print(f"Computing EL2N scores for {len(dataset)} code examples...")
    el2n_scores = scorer.compute_el2n_scores(dataset, batch_size)
    
    print(f"Saving dataset with EL2N scores to {output_file}...")
    dataset_with_scores = save_dataset_with_scores(dataset, el2n_scores, "el2n", output_file)
    
    return el2n_scores, dataset_with_scores


def average_scores_across_runs(score_arrays: List[np.ndarray]) -> np.ndarray:
    """
    Average scores across multiple training runs/initializations.
    
    Args:
        score_arrays: List of score arrays from different runs
        
    Returns:
        Averaged scores across all runs
    """
    return np.mean(np.stack(score_arrays), axis=0)