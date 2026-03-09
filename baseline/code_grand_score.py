import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import List, Dict, Any, Tuple
import json
import os


class CodeGraNdScorer:
    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Initialize GraNd scorer for code generation tasks.
        
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
    
    def compute_grand_scores_batch(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        Compute GraNd scores for a batch of code examples.
        
        Args:
            batch: Batch containing input data
            
        Returns:
            Array of GraNd scores for the batch
        """
        self.model.eval()
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        labels_mask = batch['labels_attention_mask']
        
        batch_size = input_ids.shape[0]
        grad_norms = []
        
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
            
            # Create labels for causal LM (only response part is used for loss)
            prompt_length = prompt_attention_mask.sum().item()
            full_labels = full_input_ids.clone()
            full_labels[:, :prompt_length] = -100  # Mask prompt part
            
            # Get input embeddings and require gradients
            input_embeddings = self.model.get_input_embeddings()(full_input_ids)
            input_embeddings.requires_grad_(True)
            
            # Forward pass using custom embeddings
            outputs = self.model(
                inputs_embeds=input_embeddings,
                attention_mask=full_attention_mask,
                labels=full_labels
            )
            loss = outputs.loss
            
            # Compute gradients with respect to input embeddings
            loss.backward()
            
            # Get gradient norms for this example
            if input_embeddings.grad is not None:
                grad_norm = torch.norm(input_embeddings.grad, p=2).item()
            else:
                grad_norm = 0.0
            grad_norms.append(grad_norm)
            
            # Detach and clear gradients
            input_embeddings.grad = None
            input_embeddings.requires_grad_(False)
            
            # Clear GPU cache to avoid memory issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return np.array(grad_norms)
    
    def compute_grand_scores(self, dataset, batch_size: int = 2) -> np.ndarray:
        """
        Compute GraNd scores for entire dataset.
        
        Args:
            dataset: Hugging Face dataset
            batch_size: Batch size for computation
            
        Returns:
            Array of GraNd scores for all examples
        """
        all_scores = []
        
        for i in range(0, len(dataset), batch_size):
            end_idx = min(i + batch_size, len(dataset))
            batch_data = dataset[i:end_idx]
            
            prepared_batch = self.prepare_code_data(batch_data)
            batch_scores = self.compute_grand_scores_batch(prepared_batch)
            
            all_scores.extend(batch_scores)
            
            if (i // batch_size) % 10 == 0:
                print(f"Processed {i}/{len(dataset)} examples...")
        
        return np.array(all_scores)


def save_dataset_with_scores(dataset, scores, score_type: str, output_file: str):
    """
    Save dataset with scores as JSON file with original format.
    
    Args:
        dataset: Dataset to save
        scores: Array of scores
        score_type: Type of score ('el2n' or 'grand')
        output_file: Path to output JSON file
    """
    # Convert dataset to list of dictionaries with scores
    data_list = []
    for i, item in enumerate(dataset):
        data_item = {
            "Instruction": item["Instruction"],
            "Response": item["Response"],
            "canonical_solution": item["canonical_solution"],
            "test": item["test"],
            f"{score_type}_score": float(scores[i])
        }
        data_list.append(data_item)
    
    # Save as JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset with {score_type} scores saved to {output_file} with {len(data_list)} examples")


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
    # Extract scores
    scores = [item[f"{score_type}_score"] for item in dataset_with_scores]
    
    # Calculate number of samples to keep
    k = int(len(dataset_with_scores) * k_percent / 100)
    
    # Get indices of examples with highest scores
    selected_indices = np.argsort(scores)[-k:]
    
    # Create sampled dataset
    sampled_data = [dataset_with_scores[i] for i in selected_indices]
    
    # Remove score field if desired (optional)
    for item in sampled_data:
        if f"{score_type}_score" in item:
            del item[f"{score_type}_score"]
    
    # Save sampled dataset if output file is specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sampled_data, f, indent=2, ensure_ascii=False)
        print(f"Sampled dataset saved to {output_file} with {len(sampled_data)} examples")
    
    return sampled_data, selected_indices


def compute_grand_scores_and_save(
    model_name: str,
    dataset,
    output_file: str,
    batch_size: int = 2,  # Reduced batch size for memory
    device: str = "cuda"
) -> np.ndarray:
    """
    Compute GraNd scores and save them with the dataset.
    
    Args:
        model_name: Qwen model name/path
        dataset: Training dataset
        output_file: Path to save dataset with scores
        batch_size: Batch size for computation
        device: Device to use
        
    Returns:
        Array of GraNd scores
    """
    print(f"Initializing GraNd scorer with model: {model_name}")
    scorer = CodeGraNdScorer(model_name, device)
    
    print(f"Computing GraNd scores for {len(dataset)} code examples...")
    grand_scores = scorer.compute_grand_scores(dataset, batch_size)
    
    print(f"Saving dataset with GraNd scores to {output_file}...")
    save_dataset_with_scores(dataset, grand_scores, "grand", output_file)
    
    return grand_scores