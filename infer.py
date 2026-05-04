#!/usr/bin/env python3
"""
Minimal inference script - extracted from train.py
Directly copied eval logic from train.py, skipping training loop.
"""

import argparse
import logging
import sys
from typing import Dict, Optional

import gin
import torch
import fbgemm_gpu
from generative_recommenders.research.data.eval import (
    _avg,
    eval_metrics_v2_from_tensors,
    get_eval_state,
)
from generative_recommenders.research.data.reco_dataset import get_reco_dataset
from generative_recommenders.research.indexing.utils import get_top_k_module
from generative_recommenders.research.modeling.sequential.autoregressive_losses import (
    LocalNegativesSampler,
)
from generative_recommenders.research.modeling.sequential.embedding_modules import (
    EmbeddingModule,
    LocalEmbeddingModule,
)
from generative_recommenders.research.modeling.sequential.encoder_utils import (
    get_sequential_encoder,
)
from generative_recommenders.research.modeling.sequential.features import (
    movielens_seq_features_from_row,
)
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
)
from generative_recommenders.research.modeling.sequential.output_postprocessors import (
    OutputPostprocessorModule,
    L2NormEmbeddingPostprocessor,
    LayerNormEmbeddingPostprocessor,
)
from generative_recommenders.research.modeling.similarity_utils import (
    get_similarity_function,
)
from generative_recommenders.research.trainer.data_loader import create_data_loader


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gin-config", type=str, required=True, help="Path to gin config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    return parser.parse_args()


@gin.configurable
def train_fn(
    checkpoint_path: str,
    device: str,
    dataset_name: str = "ml-20m",
    max_sequence_length: int = 200,
    positional_sampling_ratio: float = 1.0,
    local_batch_size: int = 128,
    eval_batch_size: int = 128,
    eval_user_max_batch_size: Optional[int] = None,
    main_module: str = "SASRec",
    main_module_bf16: bool = False,
    dropout_rate: float = 0.2,
    user_embedding_norm: str = "l2_norm",
    sampling_strategy: str = "in-batch",
    loss_module: str = "SampledSoftmaxLoss",
    loss_weights: Optional[Dict[str, float]] = None,
    num_negatives: int = 1,
    loss_activation_checkpoint: bool = False,
    item_l2_norm: bool = False,
    temperature: float = 0.05,
    num_epochs: int = 101,
    learning_rate: float = 1e-3,
    num_warmup_steps: int = 0,
    weight_decay: float = 1e-3,
    top_k_method: str = "MIPSBruteForceTopK",
    eval_interval: int = 100,
    full_eval_every_n: int = 1,
    save_ckpt_every_n: int = 1000,
    partial_eval_num_iters: int = 32,
    embedding_module_type: str = "local",
    item_embedding_dim: int = 240,
    interaction_module_type: str = "",
    gr_output_length: int = 10,
    l2_norm_eps: float = 1e-6,
    enable_tf32: bool = False,
    random_seed: int = 42,
) -> None:
    """Inference function - accepts all train_fn parameters but only uses inference-related ones"""
    
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cudnn.allow_tf32 = enable_tf32
    logging.info(f"cuda.matmul.allow_tf32: {enable_tf32}")
    logging.info(f"cudnn.allow_tf32: {enable_tf32}")
    
    # Load dataset (from train.py)
    dataset = get_reco_dataset(
        dataset_name=dataset_name,
        max_sequence_length=max_sequence_length,
        chronological=True,
    )
    
    # Create eval data loader (from train.py)
    eval_data_sampler, eval_data_loader = create_data_loader(
        dataset.eval_dataset,
        batch_size=eval_batch_size,
        world_size=1,
        rank=0,
        shuffle=False,
    )
    
    # Build model (from train.py lines 161-223, copied directly)
    logging.info(f"Building model: {main_module}")
    
    if embedding_module_type == "local":
        embedding_module: EmbeddingModule = LocalEmbeddingModule(
            num_items=dataset.max_item_id,
            item_embedding_dim=item_embedding_dim,
        )
    else:
        raise ValueError(f"Unknown embedding_module_type {embedding_module_type}")
    
    interaction_module, _ = get_similarity_function(
        module_type=interaction_module_type,
        query_embedding_dim=item_embedding_dim,
        item_embedding_dim=item_embedding_dim,
    )
    
    assert user_embedding_norm == "l2_norm" or user_embedding_norm == "layer_norm", (
        f"Not implemented for {user_embedding_norm}"
    )
    output_postproc_module = (
        L2NormEmbeddingPostprocessor(
            embedding_dim=item_embedding_dim,
            eps=1e-6,
        )
        if user_embedding_norm == "l2_norm"
        else LayerNormEmbeddingPostprocessor(
            embedding_dim=item_embedding_dim,
            eps=1e-6,
        )
    )
    
    input_preproc_module = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
        max_sequence_len=dataset.max_sequence_length + gr_output_length + 1,
        embedding_dim=item_embedding_dim,
        dropout_rate=dropout_rate,
    )
    
    model = get_sequential_encoder(
        module_type=main_module,
        max_sequence_length=dataset.max_sequence_length,
        max_output_length=gr_output_length + 1,
        embedding_module=embedding_module,
        interaction_module=interaction_module,
        input_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
        verbose=True,
    )
    
    # Load checkpoint (from train.py)
    logging.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Handle DDP checkpoint (remove "module." prefix)
    state_dict = {}
    for k, v in checkpoint["model_state_dict"].items():
        if k.startswith("module."):
            state_dict[k[7:]] = v
        else:
            state_dict[k] = v
    
    model.load_state_dict(state_dict)
    
    # Move model to device (from train.py)
    if main_module_bf16:
        model = model.to(torch.bfloat16)
    model = model.to(device)
    model.eval()
    
    # Create negatives sampler for eval (from train.py)
    negatives_sampler = LocalNegativesSampler(
        num_items=dataset.max_item_id,
        item_emb=model._embedding_module._item_emb,
        all_item_ids=dataset.all_item_ids,
        l2_norm=item_l2_norm,
        l2_norm_eps=l2_norm_eps,
    ).to(device)
    
    # Eval per epoch (from train.py lines 431-487, copied directly)
    logging.info("Running inference...")
    eval_dict_all = None
    
    with torch.no_grad():
        eval_state = get_eval_state(
            model=model,
            all_item_ids=dataset.all_item_ids,
            negatives_sampler=negatives_sampler,
            top_k_module_fn=lambda item_embeddings, item_ids: get_top_k_module(
                top_k_method=top_k_method,
                model=model,
                item_embeddings=item_embeddings,
                item_ids=item_ids,
            ),
            device=device,
            float_dtype=torch.bfloat16 if main_module_bf16 else None,
        )
        
        for eval_iter, row in enumerate(iter(eval_data_loader)):
            seq_features, target_ids, target_ratings = movielens_seq_features_from_row(
                row, device=device, max_output_length=gr_output_length + 1
            )
            eval_dict = eval_metrics_v2_from_tensors(
                eval_state,
                model,
                seq_features,
                target_ids=target_ids,
                target_ratings=target_ratings,
                user_max_batch_size=eval_user_max_batch_size,
                dtype=torch.bfloat16 if main_module_bf16 else None,
            )
            
            if eval_dict_all is None:
                eval_dict_all = {}
                for k, v in eval_dict.items():
                    eval_dict_all[k] = []
            
            for k, v in eval_dict.items():
                eval_dict_all[k] = eval_dict_all[k] + [v]
            del eval_dict
            
            if (eval_iter + 1 >= partial_eval_num_iters):
                logging.info(
                    f"Truncating eval to {eval_iter + 1} iters to save cost.."
                )
                break
    
    assert eval_dict_all is not None
    for k, v in eval_dict_all.items():
        eval_dict_all[k] = torch.cat(v, dim=-1)
    
    ndcg_10 = _avg(eval_dict_all["ndcg@10"], world_size=1)
    ndcg_50 = _avg(eval_dict_all["ndcg@50"], world_size=1)
    hr_10 = _avg(eval_dict_all["hr@10"], world_size=1)
    hr_50 = _avg(eval_dict_all["hr@50"], world_size=1)
    mrr = _avg(eval_dict_all["mrr"], world_size=1)
    
    # Print results
    logging.info("="*80)
    logging.info(
        f"Eval Results: "
        f"NDCG@10 {ndcg_10:.4f}, NDCG@50 {ndcg_50:.4f}, HR@10 {hr_10:.4f}, HR@50 {hr_50:.4f}, MRR {mrr:.4f}"
    )
    logging.info("="*80)


def main():
    args = parse_args()
    
    # Parse gin config
    gin.parse_config_file(args.gin_config)
    
    # Run inference
    train_fn(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )


if __name__ == "__main__":
    main()
