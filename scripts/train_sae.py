#!/usr/bin/env python3
"""Train a BatchTopK SAE on Mistral 7B for code steering.

Trains on bigcode/starcoderdata (requires HuggingFace auth: `huggingface-cli login`).
Logs to Weights & Biases (requires: `wandb login`).

Usage:
    # Quick test run (~5 min)
    python train_sae.py --tokens 50_000_000

    # Full overnight run (~6-8 hours for 4B tokens)
    python train_sae.py --tokens 4_000_000_000

    # Resume from checkpoint
    python train_sae.py --tokens 4_000_000_000 --resume checkpoints/code_sae_v1/checkpoint_2

Hardware:
    Runs on GPU 1 by default (cuda:1), leaving GPU 0 free for the backend.
    Requires ~25 GB VRAM (14.7 GB model + ~8 GB SAE training).
"""

import argparse
import os
import sys
import time

import torch

def main():
    parser = argparse.ArgumentParser(
        description="Train a BatchTopK SAE on Mistral 7B for code feature steering"
    )
    parser.add_argument(
        "--tokens", type=int, default=4_000_000_000,
        help="Number of training tokens (default: 4B, ~6-8h on H100)"
    )
    parser.add_argument("--device", type=str, default="cuda:1", help="GPU device")
    parser.add_argument("--d-sae", type=int, default=32768, help="SAE dictionary size (default: 32768 = 8x)")
    parser.add_argument("--k", type=int, default=64, help="BatchTopK sparsity target (default: 64)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--context-size", type=int, default=512, help="Context window for training samples")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/code_sae_v1")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    # ── Verify environment ──────────────────────────────────────────────
    print("=" * 60)
    print("SAE Training: BatchTopK on Mistral 7B")
    print("=" * 60)
    print(f"  Tokens:       {args.tokens:,}")
    print(f"  Dictionary:   {args.d_sae:,} ({args.d_sae // 4096}x expansion)")
    print(f"  Sparsity k:   {args.k}")
    print(f"  LR:           {args.lr}")
    print(f"  Context:      {args.context_size}")
    print(f"  Device:       {args.device}")
    print(f"  Checkpoints:  {args.checkpoint_path}")
    if args.resume:
        print(f"  Resuming from: {args.resume}")
    print()

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    device_idx = int(args.device.split(":")[-1]) if ":" in args.device else 0
    if device_idx >= torch.cuda.device_count():
        print(f"ERROR: Device {args.device} not available (have {torch.cuda.device_count()} GPUs)")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(device_idx)
    gpu_mem = torch.cuda.get_device_properties(device_idx).total_memory / 1e9
    print(f"  GPU {device_idx}: {gpu_name} ({gpu_mem:.0f} GB)")

    # Check HuggingFace auth (starcoderdata is gated)
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"  HuggingFace: logged in as {user.get('name', user.get('fullname', 'unknown'))}")
    except Exception:
        print("WARNING: Not logged into HuggingFace. Run: huggingface-cli login")
        print("  bigcode/starcoderdata requires authentication.")
        sys.exit(1)

    # Check wandb
    if not args.no_wandb:
        try:
            import wandb
            if wandb.api.api_key is None:
                print("WARNING: wandb not logged in. Run: wandb login")
                print("  Or use --no-wandb to disable logging.")
                sys.exit(1)
            print(f"  wandb: ready")
        except ImportError:
            print("WARNING: wandb not installed, disabling logging")
            args.no_wandb = True

    print()

    # ── Create configs ──────────────────────────────────────────────────
    from sae_lens.saes import BatchTopKTrainingSAEConfig
    from sae_lens import LanguageModelSAERunnerConfig
    from sae_lens import SAETrainingRunner as _DeprecatedRunner
    try:
        from sae_lens import LanguageModelSAETrainingRunner as SAETrainingRunner
    except ImportError:
        SAETrainingRunner = _DeprecatedRunner
    from sae_lens.config import LoggingConfig

    sae_cfg = BatchTopKTrainingSAEConfig(
        d_in=4096,                    # Mistral 7B hidden dim
        d_sae=args.d_sae,
        k=args.k,
        dtype="float32",              # SAE weights in float32 for training stability
        aux_loss_coefficient=1 / 32,  # AuxK dead neuron revival
        decoder_init_norm=0.1,
    )

    logger_cfg = LoggingConfig(
        log_to_wandb=not args.no_wandb,
        wandb_project="mistral-7b-code-sae",
        wandb_log_frequency=100,
        eval_every_n_wandb_logs=10,
    )

    runner_cfg = LanguageModelSAERunnerConfig(
        sae=sae_cfg,
        logger=logger_cfg,

        # Model
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        model_from_pretrained_kwargs={"dtype": "float16"},
        hook_name="blocks.16.hook_resid_post",

        # Dataset — starcoderdata, streamed (no disk download)
        dataset_path="bigcode/starcoderdata",
        streaming=True,
        is_dataset_tokenized=False,
        context_size=args.context_size,
        prepend_bos=True,

        # Training
        training_tokens=args.tokens,
        train_batch_size_tokens=4096,
        lr=args.lr,
        lr_scheduler_name="constant",
        lr_warm_up_steps=1000,
        dead_feature_window=5000,
        dead_feature_threshold=1e-8,

        # Eval
        n_eval_batches=10,

        # Checkpoints (5 evenly-spaced + final)
        n_checkpoints=5,
        checkpoint_path=args.checkpoint_path,
        save_final_checkpoint=True,
        output_path=args.checkpoint_path,

        # Hardware
        dtype="float16",              # Model activations in float16
        device=args.device,
        seed=42,

        # Resume
        resume_from_checkpoint=(args.resume if args.resume else None),
    )

    # ── Load dataset with column rename ────────────────────────────────
    # starcoderdata uses "content" column but sae_lens requires "text"
    from datasets import load_dataset

    print("Loading dataset (streaming)...")
    dataset = load_dataset(
        "bigcode/starcoderdata",
        streaming=True,
        split="train",
    ).rename_column("content", "text")
    print("  Dataset ready (streaming, 'content' renamed to 'text')")

    # ── Train ───────────────────────────────────────────────────────────
    os.makedirs(args.checkpoint_path, exist_ok=True)

    print()
    print("Starting training...")
    print(f"  Estimated time: ~{args.tokens / 500_000_000:.0f} hours on H100")
    print(f"  Checkpoints will be saved to: {args.checkpoint_path}/")
    if not args.no_wandb:
        print(f"  wandb project: mistral-7b-code-sae")
    print()

    start_time = time.time()

    runner = SAETrainingRunner(cfg=runner_cfg, override_dataset=dataset)
    sae = runner.run()

    elapsed = time.time() - start_time
    hours = elapsed / 3600

    print()
    print("=" * 60)
    print(f"Training complete in {hours:.1f} hours")
    print("=" * 60)

    # Save final model
    final_path = os.path.join(args.checkpoint_path, "final")
    os.makedirs(final_path, exist_ok=True)
    sae.save_model(final_path)
    print(f"Final SAE saved to: {final_path}/")
    print()
    print("Next steps:")
    print(f"  1. Profile: python 00_explore_sae.py --sae-path {final_path}")
    print(f"  2. Discover features: run scripts/typing/ pipeline with new SAE")
    print(f"  3. Update backend: change SAE_PATH in server.py to {final_path}")


if __name__ == "__main__":
    main()
