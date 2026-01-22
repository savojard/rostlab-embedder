#!/usr/bin/env python
"""
ProstT5 embedding generator (Hugging Face)

Adapted from ProtT5 script to work with Rostlab/ProstT5 family.

Features:
- Writes ONE .npz PER SEQUENCE into --out-dir
- Splits sequences longer than --max-len into tokenizer-aware chunks (by residues) and merges embeddings

Example (online):
  python prostt5.py input.fasta \
    --model-id Rostlab/ProstT5 \
    --device cuda:0 --dtype bf16 \
    --pooling mean \
    --out-dir ./prostt5_out

Offline:
  python prostt5.py input.fasta \
    --model-dir /models/Rostlab_ProstT5 \
    --tokenizer-dir /models/Rostlab_ProstT5 \
    --device cuda:0 --dtype bf16 \
    --pooling none --save-per-residue \
    --out-dir ./prostt5_out
"""
import os
import sys
import re
import argparse
from typing import List, Tuple

import numpy as np
import torch
from Bio import SeqIO
from transformers import AutoTokenizer, T5EncoderModel


def read_fasta(fasta_path: str) -> List[Tuple[str, str]]:
    records = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seq = str(rec.seq).replace(" ", "").replace("\n", "")
        if not seq:
            continue
        records.append((rec.id, seq))
    if not records:
        raise ValueError("No sequences found in FASTA.")
    return records


def pick_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate ProstT5 embeddings (HF) with per-sequence .npz outputs."
    )
    p.add_argument("fasta", help="Input FASTA (one or many sequences).")

    # Model sources (full HF ID, NOT auto-prefixed)
    p.add_argument(
        "--model-id",
        default="Rostlab/ProstT5",
        help="HF model id (e.g. 'Rostlab/ProstT5'). Ignored if --model-dir is set.",
    )
    p.add_argument(
        "--model-dir",
        default=None,
        help="Local directory with pre-downloaded model (offline).",
    )
    p.add_argument(
        "--tokenizer-dir",
        default=None,
        help="Local directory for tokenizer (defaults to model source).",
    )

    # Inference controls
    p.add_argument(
        "--device",
        default=None,
        help='Torch device, e.g. "cuda:0" or "cpu". Default auto.',
    )
    p.add_argument(
        "--dtype",
        choices=["bf16", "fp32"],
        default="fp32",
        help="Computation dtype for inference.",
    )
    p.add_argument(
        "--max-len",
        type=int,
        default=1024,
        help="Max tokenizer length (tokens including specials).",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Optional model-internal chunk size (if supported) to reduce VRAM "
             "(ProstT5 usually does not expose this; will be ignored if unsupported).",
    )

    # Output & pooling
    p.add_argument(
        "--pooling",
        choices=["mean", "cls", "none"],
        default="mean",
        help="Per-sequence pooling: mean/cls or none (save per-residue only). "
             "NOTE: if the sequence is chunked, 'cls' falls back to 'mean'.",
    )
    p.add_argument(
        "--save-per-residue",
        action="store_true",
        help="Also save per-residue embeddings in each .npz file.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory to store one .npz per sequence.",
    )
    p.add_argument(
        "--shard-accession",
        action="store_true",
        help="Shard outputs into two-level subdirectories based on accession prefix.",
    )
    return p.parse_args()


def tokenizer_payload_len(tokenizer, max_len: int) -> int:
    """Compute how many *non-special* tokens can fit given max_len."""
    try:
        specials = tokenizer.num_special_tokens_to_add(pair=False)
    except Exception:
        # Fallback heuristic: assume 2 special tokens (BOS/EOS or CLS/SEP)
        specials = 2
    payload = max_len - specials
    return max(payload, 1)


def mask_special_tokens(
    input_ids: torch.Tensor, attn_mask: torch.Tensor, tokenizer
) -> torch.Tensor:
    """Build a boolean mask of valid (non-special) tokens where attention_mask==1."""
    mask = attn_mask.bool()
    for attr in (
        "cls_token_id",
        "bos_token_id",
        "eos_token_id",
        "sep_token_id",
        "pad_token_id",
    ):
        tok_id = getattr(tokenizer, attr, None)
        if tok_id is not None:
            mask &= input_ids != tok_id
    return mask


def prepare_prostt5_input(seq: str) -> str:
    """
    Prepare an amino-acid sequence string for ProstT5:
    - remove whitespace
    - uppercase
    - replace U, Z, O, B with X
    - insert spaces between residues (A C D E ...)
    """
    seq = seq.replace(" ", "").upper()
    seq = re.sub(r"[UZOB]", "X", seq)
    return " ".join(list(seq))


def accession_shard_parts(seq_id: str, width: int = 2, levels: int = 2) -> List[str]:
    parts = []
    for level in range(levels):
        start = level * width
        part = seq_id[start : start + width]
        if len(part) < width:
            part = part.ljust(width, "_")
        parts.append(part)
    return parts


def build_output_path(out_dir: str, seq_id: str, shard_accession: bool) -> str:
    if not shard_accession:
        return os.path.join(out_dir, f"{seq_id}.npz")
    shard_parts = accession_shard_parts(seq_id)
    return os.path.join(out_dir, *shard_parts, f"{seq_id}.npz")


def embed_chunk(
    model,
    tokenizer,
    seq_chunk: str,
    device,
    torch_dtype,
    max_len: int,
):
    """
    Embed a single sequence chunk with ProstT5 -> returns (per_residue_embeddings [L, H]).
    """
    seq_str = prepare_prostt5_input(seq_chunk)

    enc = tokenizer(
        seq_str,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_len,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
    )
    # For T5EncoderModel / AutoModel on ProstT5, last_hidden_state is [1, T, H]
    last_hidden = outputs.last_hidden_state

    valid_mask = mask_special_tokens(input_ids, attention_mask, tokenizer)[0]  # [T]
    emb_valid = last_hidden[0][valid_mask]  # [L, H] (only non-special tokens)
    return emb_valid  # still on device


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = pick_device(args.device)
    use_bf16 = (args.dtype == "bf16") and (device.type == "cuda")
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float32

    seqs = read_fasta(args.fasta)

    # Model/tokenizer sources
    local_only = args.model_dir is not None or args.tokenizer_dir is not None
    # If model-dir not provided, use --model-id as full HF repo id
    model_src = args.model_dir or args.model_id
    tok_src = args.tokenizer_dir or args.model_dir or args.model_id

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(
        tok_src, local_files_only=local_only, do_lower_case=False,
        use_fast=False, legacy=False
    )
    model = T5EncoderModel.from_pretrained(
         model_src,
         local_files_only=local_only,
         torch_dtype=torch_dtype,
    ).to(device)
    model.eval()
    torch.set_grad_enabled(False)

    # Optional model-internal chunk size (ProstT5 usually doesn't support this;
    # we keep it for API compatibility, but it will likely be ignored).
    if args.chunk_size is not None:
        try:
            if hasattr(model, "set_chunk_size"):
                model.set_chunk_size(args.chunk_size)
                print(f"🔹 Using model chunk size = {args.chunk_size}")
            elif hasattr(model, "encoder") and hasattr(model.encoder, "set_chunk_size"):
                model.encoder.set_chunk_size(args.chunk_size)
                print(f"🔹 Using encoder chunk size = {args.chunk_size}")
        except Exception as e:
            print(f"⚠️ Could not set model chunk size (likely unsupported for ProstT5): {e}")

    payload_len = tokenizer_payload_len(tokenizer, args.max_len)

    # autocast only on CUDA BF16
    autocast_cm = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_bf16
        else torch.autocast("cuda", enabled=False)
    )

    with autocast_cm:
        for seq_id, seq in seqs:
            # Split long sequences into residue-based chunks that respect tokenizer max_len
            chunks = [seq[i : i + payload_len] for i in range(0, len(seq), payload_len)]

            per_residue_parts = []
            for ch in chunks:
                emb_valid = embed_chunk(
                    model,
                    tokenizer,
                    ch,
                    device,
                    torch_dtype,
                    args.max_len,
                )
                # move to CPU fp32 for accumulation
                per_residue_parts.append(
                    emb_valid.detach().to(torch.float32).cpu().numpy()
                )

            # Merge per-residue embeddings across chunks
            per_residue = np.concatenate(per_residue_parts, axis=0)  # [L_total, H]

            # Compute pooled vector (if requested)
            pooled = None
            if args.pooling != "none":
                if len(chunks) > 1 and args.pooling == "cls":
                    # CLS pooling across chunks is ill-defined; fall back to global mean
                    pooled = per_residue.mean(axis=0, keepdims=False)
                elif args.pooling == "mean":
                    pooled = per_residue.mean(axis=0, keepdims=False)
                else:
                    # Single-chunk + 'cls': we also fallback to mean for consistency
                    pooled = per_residue.mean(axis=0, keepdims=False)

            # Save per-sequence .npz
            out_path = build_output_path(args.out_dir, seq_id, args.shard_accession)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            save_dict = {"id": np.array(seq_id, dtype=object)}
            if args.pooling != "none" and pooled is not None:
                save_dict["pooled"] = pooled.astype(np.float32)
            if args.save_per_residue:
                save_dict["token_embeddings"] = per_residue.astype(np.float32)
            np.savez_compressed(out_path, **save_dict)

            print(
                f"✅ Saved {out_path}  "
                f"(pooled={'yes' if 'pooled' in save_dict else 'no'}, "
                f"per_residue={'yes' if 'token_embeddings' in save_dict else 'no'}, "
                f"L={per_residue.shape[0]}, "
                f"H={per_residue.shape[1] if per_residue.ndim == 2 else 'NA'})"
            )

    print(
        f"\nDone. device={device.type}, dtype={'bf16' if use_bf16 else 'fp32'}, "
        f"max_len={args.max_len}, payload_len={payload_len}, pooling={args.pooling}"
    )
    print(f"Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
