# rostlab-embedder

## Container usage

### Build

```bash
docker build -t rostlab-embedder .
```

### ProtT5 embeddings (default entrypoint)

The image entrypoint runs `prott5.py` by default.

```bash
docker run --rm \
  -v "$(pwd)":/data \
  rostlab-embedder \
  /data/input.fasta \
  --out-dir /data/prott5_out
```

### ProstT5 embeddings (override entrypoint)

Run `prostt5.py` by overriding the entrypoint.

```bash
docker run --rm \
  -v "$(pwd)":/data \
  --entrypoint /workspace/prostt5.py \
  rostlab-embedder \
  /data/input.fasta \
  --out-dir /data/prostt5_out
```

Both scripts accept the same CLI flags. For offline use, mount a directory with
pre-downloaded model/tokenizer files and pass `--model-dir` and `--tokenizer-dir`.
