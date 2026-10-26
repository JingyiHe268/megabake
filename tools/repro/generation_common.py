"""Shared fixed-prompt continuous-generation contract (stdlib only until execution)."""
import hashlib
import json
from pathlib import Path
import statistics

LENGTHS = [128, 256, 512, 1024, 2048, 4096]
PROMPT = list(range(100, 228))
CHECKS = [0, 1, 127, 255, 511, 1023, 2047, 4095]


def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(folder):
    folder = Path(folder)
    return {p.name: digest_file(p) for p in sorted(folder.iterdir())
            if p.name in ('config.json','model.safetensors','tokenizer.json','tokenizer_config.json')}


def token_digest(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


def record(length, trial, elapsed_s, prefill_s, tokens):
    if len(tokens) != length:
        raise ValueError('Incorrect number of output tokens')
    steps = length - 1
    return dict(output_tokens=length, decode_steps=steps, prefill_tokens=128, batch_size=1,
                trial=trial, decode_total_ms=elapsed_s*1000,
                decode_mean_ms_per_token=elapsed_s*1000/steps,
                decode_tokens_per_second=steps/elapsed_s, prefill_ms=prefill_s*1000,
                final_cached_tokens=128+steps, output_sha256=token_digest(tokens),
                output_token_ids=tokens)
