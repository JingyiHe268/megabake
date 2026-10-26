#!/usr/bin/env python3
"""Create one checkpoint for both stacks; never overwrite an existing model."""
import argparse
import json
from pathlib import Path
from generation_common import fingerprint, PROMPT

p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);args=p.parse_args()
if args.model.exists():
    saved=json.loads((args.model/'mock-manifest.json').read_text())
    if saved['files'] != fingerprint(args.model): raise RuntimeError('Existing checkpoint fingerprint differs')
    print(json.dumps(saved));raise SystemExit(0)
import torch
from transformers import LlamaConfig,LlamaForCausalLM,PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

torch.manual_seed(20261026)
# Official SmolLM-135M structural config, with an explicit synthetic context extension.
config=LlamaConfig(hidden_size=576,intermediate_size=1536,num_attention_heads=9,num_key_value_heads=3,
                   num_hidden_layers=30,vocab_size=49152,max_position_embeddings=8192,
                   rms_norm_eps=1e-5,tie_word_embeddings=True,attention_bias=False,mlp_bias=False,
                   rope_theta=10000.,bos_token_id=0,eos_token_id=1,pad_token_id=2,use_cache=True)
with torch.device('cuda'): model=LlamaForCausalLM(config).to(dtype=torch.bfloat16).eval()
args.model.mkdir(parents=True)
model.save_pretrained(args.model,safe_serialization=True)
tok=Tokenizer(WordLevel({f't{i}':i for i in range(49152)},unk_token='t3'));tok.pre_tokenizer=Whitespace()
fast=PreTrainedTokenizerFast(tokenizer_object=tok,bos_token='t0',eos_token='t1',pad_token='t2',unk_token='t3')
fast.save_pretrained(args.model)
manifest=dict(mock_weights=True,seed=20261026,architecture='SmolLM-135M structural config',
              parameters=sum(p.numel() for p in model.parameters()),context_extension=8192,
              tie_word_embeddings=True,rms_norm_eps=1e-5,prompt_ids=PROMPT,files=fingerprint(args.model))
(args.model/'mock-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
