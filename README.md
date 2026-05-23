# Laboratório 10 — Pipeline Definitivo (QLoRA + RAG + KV Cache + FlashAttention-2)

> Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por João Vinícius Passos Castello Branco Carvalho.

## Contexto Corporativo

A HealthTech está colocando em produção o sistema RAG médico construído no Lab 09. O fluxo é:

1. O RAG recupera ~5 capítulos de manuais médicos (≈12.000 tokens).
2. O contexto massivo é injetado num LLM fine-tunado em jargão clínico.
3. O modelo gera um resumo clínico de 500 palavras.

Em VRAM limitada, a complexidade O(n²) do Self-Attention causa OOM. Este lab demonstra a combinação **QLoRA (4-bit) + KV Cache + FlashAttention-2** que evita o colapso da GPU.

## Stack

- **Modelo gerador:** `Qwen/Qwen2.5-1.5B-Instruct` (janela de 32k tokens, Apache 2.0)
- **Quantização:** `bitsandbytes` 4-bit (NF4 + double quant, compute em float16)
- **Dataset de contexto:** abstracts do PubMed via biblioteca `datasets`
- **Otimizações:** `use_cache=True` (KV Cache) + `attn_implementation="flash_attention_2"` com fallback para `sdpa`

## Execução

Notebook único: `lab10.ipynb`, alvo Google Colab Free (GPU T4, 15GB).
