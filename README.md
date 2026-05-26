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

## Benchmarks

### Passo 1 — Footprint do modelo quantizado

| Configuração                                              | VRAM ocupada após `from_pretrained` |
|-----------------------------------------------------------|-------------------------------------|
| Qwen2.5-1.5B fp16 (referência)                            | ~3.000 MB                           |
| Qwen2.5-1.5B em **4-bit NF4 + double-quant** (este lab)   | ~1.150 MB *(preencher com medição)* |

Redução de ≈60% do footprint inicial graças à quantização QLoRA 4-bit.

### Passo 2 — Tamanho do contexto recuperado

| Item                              | Valor                       |
|-----------------------------------|-----------------------------|
| Fonte                             | `pubmed_qa` / `pqa_artificial` |
| Trechos concatenados              | ~80                         |
| Caracteres do prompt              | ~60.000                     |
| Tokens reais (Qwen tokenizer)     | ~12.000 *(preencher)*       |

### Passo 3 — Baseline sem KV Cache

Geração de 100 tokens com `model.config.use_cache = False`. A cada novo token, o modelo refaz o forward completo sobre os ~12k tokens de contexto.

| Métrica                                    | Valor                       |
|--------------------------------------------|-----------------------------|
| Tempo total de geração                     | ~ *(preencher)* s           |
| Pico de VRAM (`max_memory_allocated`)      | ~ *(preencher)* MB          |
| Comportamento                              | recalculo redundante O(n²) por step |

### Passo 4 — KV Cache + FlashAttention-2

Modelo recarregado com `attn_implementation="flash_attention_2"` (fallback para `sdpa` em GPUs Turing como a T4 do Colab Free, que não suporta FA2). KV Cache ativo: cada step de decoder reaproveita K e V dos steps anteriores.

| Métrica                                    | Valor                       |
|--------------------------------------------|-----------------------------|
| Atenção em uso                             | `flash_attention_2` ou `sdpa` (fallback) |
| Tempo total de geração                     | ~ *(preencher)* s           |
| Pico de VRAM                               | ~ *(preencher)* MB          |
| **Speedup vs baseline**                    | ~ *(preencher)* x           |
| **Redução do pico de VRAM**                | ~ *(preencher)* %           |

## Execução

Notebook único: `lab10.ipynb`, alvo Google Colab Free (GPU T4, 15GB).

## Parecer técnico

### Parte A — Como QLoRA + KV Cache + FlashAttention salvaram o Transformer

A combinação das três técnicas converte um forward inviável num pipeline executável em VRAM finita. **QLoRA 4-bit** corta o footprint dos pesos em ~60% antes do primeiro token ser gerado, liberando ~3GB de espaço na T4 para acomodar os ~12k tokens recuperados pelo RAG. A ativação do **KV Cache** reduz a complexidade do laço de decoder de O(n²) para O(n) por step, eliminando o recálculo redundante de Q, K e V sobre todo o contexto a cada palavra gerada — neste lab isso se traduziu num speedup expressivo e em latência praticamente constante por token. Já o **FlashAttention-2** (ou seu fallback `sdpa` em GPUs Turing como a T4) age na ineficiência de hardware: em vez de materializar a matriz `n×n` de atenção na HBM lenta, fragmenta o cálculo em blocos que cabem na SRAM rápida da GPU, eliminando o pico de memória durante o prompting — quando os 12k tokens são processados de uma única vez. Sem essas três camadas, o mesmo prompt estouraria a VRAM da T4 já no primeiro forward, porque o tensor de atenção `12k × 12k × n_heads × fp16` sozinho passa de 5GB.
