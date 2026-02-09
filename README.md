# kv-cache-pruner

  `kv-cache-pruner` is a toolkit for **KV cache pruning during long-context LLM inference**.
  It integrates with Hugging Face Transformers attention modules and provides multiple eviction strategies to reduce
  memory usage while preserving generation quality.

  - Loads causal LMs and tokenizers with configurable precision.
  - Patches model attention modules so cache scoring/eviction runs during inference.
  - Builds `past_key_values` objects for different pruning strategies.
  - Supports prefill-only eviction or prefill+decode eviction.
  - Evaluates quality on:
  - Needle-in-a-Haystack passkey retrieval.
  - LongBench tasks.

  ## Core Components
  - `kv_cache_pruner/cache_utils.py`
  - Cache classes and score tracking:
  - `SinkCache`
  - `KVScoreTracker`
  - `kv_cache_pruner/monkey_patch/llama.py`
  - Attention forward implementations with score updates and eviction hooks.
  - `kv_cache_pruner/monkey_patch/utils.py`
  - Model patching helpers:
  - `enable_kv_cache_pruning_flashattn2`
  - `enable_kv_cache_pruning_streamingattn`
  - `kv_cache_pruner/utils.py`
  - Cache factory (`load_kv_cache`).

   ## Environment Setup

  conda activate kv-cache-pruner
  pip install transformers==4.47.0
  pip install flash-attn==2.7.3 --no-build-isolation

  ## Quick Inference Example

  import torch
  from kv_cache_pruner.monkey_patch import enable_kv_cache_pruning

  model, tokenizer = load_model_and_tokenizer(model_name, precision="bf16")

  enable_kv_cache_pruning(model)

      method="kvcV",
      num_recent=16,
      num_heavy=48,
      decode_evict=True,
  )

  prompt = "Summarize the main benefits of cache pruning in one sentence."
  inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

      attention_mask=inputs.attention_mask,
      past_key_values=past_key_values,
      max_new_tokens=64,
      do_sample=False,
  )

  output = tokenizer.decode(
      generated_ids[0][inputs.input_ids.shape[1]:],
      skip_special_tokens=True

  For a full runnable script, see example_generate.py.
  ## Available Cache Methods

  Use load_kv_cache(method=...) with one of:

  - full
  - tova
  - snapkv
  - snapkv_avgpool
  - kvcV, kvcK, kvcVK
  - kvcV+tova, kvcK+tova, kvcVK+tova
  - kvcV+maxpool, kvcK+maxpool, kvcVK+maxpool
  - kvcV+avgpool, kvcK+avgpool, kvcVK+avgpool
  - kvcV_fullhist, kvcK_fullhist, kvcVK_fullhist
  - kvcVK_no_cross and corresponding +tova, +maxpool, +avgpool, _fullhist variants


  - num_heavy
  - heavy_ratio
  - decode_evict
  - fix_recent_token

  ## Evaluation
  ### Needle-in-a-Haystack
  bash scripts/eval_niah.sh

  Input data is expected under:

  - evaluation/needle/data/

  ### LongBench

  bash scripts/eval_longbench.sh

  - evaluation/longbench/config/

  ## Evaluation Entry Points

  - evaluation/needle/pred.py
  - evaluation/needle/eval.py
  - evaluation/longbench/pred.py
  - evaluation/longbench/eval.py

  ## Typical Workflow

  1. Load model and tokenizer.
  2. Apply monkey patch (enable_kv_cache_pruning or flash-attn2 version).
  3. Create cache object with load_kv_cache.
  4. Run generation with past_key_values.
  5. Reset cache between independent samples if needed.

  ## License

  MIT (LICENSE.md)
