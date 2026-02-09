# kv-cache-pruner

`kv-cache-pruner` is a renamed, functionality-equivalent derivative of OBCache focused on one goal: **prune KV cache for long-context LLM inference efficiency**. 🧠

It keeps the same core behavior:
- KV cache eviction/pruning during generation
- H2O/TOVA/SnapKV-style variants
- evaluation pipelines for Needle-in-a-Haystack and LongBench 📊

## What Changed

- Repo name: `OBCache` -> `kv-cache-pruner`
- Code package: `obc/` -> `kv_cache_pruner/`
- Main class naming:
  - `OBCache` -> `KVCachePruner`
  - `OBCScoreTracker` -> `KVScoreTracker`
- Monkey patch naming:
  - `enable_optimal_brain_kv` -> `enable_kv_cache_pruning`
  - `enable_optimal_brain_kv_flashattn2` -> `enable_kv_cache_pruning_flashattn2`
- Method tags renamed for clarity:
  - `obcV/obcK/obcVK` -> `kvcV/kvcK/kvcVK`
  - backward compatibility for old `obc*` tags is preserved in `load_kv_cache()`

## Environment Setup

```bash
conda create -n kv-cache-pruner python=3.12
conda activate kv-cache-pruner

pip install transformers==4.47.0
pip install flash-attn==2.7.3 --no-build-isolation
```

## Inference Example

```python
from kv_cache_pruner.monkey_patch.utils import enable_kv_cache_pruning
from kv_cache_pruner.utils import load_kv_cache, load_model_and_tokenizer

model_name = "meta-llama/Llama-3.1-8B-Instruct"
model, tokenizer = load_model_and_tokenizer(model_name)
enable_kv_cache_pruning(model)

past_key_values = load_kv_cache(method="kvcV", num_recent=16, num_heavy=48)
prompt = "YOUR PROMPT"
model_inputs = tokenizer([prompt], return_tensors="pt")
generated_ids = model.generate(
    **model_inputs, max_new_tokens=512, past_key_values=past_key_values
)
generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

For a full demo, see `example_generate.py`.

## Evaluation

### Needle-In-A-Haystack

```bash
bash scripts/eval_niah.sh
```

### LongBench

```bash
bash scripts/eval_longbench.sh
```

## Notes

- Existing `obc*` method names still work for compatibility.
- Existing `enable_optimal_brain_kv*` function names are kept as aliases.

## License

MIT (`LICENSE.md`).

## Credits

Built and maintained by our team. 🚀
