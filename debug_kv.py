"""
Isolate which step triggers the CUDA assert.

Run:  CUDA_LAUNCH_BLOCKING=1 python debug_kv.py

Each step prints PASS or the exact exception. With CUDA_LAUNCH_BLOCKING=1 the
error is raised at the operation that caused it instead of at the next sync.
"""
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
PROMPT_TOKENS = 3900
BUDGET, SINK = 128, 4

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
print("model loaded:", model.config.model_type,
      "layers", model.config.num_hidden_layers,
      "kv heads", getattr(model.config, "num_key_value_heads", "?"),
      "vocab", model.config.vocab_size)

text = "The quick brown fox jumps over the lazy dog. " * 500
ids = tok(text, return_tensors="pt", truncation=True,
          max_length=PROMPT_TOKENS).to("cuda")
n_prompt = ids.input_ids.shape[-1]
print("prompt tokens:", n_prompt,
      "max id:", int(ids.input_ids.max()), "/ vocab", model.config.vocab_size)


def step(name, fn):
    try:
        out = fn()
        torch.cuda.synchronize()
        print(f"PASS  {name}")
        return out
    except Exception as e:
        print(f"FAIL  {name}\n      {type(e).__name__}: {e}")
        raise SystemExit(1)


with torch.inference_mode():
    # 1. plain forward, no cache object of our own
    step("forward without explicit cache",
         lambda: model(**ids, use_cache=False))

    # 2. forward writing into a DynamicCache we created
    cache = DynamicCache()
    o = step("prefill into DynamicCache",
             lambda: model(**ids, past_key_values=cache, use_cache=True))
    L0 = cache.key_cache[0].shape[-2]
    print("      cache length after prefill:", L0)

    # 3. hidden states requested (used for the feature vector)
    cache2 = DynamicCache()
    step("prefill with output_hidden_states=True",
         lambda: model(**ids, past_key_values=cache2, use_cache=True,
                       output_hidden_states=True))

    # 4. trim the cache by slicing
    def trim():
        win = BUDGET - SINK
        cache.key_cache = [torch.cat([k[..., :SINK, :], k[..., -win:, :]], -2).contiguous()
                           for k in cache.key_cache]
        cache.value_cache = [torch.cat([v[..., :SINK, :], v[..., -win:, :]], -2).contiguous()
                             for v in cache.value_cache]
        return cache.key_cache[0].shape[-2]
    L1 = step("trim cache to budget", trim)
    print("      cache length after trim:", L1)

    nxt = int(torch.argmax(o.logits[:, -1, :], -1))
    print("      first token id:", nxt)

    # 5. decode step: mask sized to the trimmed cache, true absolute position
    step("decode step (mask=cache+1, position_ids=absolute)",
         lambda: model(input_ids=torch.tensor([[nxt]], device="cuda"),
                       attention_mask=torch.ones(1, L1 + 1, dtype=torch.long, device="cuda"),
                       position_ids=torch.tensor([[n_prompt]], device="cuda"),
                       past_key_values=cache, use_cache=True))

    # 6. same, but position_ids follow the SHRUNKEN cache instead
    cache3 = DynamicCache()
    model(**ids, past_key_values=cache3, use_cache=True)
    win = BUDGET - SINK
    cache3.key_cache = [torch.cat([k[..., :SINK, :], k[..., -win:, :]], -2).contiguous()
                        for k in cache3.key_cache]
    cache3.value_cache = [torch.cat([v[..., :SINK, :], v[..., -win:, :]], -2).contiguous()
                          for v in cache3.value_cache]
    step("decode step (position_ids=cache length)",
         lambda: model(input_ids=torch.tensor([[nxt]], device="cuda"),
                       attention_mask=torch.ones(1, BUDGET + 1, dtype=torch.long, device="cuda"),
                       position_ids=torch.tensor([[BUDGET]], device="cuda"),
                       past_key_values=cache3, use_cache=True))

print("\nAll steps passed — the failure is elsewhere in budget_sweep.py")
