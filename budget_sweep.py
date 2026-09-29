"""
Pilot experiment for a request-adaptive KV cache budget.

The question this answers
-------------------------
Existing methods (FastKV, RocketKV) apply ONE compression setting to every
request. Does the right setting actually differ from request to request, and by
how much?  If an oracle that picks the budget per request beats a fixed budget
at the SAME average budget, then there is something to predict, and the thesis
has a target. If it does not, the idea is dead and you learn that in two days.

What it does
------------
For each sample of a LongBench task, generate an answer under several KV budgets
(including full KV), score it with the LongBench metric, and record:
  * the score at each budget
  * the output length at each budget
  * a feature vector for the prompt (mean-pooled prefill hidden state), so a
    cheap predictor can later be trained to choose the budget

Compression: SnapKV via NVIDIA kvpress if installed (recommended — this is the
same observation-window selection that FastKV's KV retention and RocketKV's
first stage are built on), otherwise a built-in sink+window fallback.

Install:
  pip install -U transformers accelerate datasets kvpress scikit-learn

Run (RTX 4060, 8 GB):
  python budget_sweep.py --task qasper     --n 60
  python budget_sweep.py --task multi_news --n 60
Then:
  python oracle_analysis.py results_sweep/*.csv
"""

import argparse, csv, json, os, re, string, time
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------------ LongBench

TASKS = {
    # name: (max generated tokens, metric)
    "qasper":        (128, "qa_f1"),
    "hotpotqa":      (32,  "qa_f1"),
    "2wikimqa":      (32,  "qa_f1"),
    "multifieldqa_en": (64, "qa_f1"),
    "multi_news":    (512, "rouge_l"),
    "gov_report":    (512, "rouge_l"),
    "triviaqa":      (32,  "qa_f1"),
}

# Prompts follow the instructions LongBench ships with each task. The
# "do not output any other words" line matters: without it the model answers
# with a sentence and the F1 metric, which compares against a short gold span,
# scores it near zero even when the answer is right.
PROMPT = {
    "qasper": ("You are given a scientific article and a question. Answer the question "
               "as concisely as you can, using a single phrase or sentence if possible. "
               "If the question cannot be answered based on the information in the "
               "article, write \"unanswerable\".\n\nArticle: {context}\n\n"
               "Question: {input}\nAnswer:"),
    "qa_f1": ("Answer the question based on the given passages. Only give me the answer "
              "and do not output any other words.\n\nPassages: {context}\n\n"
              "Question: {input}\nAnswer:"),
    "rouge_l": ("You are given several news passages. Write a one-page summary of all "
                "the news.\n\nNews: {context}\n\nSummary:"),
}


def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1(pred, golds):
    best = 0.0
    for g in golds:
        p, t = normalize(pred).split(), normalize(g).split()
        common = Counter(p) & Counter(t)
        ns = sum(common.values())
        if ns == 0:
            continue
        prec, rec = ns / len(p), ns / len(t)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def lcs(a, b):
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            prev, dp[j] = dp[j], (prev + 1 if x == y else max(dp[j], dp[j - 1]))
    return dp[-1]


def rouge_l(pred, golds):
    best = 0.0
    p = normalize(pred).split()
    for g in golds:
        t = normalize(g).split()
        if not p or not t:
            continue
        l = lcs(p, t)
        if l == 0:
            continue
        prec, rec = l / len(p), l / len(t)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


SCORERS = {"qa_f1": qa_f1, "rouge_l": rouge_l}


# ---------------------------------------------------------------- compression

def load_longbench(task, n, data_dir=None):
    """LongBench ships as a loader script plus data.zip, and `datasets` no longer
    runs loader scripts. So fetch data.zip from the hub once, unpack the task's
    .jsonl, and read it directly. Set --data-dir if you already have the files.

    Behind a slow connection, HF_ENDPOINT=https://hf-mirror.com helps.
    """
    import json, zipfile
    cache = data_dir or os.path.expanduser("~/.cache/longbench_jsonl")
    os.makedirs(cache, exist_ok=True)
    jf = os.path.join(cache, f"{task}.jsonl")

    if not os.path.exists(jf):
        from huggingface_hub import hf_hub_download
        print("downloading LongBench data.zip (once, ~800 MB unpacked) ...", flush=True)
        z = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
        with zipfile.ZipFile(z) as zf:
            members = [m for m in zf.namelist() if m.endswith(f"{task}.jsonl")]
            if not members:
                raise SystemExit(
                    f"{task}.jsonl not found in data.zip; available: "
                    + ", ".join(sorted(os.path.basename(m) for m in zf.namelist()
                                       if m.endswith('.jsonl'))[:25]))
            with zf.open(members[0]) as src, open(jf, "wb") as dst:
                dst.write(src.read())
        print("extracted", jf, flush=True)

    rows = []
    with open(jf, encoding="utf8") as f:
        for line in f:
            if len(rows) >= n:
                break
            d = json.loads(line)
            ans = d.get("answers", [])
            rows.append({"context": d.get("context", ""),
                         "input": d.get("input", ""),
                         "answers": ans if isinstance(ans, list) else [ans]})
    if not rows:
        raise SystemExit(f"no samples read from {jf}")
    print(f"loaded {len(rows)} samples of {task}", flush=True)
    return rows


def make_press(budget, prompt_len, window=32):
    """SnapKV press from kvpress, sized so that ~budget entries survive."""
    from kvpress import SnapKVPress
    ratio = max(0.0, min(0.95, 1.0 - budget / max(1, prompt_len)))
    return SnapKVPress(compression_ratio=ratio, window_size=window)


def get_kv(cache):
    """Key and value tensor lists of a transformers Cache, across versions."""
    if hasattr(cache, "layers") and cache.layers and hasattr(cache.layers[0], "keys"):
        return [l.keys for l in cache.layers], [l.values for l in cache.layers]
    return cache.key_cache, cache.value_cache


def set_kv(cache, keys, values):
    if hasattr(cache, "layers") and cache.layers and hasattr(cache.layers[0], "keys"):
        for layer, k, v in zip(cache.layers, keys, values):
            layer.keys, layer.values = k, v
    else:
        cache.key_cache, cache.value_cache = keys, values


def cache_len(cache):
    return get_kv(cache)[0][0].shape[-2]


# def fallback_evict(cache, budget, sink=4):
#     """Sink + recent window eviction, used when kvpress is not installed.

#     Keys are stored after rotary embedding, so dropping rows is true eviction:
#     every surviving entry keeps the position it was computed with.
#     """
#     keys, values = get_kv(cache)
#     L = keys[0].shape[-2]
#     if L <= budget or budget <= sink:
#         return
#     win = budget - sink                       # size of the recent window
#     # Plain slicing, no index kernel: [sink oldest] + [win newest]
#     set_kv(cache,
#            [torch.cat([k[..., :sink, :], k[..., L - win:, :]], dim=-2).contiguous()
#             for k in keys],
#            [torch.cat([v[..., :sink, :], v[..., L - win:, :]], dim=-2).contiguous()
#             for v in values])
#     # keep the cache's own length counter consistent with the tensors
#     if hasattr(cache, "_seen_tokens"):
#         cache._seen_tokens = budget



def keydiff_evict(cache, budget, sink=4, recent=32):
    """
    KeyDiff Eviction (Option B): Keeps attention sinks, the recent question, 
    and uses vector norms to save the most important tokens from the document body.
    """
    keys, values = get_kv(cache)
    L = keys[0].shape[-2]
    
    # If the document is already smaller than our budget, do nothing
    if L <= budget or budget <= (sink + recent):
        return
        
    keep_middle = budget - sink - recent
    new_keys, new_values = [], []
    
    for k, v in zip(keys, values):
        # 1. Grab the Sink tokens (first few words)
        k_sink, v_sink = k[..., :sink, :], v[..., :sink, :]
        
        # 2. Grab the Recent tokens (the question at the end)
        k_recent, v_recent = k[..., L - recent:, :], v[..., L - recent:, :]
        
        # 3. Look at the Document (the middle)
        k_mid, v_mid = k[..., sink:L - recent, :], v[..., sink:L - recent, :]
        
        # KEYDIFF MATH: Calculate the L2 norm (size) of the key vectors
        # We average across the attention heads to get one score per word
        norms = k_mid.norm(p=2, dim=-1).mean(dim=1) 
        
        # Find the index numbers of the tokens with the biggest vectors
        _, top_idx = torch.topk(norms, keep_middle, dim=-1)
        
        # Sort the index numbers so the words stay in their original chronological order!
        top_idx, _ = torch.sort(top_idx, dim=-1)
        
        # Expand the indices to gather the actual Key and Value data
        gather_idx = top_idx.unsqueeze(1).unsqueeze(-1).expand(-1, k.shape[1], -1, k.shape[-1])
        k_mid_kept = torch.gather(k_mid, 2, gather_idx)
        v_mid_kept = torch.gather(v_mid, 2, gather_idx)
            
        # 4. Stitch them all back together (Sink + Important Middle + Recent)
        new_k = torch.cat([k_sink, k_mid_kept, k_recent], dim=-2).contiguous()
        new_v = torch.cat([v_sink, v_mid_kept, v_recent], dim=-2).contiguous()
        
        new_keys.append(new_k)
        new_values.append(new_v)
        
    set_kv(cache, new_keys, new_values)
    if hasattr(cache, "_seen_tokens"):
        cache._seen_tokens = budget

# --------------------------------------------------------------------- runner

def eos_ids_of(model, tok):
    """Every id that should stop generation. Qwen2.5-Instruct ends turns with
    <|im_end|> but can also emit <|endoftext|>; the manual loop must use the
    same stop set as generate(), or the two paths produce different lengths
    and the length comparison is meaningless."""
    ids = set()
    for src in (tok.eos_token_id, getattr(model.generation_config, "eos_token_id", None)):
        if isinstance(src, (list, tuple)):
            ids.update(int(i) for i in src)
        elif src is not None:
            ids.add(int(src))
    return ids


@torch.inference_mode()
def run_one(model, tok, prompt, budget, max_new, use_press, eos=None):
    eos = eos or eos_ids_of(model, tok)
    if getattr(tok, "chat_template", None):
        # Instruct models answer concisely and emit a stop token only when the
        # prompt is wrapped in their chat template.
        prompt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                         tokenize=False, add_generation_prompt=True)
    tok.truncation_side = "left"          # keep the question and the generation prompt
    ids = tok(prompt, return_tensors="pt", truncation=True, add_special_tokens=False,
              max_length=model.config.max_position_embeddings - max_new - 8).to(model.device)
    n_prompt = ids.input_ids.shape[-1]
    t0 = time.time()

    if use_press and budget != 0:
        press = make_press(budget, n_prompt)
        with press(model):
            out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                                 eos_token_id=sorted(eos), pad_token_id=tok.eos_token_id)
        gen = out[0, n_prompt:]
        feat = np.zeros(model.config.hidden_size, dtype="float32")
    else:
        # One decode path for every budget, including full KV (budget == 0
        # simply never evicts). Same loop, same stop rule, so the lengths of
        # the compressed and uncompressed runs are directly comparable.
        from transformers import DynamicCache
        cache = DynamicCache()
        # Run the body of the model, not the LM head: the head would produce
        # logits for EVERY prompt position and cast them to float32, which is
        # about 4 GB for a 7K-token prompt on this vocabulary and is what ran
        # the 8 GB card out of memory. Only the last position is needed.
        hs = model.model(**ids, past_key_values=cache, use_cache=True).last_hidden_state
        feat = hs[0].float().mean(0).cpu().numpy()          # prompt representation
        logits = model.lm_head(hs[:, -1:, :])[:, -1, :].float()
        del hs
        if budget:
            keydiff_evict(cache, budget)
        toks, pos = [], n_prompt
        for _ in range(max_new):
            nxt = int(torch.argmax(logits, -1))
            if nxt in eos:
                break
            toks.append(nxt)
            # Positions are assigned by slot in the (trimmed) cache, not by
            # position in the original text. This is the StreamingLLM
            # convention [6]: "we assign positions by their position in the
            # cache rather than in the original text." Letting transformers
            # derive the position from the cache length gives exactly that.
            o = model(input_ids=torch.tensor([[nxt]], device=model.device),
                      past_key_values=cache, use_cache=True)
            logits, pos = o.logits[:, -1, :], pos + 1
            # if budget:
            #     keydiff_evict(cache, budget)
        gen = torch.tensor(toks, dtype=torch.long)
        del cache

    text = tok.decode(gen, skip_special_tokens=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()          # long prompts fragment 8 GB quickly
    return text, int(len(gen)), n_prompt, time.time() - t0, feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--task", default="qasper", choices=list(TASKS))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--budgets", default="0,128,256,512,1024")   # 0 = full KV
    ap.add_argument("--out", default="results_sweep")
    ap.add_argument("--data-dir", default=None,
                    help="folder holding LongBench <task>.jsonl files, if you downloaded them by hand")
    args = ap.parse_args()

    max_new, metric = TASKS[args.task]
    budgets = [int(b) for b in args.budgets.split(",")]

    ds = load_longbench(args.task, args.n, args.data_dir)

    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="sdpa",
        device_map="auto" if torch.cuda.is_available() else None).eval()

    for k in ("temperature", "top_p", "top_k"):      # silence sampling warnings
        if hasattr(model.generation_config, k):
            setattr(model.generation_config, k, None)

    try:
        import kvpress  # noqa: F401
        use_press = True # FORCE THIS TO FALSE
        print("using custom KeyDiff eviction") # <--- UPDATE THIS PRINT
    except ImportError:
        use_press = True
        print("using custom KeyDiff eviction") # <--- UPDATE THIS PRINT TOO

    eos = eos_ids_of(model, tok)
    print("stop token ids:", sorted(eos))

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.task}_{os.path.basename(args.model)}.csv")
    feats = {}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "task", "budget", "score", "out_tokens", "prompt_tokens", "seconds"])
        for i, d in enumerate(ds):
            template = PROMPT.get(args.task, PROMPT[metric])
            ctx = template.format(context=d["context"], input=d.get("input", ""))
            golds = d["answers"] if isinstance(d["answers"], list) else [d["answers"]]
            for b in budgets:
                text, n_out, n_prompt, dt, feat = run_one(model, tok, ctx, b, max_new,
                                                          use_press, eos)
                s = SCORERS[metric](text, golds)
                w.writerow([i, args.task, b, round(s, 4), n_out, n_prompt, round(dt, 1)])
                f.flush()
                # keep the generations: needed to quote a representative failure
                # in the proposal and to check that low scores are real
                with open(os.path.join(args.out, f"{args.task}_gen.jsonl"), "a") as g:
                    g.write(json.dumps({"idx": i, "budget": b, "score": round(s, 4),
                                        "gold": golds[:3], "text": text[:1500]}) + "\n")
                if b == 0:
                    feats[i] = feat
                print(f"[{i+1}/{len(ds)}] budget={b or 'full'} score={s:.3f} "
                      f"len={n_out} prompt={n_prompt} {dt:.0f}s", flush=True)
    np.save(os.path.join(args.out, f"{args.task}_feats.npy"),
            np.stack([feats[i] for i in sorted(feats)]))
    print("wrote", path)


if __name__ == "__main__":
    main()
