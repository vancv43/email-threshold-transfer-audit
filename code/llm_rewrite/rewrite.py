"""Rewrite spam bodies with one open LLM (G1 or G2 of PROTOCOL.md).

Usage: python rewrite.py <g1|g2> <model_dir> <input_parquet> <output_dir>
vLLM is used when it is already installed and starts on the GPU; otherwise Hugging Face generate runs with the
same prompt and sampling settings. Generation stops after BUDGET_S seconds and saves the finished rows. The backend and library versions are written to the summary.
Links are kept verbatim so that only the wording changes. Refusals are flagged and excluded later.
"""
import json, re, sys, time, pathlib
import pandas as pd

GEN, MODEL, INP, OUT = sys.argv[1], sys.argv[2], sys.argv[3], pathlib.Path(sys.argv[4])
SYSTEM = "You paraphrase emails for a study that measures how well spam filters handle reworded messages."
USER = ("Rewrite the email below. Keep its purpose, its call to action, and every link exactly as written. "
        "Change the wording, sentence structure, and tone so that it reads as fluent, natural English. "
        "Output only the rewritten email body.\n\nEMAIL:\n{body}")
REFUSAL = re.compile(r"^\s*(i can['’]?t|i cannot|i'm sorry|i am sorry|sorry, i|i won['’]?t)", re.I)
TEMP, TOP_P, MAX_NEW, SEED = 0.7, 0.9, 1024, 0
BUDGET_S = 10.5 * 3600  # the GPU server sessions end at 12 h
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(INP)


def build_prompts(tok):
    def clip(text, n=1500):
        return tok.decode(tok.encode(text, add_special_tokens=False)[:n])
    def chat(body):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.format(body=clip(body))}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:  # templates without a system role, e.g. Mistral: put the system text first in the user turn
            msgs = [{"role": "user", "content": SYSTEM + "\n\n" + msgs[1]["content"]}]
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return [chat(b) for b in df["body"]]


def run_vllm():
    import vllm
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, dtype="float16", tensor_parallel_size=2, max_model_len=4096, seed=SEED,
              gpu_memory_utilization=0.9)
    prompts = build_prompts(llm.get_tokenizer())
    outs = llm.generate(prompts, SamplingParams(temperature=TEMP, top_p=TOP_P, max_tokens=MAX_NEW, seed=SEED))
    return ([o.outputs[0].text.strip() for o in outs], [len(o.outputs[0].token_ids) for o in outs],
            {"backend": "vllm", "vllm": vllm.__version__})


def run_hf(batch=16):
    import torch, transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="auto").eval()
    prompts = build_prompts(tok)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # similar lengths per batch
    text, ntok = [None] * len(prompts), [0] * len(prompts)
    torch.manual_seed(SEED)
    for b in range(0, len(order), batch):
        if time.time() - t0 > BUDGET_S:
            break
        idx = order[b:b + batch]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=True, temperature=TEMP, top_p=TOP_P, max_new_tokens=MAX_NEW,
                                 pad_token_id=tok.pad_token_id)
        for i, g in zip(idx, gen[:, enc["input_ids"].shape[1]:]):
            g = g[g != tok.pad_token_id]
            text[i], ntok[i] = tok.decode(g, skip_special_tokens=True).strip(), int(len(g))
        print(f"{min(b + batch, len(order))}/{len(order)}", flush=True)
    return text, ntok, {"backend": "transformers", "transformers": transformers.__version__, "batch": batch}


t0 = time.time()
try:
    text, ntok, info = run_vllm()
except Exception as e:  # vLLM missing or unable to start on this GPU
    print(f"vLLM unavailable: {e!r}; using transformers", flush=True)
    text, ntok, info = run_hf()
done = [t is not None for t in text]
df, text, ntok = df[done], [t for t in text if t is not None], [n for n, d in zip(ntok, done) if d]
res = pd.DataFrame({"msg_id": df["msg_id"], "generator": GEN, "model": pathlib.Path(MODEL).name,
                    "rewritten": text, "refusal": [bool(REFUSAL.match(t)) for t in text], "n_out_tokens": ntok})
res.to_parquet(OUT / f"rewrite_{GEN}.parquet", index=False)
summary = {"generator": GEN, "model_dir": MODEL, "n": len(res), "refusals": int(res["refusal"].sum()),
           "complete": bool(all(done)), "mean_out_tokens": float(res["n_out_tokens"].mean()), "seconds": round(time.time() - t0, 1),
           "sampling": {"temperature": TEMP, "top_p": TOP_P, "max_tokens": MAX_NEW, "seed": SEED}, **info}
(OUT / f"rewrite_{GEN}_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
