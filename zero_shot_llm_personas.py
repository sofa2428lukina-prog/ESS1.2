"""Zero-shot LLM predictions for ESS WITH PERSONA VARIANTS (full / reduced / none / shuffled).
Outputs: llm_predictions_a1.csv, llm_predictions_a2.csv (column `variant`), llm_run_summary.json.

Based on zero_shot_llm.py (A1: household food spending, A2: job transition).

Based on llm_benchmark.py. Differences:
  * strictly zero-shot: no examples from RLMS in any prompt (asserted below);
  * larger test samples (N_A1 / N_A2) instead of 150 / 400;
  * per-row predictions are SAVED (needed for paired LLM-vs-ML comparison);
  * greedy decoding by default (a fixed LLM = a fixed set of predictions);
  * optional Qwen2.5-7B in 4-bit.
Outputs (in /kaggle/working): llm_predictions_a1.csv, llm_predictions_a2.csv, llm_run_summary.json
Quick check first:  SMOKE=1  (30 rows, first model only).
"""
import gc
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ----------------------------- config ---------------------------------------
INPUT_ROOT = Path(os.environ.get("INPUT_ROOT", "/kaggle/input"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working" if Path("/kaggle/working").exists() else "."))
SMOKE = os.environ.get("SMOKE") == "1"
N_A1, N_A2 = (30, 30) if SMOKE else (1000, 1500)
SAMPLE_SEED = 2026
SEEDS = [17]                 # with DO_SAMPLE=False the seed only matters for reproducibility bookkeeping
DO_SAMPLE = False            # greedy decoding: fixed predictions per prompt
# (hf name, short name, load 4-bit?)
MODELS = [
    ("Qwen/Qwen2.5-1.5B-Instruct", "qwen1.5b", False),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "smollm1.7b", False),
    ("Qwen/Qwen2.5-7B-Instruct", "qwen7b", True),
]
if SMOKE:
    MODELS = MODELS[:1]
INCLUDE_MEASURED_BIG_FIVE = False   # Big Five was measured at or before t in this data (checked), so it is allowed;
                                    # but the main ESS question is the plain persona. Set True for an extra variant.
BATCH = {False: 16, True: 8}
CLASSES = ["different_job", "no_job", "same_job"]
TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
OCCUPATIONS = {0: "military", 1: "manager", 2: "professional", 3: "associate professional", 4: "clerical",
               5: "service or sales", 6: "agricultural", 7: "craft", 8: "machine operator", 9: "elementary occupation"}
SETTLEMENTS = {1: "regional capital", 2: "city", 3: "small town", 4: "rural"}


# ----------------------------- prompts (identical to llm_benchmark.py) -------
def value(row, name, digits=0):
    item = row.get(name)
    return "unknown" if pd.isna(item) else str(round(float(item), digits))


def education(code):
    if pd.isna(code): return "unknown"
    code = int(code)
    if code <= 9: return "primary or lower secondary"
    if code <= 17: return "secondary or vocational, no tertiary diploma"
    if code == 18: return "vocational diploma"
    if code <= 20: return "some higher education"
    return "university or postgraduate diploma"


def profile(row):
    sex = "male" if row.sex == 1 else "female"
    marital = "married or partnered" if row.marital in [2, 3] else "not partnered"
    occupation = OCCUPATIONS.get(int(row.occupation), "unknown") if pd.notna(row.occupation) else "unknown"
    settlement = SETTLEMENTS.get(int(row.settlement), "unknown") if pd.notna(row.settlement) else "unknown"
    return (
        f"year={int(row.year)}; age={value(row, 'age')}; sex={sex}; education={education(row.education)}; "
        f"settlement={settlement}; marital={marital}; occupation={occupation}; monthly_wage={value(row, 'wage')}; "
        f"monthly_hours={value(row, 'hours')}; job_tenure_years={value(row, 'tenure', 1)}; "
        f"observed_history_years={value(row, 'history_years')}; wage_log_change={value(row, 'wage_change', 2)}; "
        f"occupation_changed_last_year={value(row, 'occupation_changed')}; prior_job_change={value(row, 'prior_job_change')}")


A1_FIELDS = {
    "full": ["current_food", "income", "household_size", "employed", "children", "adults", "previous_food", "previous_income"],
    "reduced": ["current_food", "household_size"],
    "none": [],
}
A1_VALUE = {"current_food": "food", "income": "income", "household_size": "hh_size", "employed": "employed_count",
            "children": "children_count", "adults": "adult_count", "previous_food": "prev_food", "previous_income": "prev_income"}


def a1_prompt(row, persona="full", source=None):
    """persona: full | reduced | none | shuffled. For shuffled the household features come from `source` (another random row);
    the year always comes from the target row."""
    src = row if persona != "shuffled" else source
    fields = A1_FIELDS["full" if persona == "shuffled" else persona]
    parts = [f"year={int(row.year)}"] + [f"{name}={value(src, A1_VALUE[name])}" for name in fields]
    return ("Forecast next year's monthly household food spending in rubles. Return only JSON: {\"food\": number}. "
            "Use only the information below.\n" + "; ".join(parts) + ".")


A2_REDUCED = ["age", "sex", "education"]


def a2_profile(row, persona):
    if persona == "none":
        return f"year={int(row.year)}"
    if persona == "reduced":
        sex = "male" if row.sex == 1 else "female"
        return f"year={int(row.year)}; age={value(row, 'age')}; sex={sex}; education={education(row.education)}"
    return profile(row)


def a2_prompt(row, persona="full", source=None, traits=None):
    src = row if persona != "shuffled" else source
    text = a2_profile(src, "full" if persona == "shuffled" else persona)
    if persona == "shuffled":      # keep the target row's own survey year, everything personal is someone else's
        text = text.replace(f"year={int(src.year)}", f"year={int(row.year)}", 1)
    return (
        "Forecast the person's status in the next annual survey. Return only JSON probabilities that sum to 1: "
        "{\"same_job\": number, \"different_job\": number, \"no_job\": number}. "
        "same_job means same main workplace; different_job means another main workplace; no_job means no main job.\n" + text)


def assert_zero_shot(prompts):
    """No RLMS examples may appear in a zero-shot prompt."""
    bad = [p for p in prompts if "Training examples" in p or "\nOutput:" in p or "Input:" in p]
    assert not bad, "few-shot content found in a zero-shot prompt"


# ----------------------------- generation / parsing --------------------------
def load_model(hf_name, quant):
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if quant:
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(hf_name, quantization_config=cfg, device_map="auto").eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(hf_name, torch_dtype=torch.float16).to("cuda").eval()
    return model, tokenizer


def generate(model, tokenizer, prompts, seed, batch_size):
    assert_zero_shot(prompts)
    torch.manual_seed(seed)
    kwargs = dict(max_new_tokens=80, pad_token_id=tokenizer.pad_token_id)
    kwargs.update(dict(do_sample=True, temperature=0.2, top_p=0.9) if DO_SAMPLE else dict(do_sample=False))
    answers = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in batch]
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        with torch.inference_mode():
            output = model.generate(**encoded, **kwargs)
        answers.extend(tokenizer.batch_decode(output[:, encoded.input_ids.shape[1]:], skip_special_tokens=True))
        if (start // batch_size) % 20 == 0:
            print(f"  {start + len(batch)}/{len(prompts)}", flush=True)
    return answers


def json_object(text):
    match = re.search(r"\{.*?\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:
        return None


def parse_food(text):
    try:
        return max(0.0, float(str(json_object(text)["food"]).replace(" ", "")))
    except Exception:
        return np.nan


def parse_probabilities(text):
    data = json_object(text)
    try:
        values = np.array([max(0.0, float(data[key])) for key in CLASSES])
        return values / values.sum() if values.sum() > 0 else None
    except Exception:
        return None


# ----------------------------- main ------------------------------------------
PERSONAS = ["full", "reduced", "none", "shuffled"]


def find(name):
    paths = list(INPUT_ROOT.rglob(name))
    assert len(paths) == 1, (name, paths)
    return paths[0]


def shuffled_sources(n, seed):
    """Derangement-like permutation: each row gets another row's persona (never its own)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    for i in np.where(perm == np.arange(n))[0]:
        j = (i + 1) % n
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def main():
    a1, a2 = pd.read_csv(find("a1_pairs.csv")), pd.read_csv(find("a2_pairs.csv"))
    a1_sample = a1[a1.target_year >= 2024].sample(N_A1, random_state=SAMPLE_SEED).reset_index(drop=True)
    a2_sample = a2[a2.target_year >= 2024].dropna(subset=["target"]).sample(N_A2, random_state=SAMPLE_SEED).reset_index(drop=True)
    perm1, perm2 = shuffled_sources(len(a1_sample), SAMPLE_SEED), shuffled_sources(len(a2_sample), SAMPLE_SEED + 1)
    if any(q for _, _, q in MODELS):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes>=0.46.1"], check=True)

    a1_rows, a2_rows, summary = [], [], {"n_a1": len(a1_sample), "n_a2": len(a2_sample), "do_sample": DO_SAMPLE, "models": {}}
    for hf_name, short, quant in MODELS:
        print("MODEL", hf_name, "quant4bit=", quant, flush=True)
        model, tokenizer = load_model(hf_name, quant)
        bs, seed = BATCH[quant], SEEDS[0]
        summary["models"][short] = {"quantized_4bit": quant, "a1_parse_rate": {}, "a2_parse_rate": {}}
        for persona in PERSONAS:
            p1 = [a1_prompt(r, persona, a1_sample.iloc[perm1[i]]) for i, (_, r) in enumerate(a1_sample.iterrows())]
            texts = generate(model, tokenizer, p1, seed, bs)
            ok = 0
            for (_, r), t in zip(a1_sample.iterrows(), texts):
                food = parse_food(t); ok += np.isfinite(food)
                a1_rows.append({"row_id": int(r.row_id), "model": short, "seed": seed, "variant": persona, "pred_food": food, "raw": t.strip()[:200]})
            summary["models"][short]["a1_parse_rate"][persona] = ok / len(a1_sample)

            p2 = [a2_prompt(r, persona, a2_sample.iloc[perm2[i]]) for i, (_, r) in enumerate(a2_sample.iterrows())]
            texts = generate(model, tokenizer, p2, seed, bs)
            ok = 0
            for (_, r), t in zip(a2_sample.iterrows(), texts):
                p = parse_probabilities(t); ok += p is not None
                a2_rows.append({"row_id": int(r.row_id), "model": short, "seed": seed, "variant": persona,
                                **{f"p_{c}": (np.nan if p is None else float(p[i])) for i, c in enumerate(CLASSES)}, "raw": t.strip()[:200]})
            summary["models"][short]["a2_parse_rate"][persona] = ok / len(a2_sample)
            # save after every persona so a crash does not lose finished work
            pd.DataFrame(a1_rows).to_csv(OUT_DIR / "llm_predictions_a1.csv", index=False)
            pd.DataFrame(a2_rows).to_csv(OUT_DIR / "llm_predictions_a2.csv", index=False)
            (OUT_DIR / "llm_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        del model, tokenizer
        gc.collect(); torch.cuda.empty_cache()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
