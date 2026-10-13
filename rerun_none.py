"""Перезапуск ТОЛЬКО варианта persona = none с промптом, который не даёт модели отказаться.

Зачем: у Qwen 7B в варианте `none` parse_rate был 0.0: модель писала «мне нужно больше данных» вместо числа.
Здесь в промпт добавлено: «персональных данных нет, дай оценку для типичного респондента, не отказывайся».
Промпт одинаковый для обеих моделей (1.5B и 7B), чтобы сравнение было честным.

Запуск в Colab (файл должен лежать рядом с zero_shot_llm_personas.py и a1_pairs.csv, a2_pairs.csv):
    %env INPUT_ROOT=/content
    %env SMOKE=0
    !python rerun_none.py
Результат: /content/llm_predictions_none_a1.csv и /content/llm_predictions_none_a2.csv
"""
import os
import shutil
from pathlib import Path

TMP = Path(os.environ.get("NONE_TMP", "/content/none_tmp"))
TMP.mkdir(parents=True, exist_ok=True)
os.environ["OUT_DIR"] = str(TMP)          # писать во временную папку, чтобы не затереть основные файлы

import zero_shot_llm_personas as z         # noqa: E402

_A1_NONE = ("Forecast next year's monthly household food spending in rubles for a typical household in a Russian household panel survey. "
            "You have no details about this household, so give your best estimate for a typical one. "
            "Always answer, never ask for more information. Return only JSON: {\"food\": number}.\n")
_A2_NONE = ("Forecast the status of a typical respondent of a Russian household panel survey in the next annual survey. "
            "You have no personal details about this person, so give your best probabilities for a typical respondent. "
            "Always answer, never ask for more information. "
            "Return only JSON probabilities that sum to 1: {\"same_job\": number, \"different_job\": number, \"no_job\": number}. "
            "same_job means same main workplace; different_job means another main workplace; no_job means no main job.\n")

_orig_a1, _orig_a2 = z.a1_prompt, z.a2_prompt


def a1_prompt(row, persona="full", source=None):
    if persona == "none":
        return _A1_NONE + f"year={int(row.year)}."
    return _orig_a1(row, persona, source)


def a2_prompt(row, persona="full", source=None, traits=None):
    if persona == "none":
        return _A2_NONE + f"year={int(row.year)}"
    return _orig_a2(row, persona, source, traits)


z.a1_prompt, z.a2_prompt = a1_prompt, a2_prompt
z.PERSONAS = ["none"]

if __name__ == "__main__":
    z.main()
    out = Path(os.environ.get("FINAL_DIR", "."))
    for task in ("a1", "a2"):
        shutil.copy(TMP / f"llm_predictions_{task}.csv", out / f"llm_predictions_none_{task}.csv")
    shutil.copy(TMP / "llm_run_summary.json", out / "llm_run_summary_none.json")
    print("ГОТОВО. Скачай: llm_predictions_none_a1.csv, llm_predictions_none_a2.csv, llm_run_summary_none.json")
