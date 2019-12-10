"""Third quick script to analyze the original method of evaluating agreement predictions."""

import os

import fire
import numpy as np
import pandas as pd
from tqdm import tqdm

from filenames import CLOZE_DIR, FEATURES_DIR, PROBABILITIES_DIR

cols = ["number", "gender", "case", "person"]

def run(lg):
    results = []
    features_filename = os.path.join(FEATURES_DIR, f"{lg}.csv")
    features = pd.read_csv(features_filename, dtype={"person": str})
    cloze_filename = os.path.join(CLOZE_DIR, f"{lg}.csv")
    cloze = pd.read_csv(cloze_filename)
    for _, row in tqdm(cloze.iterrows()):
        uid = row["uid"]
        probabilities_filename = os.path.join(PROBABILITIES_DIR, lg, f"{uid}.csv")
        try:  # we may have skipped this cloze example
            probs = pd.read_csv(probabilities_filename)
            pos = row["pos"]
            lemma = row["lemma"]
            correct_form = row["correct_form"]
            p_correct_form = probs[probs["word"] == correct_form]["p"].max()
            if np.isnan(
                p_correct_form
            ):  # the correct form didn't appear in the lexicon
                continue
            else:
                is_same_lemma = features["lemma"] == lemma
                is_same_pos = features["pos"] == pos
                is_incorrect_form = features["word"] != correct_form
                incorrect_forms = features[
                    is_same_lemma & is_same_pos & is_incorrect_form
                ]["word"]
                if (
                    incorrect_forms.empty
                ):  # we don't have feature data on any incorrect forms
                    continue
                else:
                    probs_incorrect_forms = probs[probs["word"].isin(incorrect_forms)]
                    p_incorrect_form = probs_incorrect_forms["p"].max()
                    if np.isnan(
                        p_incorrect_form
                    ):  # no incorrect forms appear in the lexicon
                        continue
                    else:
                        old_right = p_correct_form > p_incorrect_form
                        example = {
                            "lg": lg,
                            "uid": uid,
                            "masked": row["masked"],
                            "old_right": old_right
                        }
                        results.append(example)
        except FileNotFoundError:
            continue
    results = pd.DataFrame(results)
    results.to_csv(f"data/previous/{lg}.csv", index=False)


if __name__ == "__main__":
    fire.Fire(run)

