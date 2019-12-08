"""Second quick script to analyze the original method of evaluating agreement predictions."""

import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from filenames import CLOZE_DIR, FEATURES_DIR, PROBABILITIES_DIR

cols = ["number", "gender", "case", "person"]
languages = os.listdir(PROBABILITIES_DIR)
for lg in ["hun", "gle", "fin"]:  # already done
    languages.remove(lg)
for lg in languages:
    results = []
    print(lg)
    features_filename = os.path.join(FEATURES_DIR, f"{lg}.csv")
    features = pd.read_csv(features_filename, dtype={"person": str})
    cloze_filename = os.path.join(CLOZE_DIR, f"{lg}.csv")
    cloze = pd.read_csv(cloze_filename)
    for _, row in tqdm(cloze.iterrows()):
        uid = row["uid"]
        pos = row["pos"]
        probabilities_filename = os.path.join(PROBABILITIES_DIR, lg, f"{uid}.csv")
        try:  # we may have skipped this cloze example
            probs = pd.read_csv(probabilities_filename)
            lemma = row["lemma"]
            correct_form = row["correct_form"]
            p_correct_form = probs[probs["word"] == correct_form]["p"].max()
            if np.isnan(
                p_correct_form
            ):  # the correct form didn't appear in the lexicon
                continue
            else:
                is_same_pos = features["pos"] == pos
                is_different_lemma = features["lemma"] != lemma
                other_lemmata = features[is_same_pos & is_different_lemma]
                if (
                    other_lemmata.empty
                ):  # we don't have feature data on any other lemmata
                    continue
                else:
                    num_lemmata = len(other_lemmata["lemma"].unique())
                    merged = pd.merge(
                        probs, other_lemmata, left_on=["word"], right_on=["word"]
                    )
                    merged["correct"] = (merged[cols] == row[cols]).all(axis=1)
                    incorrect_forms = merged[~merged["correct"]]
                    lemmata = incorrect_forms["lemma"]
                    grouped = merged[merged["lemma"].isin(lemmata)].groupby("lemma")
                    count = 0
                    for _, group in grouped:
                        try:
                            p_correct = group[group["correct"]]["p"].max()
                            try:
                                p_incorrect = group[~group["correct"]]["p"].max()
                                if p_incorrect >= p_correct:
                                    count += 1
                            except KeyError:
                                continue
                        except KeyError:
                            continue
                    example = {
                        "lg": lg,
                        "uid": uid,
                        "lemma": lemma,
                        "correct_form": correct_form,
                        "num_incorrect_lemmata": count,
                        "num_lemmata": num_lemmata,
                    }
                    results.append(example)
        except FileNotFoundError:
            continue
    results = pd.DataFrame(results)
    results["percentage"] = 100 * (
        results["num_incorrect_lemmata"] / results["num_lemmata"]
    )
    results.to_csv(f"data/bylemmata/{lg}.csv", index=False)