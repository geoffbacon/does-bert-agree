"""Quick script to analyze the original method of evaluating agreement predictions."""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from filenames import PROBABILITIES_DIR, FEATURES_DIR, CLOZE_DIR

results = []
for lg in os.listdir(PROBABILITIES_DIR):
    print(lg)
    features_filename = os.path.join(FEATURES_DIR, f'{lg}.csv')
    features = pd.read_csv(features_filename)
    cloze_filename = os.path.join(CLOZE_DIR, f'{lg}.csv')
    cloze = pd.read_csv(cloze_filename)
    for _,row in tqdm(cloze.iterrows()):
        uid = row['uid']
        pos = row["pos"]
        probabilities_filename = os.path.join(PROBABILITIES_DIR, lg, f'{uid}.csv')
        try:  # we may have skipped this cloze example
            probs = pd.read_csv(probabilities_filename)
            lemma = row["lemma"]
            correct_form = row["correct_form"]
            p_correct_form = probs[probs["word"] == correct_form]["p"].max()
            if np.isnan(p_correct_form):  # the correct form didn't appear in the lexicon
                continue
            else:
                is_same_lemma = features["lemma"] == lemma
                is_same_pos = features["pos"] == pos
                is_incorrect_form = features["word"] != correct_form
                incorrect_forms = features[is_same_lemma & is_same_pos & is_incorrect_form]["word"]
                if incorrect_forms.empty:  # we don't have feature data on any incorrect forms
                    continue
                else:
                    probs_incorrect_forms = probs[probs["word"].isin(incorrect_forms)]
                    p_incorrect_form = probs_incorrect_forms["p"].max()
                    if np.isnan(p_incorrect_form):  # no incorrect forms appear in the lexicon
                        continue
                    else:
                        incorrect_form = probs_incorrect_forms[probs_incorrect_forms["p"] == p_incorrect_form]["word"].iloc[0]
                        example = {"lg": lg, "uid": uid, "lemma": lemma, "correct_form": correct_form, 
                                   "p_correct_form": p_correct_form, "incorrect_form": incorrect_form,
                                   "p_incorrect_form": p_incorrect_form}
                        results.append(example)
        except FileNotFoundError:
            continue
results = pd.DataFrame(results)
results["right"] = results["p_correct_form"] > results["p_incorrect_form"]
results["margin"] = results["p_correct_form"] - results["p_incorrect_form"]
results.to_csv("data/bymargin.csv", index=False)
