"""Paired contrasts per model, with a bootstrap over questions.

Replies to one question share its numbers and wording, so they are related. The
difference is computed per question first, and whole questions are resampled,
never single replies. Models are never averaged together.
"""
import numpy as np
import pandas as pd


def question_means(df, measure):
    """One row per model x question, one column per condition (mean over seeds and phrasings)."""
    return df.pivot_table(index=["model", "question_id"], columns="condition",
                          values=measure, aggfunc="mean")


def paired_diffs(means, a, b):
    """Per-question difference a minus b. Questions missing either cell are dropped."""
    if a not in means or b not in means:
        return pd.Series(dtype=float)
    return (means[a] - means[b]).dropna()


def bootstrap_ci(diffs, n_boot, ci, seed):
    values = np.asarray(diffs, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    tail = (1 - ci) / 2
    return values.mean(), np.quantile(boots, tail), np.quantile(boots, 1 - tail)


def list_contrasts(cfg):
    an = cfg["analysis"]
    contrasts = [{**an["primary"], "role": "primary"}, {**an["control"], "role": "control"}]
    for cond in cfg["cues"]:
        if cond != "neutral":
            contrasts.append({"name": f"{cond}_vs_neutral", "a": cond, "b": "neutral",
                              "measure": None, "role": "exploratory"})
    return contrasts


def role_of(contrast, measure):
    # Only the pre-registered measure of a pre-registered contrast keeps its role.
    return contrast["role"] if measure == contrast["measure"] else "exploratory"


def summary_row(model, contrast, measure, diffs, cfg):
    an = cfg["analysis"]
    mean, lo, hi = bootstrap_ci(diffs, an["n_boot"], an["ci"], cfg["seeds"]["bootstrap"])
    return {"model": model, "contrast": contrast["name"], "a": contrast["a"], "b": contrast["b"],
            "measure": measure, "role": role_of(contrast, measure), "n_questions": len(diffs),
            "mean_diff": mean, "ci_low": lo, "ci_high": hi}


def summarize(df, cfg):
    """One row per model x contrast x measure."""
    measures = cfg["analysis"]["measures"]
    means = {m: question_means(df, m) for m in measures}
    rows = []
    for model in sorted(df["model"].unique()):
        for contrast in list_contrasts(cfg):
            for measure in measures:
                diffs = paired_diffs(means[measure].loc[model], contrast["a"], contrast["b"])
                rows.append(summary_row(model, contrast, measure, diffs, cfg))
    return pd.DataFrame(rows)
