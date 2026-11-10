"""Paired contrasts per model, with a bootstrap over questions.

Replies to one question share its numbers and wording, so they are related. The
difference is computed per question first, and whole questions are resampled,
never single replies. Models are never averaged, ranked or singled out.
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


# ---------- ceiling or floor rule ----------

def key_measures(neutral_df, cfg):
    """Per model, the measure for the primary and control contrast, from neutral replies only."""
    if set(neutral_df["condition"]) != {"neutral"}:
        raise ValueError("the ceiling/floor rule is decided from neutral replies only")
    rule, default = cfg["analysis"]["ceiling_rule"], cfg["analysis"]["primary"]["measure"]
    rates = neutral_df.groupby("model")["gives_answer"].mean()
    return {model: rule["fallback"] if rate > rule["ceiling"] or rate < rule["floor"] else default
            for model, rate in rates.items()}


# ---------- summary ----------

def list_contrasts(cfg):
    an = cfg["analysis"]
    contrasts = [{**an["primary"], "role": "primary"}, {**an["control"], "role": "control"}]
    for cond in cfg["cues"]:
        if cond != "neutral":
            contrasts.append({"name": f"{cond}_vs_neutral", "a": cond, "b": "neutral",
                              "measure": None, "role": "exploratory"})
    return contrasts


def role_of(contrast, measure, key):
    # Only the key measure of the primary and control contrasts keeps that role.
    # key is the configured measure, or the fallback if the ceiling/floor rule applied.
    if contrast["role"] == "exploratory":
        return "exploratory"
    return contrast["role"] if measure == (key or contrast["measure"]) else "exploratory"


def summary_row(model, contrast, measure, diffs, key, cfg):
    an = cfg["analysis"]
    mean, lo, hi = bootstrap_ci(diffs, an["n_boot"], an["ci"], cfg["seeds"]["bootstrap"])
    return {"model": model, "contrast": contrast["name"], "a": contrast["a"], "b": contrast["b"],
            "measure": measure, "role": role_of(contrast, measure, key), "n_questions": len(diffs),
            "mean_diff": mean, "ci_low": lo, "ci_high": hi}


def summarize(df, cfg, keys=None, min_detectable=np.nan):
    """One row per model x contrast x measure, every contrast for every model.

    keys: model -> key measure from the ceiling/floor rule (default: the configured one).
    min_detectable: from the mock sensitivity runs, written on gives_answer primary rows.
    """
    measures, keys = cfg["analysis"]["measures"], keys or {}
    means = {m: question_means(df, m) for m in measures}
    rows = []
    for model in sorted(df["model"].unique()):
        for contrast in list_contrasts(cfg):
            for measure in measures:
                diffs = paired_diffs(means[measure].loc[model], contrast["a"], contrast["b"])
                rows.append(summary_row(model, contrast, measure, diffs, keys.get(model), cfg))
    out = pd.DataFrame(rows)
    # The sensitivity runs plant an effect on gives_answer only, so they speak to those rows only.
    on_primary = (out.role == "primary") & (out.measure == cfg["analysis"]["primary"]["measure"])
    out["min_detectable_diff"] = np.where(on_primary, min_detectable, np.nan)
    return out


# ---------- sensitivity (mock model) ----------

def excludes_zero(df, a, b, measure, cfg):
    """For a one-model frame: does the interval for a minus b exclude zero?"""
    an = cfg["analysis"]
    means = question_means(df, measure).droplevel("model")
    _, lo, hi = bootstrap_ci(paired_diffs(means, a, b), an["n_boot"], an["ci"], cfg["seeds"]["bootstrap"])
    return lo > 0 or hi < 0


def primary_excludes_zero(df, cfg):
    p = cfg["analysis"]["primary"]
    return excludes_zero(df, p["a"], p["b"], p["measure"], cfg)


def sensitivity(make_dataset, cfg):
    """Share of simulated runs whose primary interval excludes zero, per planted difference.

    make_dataset(diff, run) returns one scored, one-model frame (see mock.sensitivity_run).
    """
    sens = cfg["mock"]["sensitivity"]
    rows = []
    for diff in sens["diffs"]:
        hits = [primary_excludes_zero(make_dataset(diff, run), cfg) for run in range(sens["runs"])]
        rows.append({"diff": diff, "runs": sens["runs"], "share_excluding_zero": float(np.mean(hits))})
    return pd.DataFrame(rows)


def smallest_detectable(table, power):
    ok = table[table["share_excluding_zero"] >= power]
    return float(ok["diff"].min()) if len(ok) else np.nan
