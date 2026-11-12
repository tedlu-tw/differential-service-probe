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


# ---------- pre-freeze checks ----------

def pilot_report(neutral_df, solve_df, cfg):
    """Checks a, b and d per model. Refuses anything but neutral replies: no contrasts here."""
    if set(neutral_df["condition"]) != {"neutral"}:
        raise ValueError("pilot mode computes no contrasts: pass neutral replies only")
    g = neutral_df.groupby("model")
    out = pd.DataFrame({"n_replies": g.size(), "gives_answer_rate": g["gives_answer"].mean(),
                        "mean_steps": g["steps"].mean(), "truncation_rate": g["truncated"].mean()})
    out["accuracy"] = solve_df.groupby("model")["solve_correct"].mean()
    out["key_measure"] = pd.Series(key_measures(neutral_df, cfg))
    out["truncation_ok"] = out["truncation_rate"] < cfg["pilot"]["truncation_limit"]
    return out.rename_axis("model").reset_index()


def check_report(df):
    """Check c per model x condition: accuracy over the yes/no items of readable replies."""
    g = df.groupby(["model", "condition"])
    out = pd.DataFrame({"n_prompts": g.size(), "n_unparsed": g["parsed"].apply(lambda s: int((s == 0).sum()))})
    readable = df[df["parsed"] == 1].groupby(["model", "condition"])
    out["accuracy"] = readable[["item1_correct", "item2_correct"]].mean().mean(axis=1)
    return out.reset_index()


def diagnostics(df):
    """Per model x condition: truncation rate, and the share of cells where every seed gave the same reply."""
    g = df.groupby(["model", "condition"])
    out = pd.DataFrame({"n_replies": g.size(), "truncation_rate": g["truncated"].mean()})
    cell = df.groupby(["model", "question_id", "condition", "phrasing"])["reply"].agg(["nunique", "size"])
    same = (cell["nunique"] == 1) & (cell["size"] > 1)
    out["identical_across_seeds"] = same.groupby(level=["model", "condition"]).mean()
    return out.reset_index()


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


# =====================================================================================
# v3 (docs/protocol_v3.md). New functions only; everything above is v2 and unchanged.
# A dialogue is the unit: one row per dialogue, built from the scored tutor-reply rows.
# Every contrast is computed within one system prompt, per model, paired per question.
# =====================================================================================

DIALOGUE_KEY = ["model", "question_id", "prompt", "condition", "phrasing", "seed"]
TURN1_MEASURES = ["gives_answer", "hint_level", "steps", "praises", "encourages", "words", "mentions_history"]


def dialogues_v3(rows):
    """One row per dialogue: answer_turn (NaN = none), gave_by_turn2, and the turn-1 measures."""
    df = pd.DataFrame(rows)
    first = df[df["turn"] == 1].set_index(DIALOGUE_KEY)
    g = df.groupby(DIALOGUE_KEY)
    if len(first) != g.ngroups:
        raise ValueError("every dialogue needs a turn-1 reply")
    out = first[TURN1_MEASURES + ["reply"]].rename(columns={"gives_answer": "gives_answer_turn1"})
    out = out.join(g.agg(n_turns=("turn", "max"), any_truncated=("truncated", "max")))
    out["answer_turn"] = df[df["gives_answer"] == 1].groupby(DIALOGUE_KEY)["turn"].min().reindex(out.index)
    out["gave_by_turn2"] = (out["answer_turn"] <= 2).astype(int)
    return out.reset_index()


# ---------- contrasts and summary ----------

def contrasts_v3(cfg):
    an = cfg["analysis"]
    return ([{**an["primary"], "role": "primary"}, {**an["control"], "role": "control"}]
            + [{**c, "measure": None, "role": "exploratory"} for c in an["exploratory"]])


def summary_row_v3(model, contrast, measure, diffs, key, cfg, min_detectable):
    row = {"prompt": contrast["prompt"], **summary_row(model, contrast, measure, diffs, key, cfg)}
    wide = measure in cfg["analysis"]["precision_measures"]
    row["precision"] = (row["ci_high"] - row["ci_low"]) / 2 if wide else np.nan
    # Each possible primary measure has its own detectable difference from the sensitivity runs.
    row["min_detectable_diff"] = min_detectable.get(measure, np.nan) if row["role"] == "primary" else np.nan
    return row


def summarize_v3(d, cfg, keys=None, min_detectable=None):
    """One row per model x contrast x measure, every contrast for every model, within one prompt each.

    keys: model -> key measure from the ceiling/floor rule.
    min_detectable: model -> measure -> smallest detectable size (min_detectable_by_model).
    """
    measures, keys, min_detectable = cfg["analysis"]["measures"], keys or {}, min_detectable or {}
    means = {(p, m): question_means(d[d["prompt"] == p], m) for p in d["prompt"].unique() for m in measures}
    rows = []
    for model in sorted(d["model"].unique()):
        for contrast in contrasts_v3(cfg):
            for measure in measures:
                table = means.get((contrast["prompt"], measure))
                has = table is not None and model in table.index.get_level_values("model")
                diffs = paired_diffs(table.loc[model], contrast["a"], contrast["b"]) if has else pd.Series(dtype=float)
                rows.append(summary_row_v3(model, contrast, measure, diffs, keys.get(model), cfg,
                                            min_detectable.get(model, {})))
    return pd.DataFrame(rows)


def replication_v3(d, cfg):
    """Turn-1 answer rate under the plain prompt, neutral cells, per model (a rate, not a contrast)."""
    an, r = cfg["analysis"], cfg["analysis"]["replication"]
    sub, rows = d[(d["prompt"] == r["prompt"]) & (d["condition"] == r["condition"])], []
    for model, g in sub.groupby("model"):
        per_question = g.groupby("question_id")[r["measure"]].mean()
        rate, lo, hi = bootstrap_ci(per_question, an["n_boot"], an["ci"], cfg["seeds"]["bootstrap"])
        rows.append({"model": model, "prompt": r["prompt"], "condition": r["condition"], "measure": r["measure"],
                     "n_questions": len(per_question), "rate": rate, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


# ---------- distributions ----------

def shares(d, column, levels):
    """Share of dialogues at each level, per model x prompt x condition. Every level gets a column."""
    counts = pd.crosstab([d["model"], d["prompt"], d["condition"]], d[column]).reindex(columns=levels, fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0).rename(columns=lambda c: f"{column}_{c}").reset_index()


def answer_turn_shares(d):
    turns = d["answer_turn"].map({1: "1", 2: "2", 3: "3"}).fillna("none")
    return shares(d.assign(answer_turn=turns), "answer_turn", ["1", "2", "3", "none"])


def hint_level_shares(d):
    return shares(d, "hint_level", [0, 1, 2, 3])


# ---------- ceiling or floor rule ----------

def key_measures_v3(neutral_d, cfg, hint_valid=True):
    """Per model, the measure for the primary and control contrasts, from the neutral pilot dialogues only."""
    an = cfg["analysis"]
    if set(zip(neutral_d["prompt"], neutral_d["condition"])) != {(an["primary"]["prompt"], "neutral")}:
        raise ValueError("the ceiling/floor rule is decided from neutral dialogues under the pedagogical prompt only")
    rule = an["ceiling_rule"]
    fallback = rule["fallback"] if hint_valid else rule["fallback_if_invalid"]
    rates = neutral_d.groupby("model")["gave_by_turn2"].mean()
    return {model: fallback if rate > rule["ceiling"] or rate < rule["floor"] else an["primary"]["measure"]
            for model, rate in rates.items()}


# ---------- pre-freeze reports ----------

def pilot_report_v3(rows, solve_df, cfg):
    """Step a per model. Refuses anything but neutral pilot dialogues: no contrasts here."""
    d, p, rule = dialogues_v3(rows), cfg["pilot"], cfg["analysis"]["ceiling_rule"]
    if set(zip(d["prompt"], d["condition"])) != {(p["prompt"], p["condition"])}:
        raise ValueError("pilot mode computes no contrasts: pass neutral pedagogical dialogues only")
    g = d.groupby("model")
    out = pd.DataFrame({"n_dialogues": g.size(), "gave_by_turn2": g["gave_by_turn2"].mean()})
    turns = answer_turn_shares(d).drop(columns=["prompt", "condition"]).set_index("model")
    out = out.join(turns)
    out["truncation_rate"] = pd.DataFrame(rows).groupby("model")["truncated"].mean()
    solve = solve_df.groupby("model")
    out["accuracy_with_working"] = solve["solve_correct"].mean()
    out["solve_unparsed_rate"] = 1 - solve["solve_parsed"].mean()
    out["ceiling_rule_applies"] = (out["gave_by_turn2"] > rule["ceiling"]) | (out["gave_by_turn2"] < rule["floor"])
    return out.rename_axis("model").reset_index()


def check_report_v3(df):
    """Step d per model x condition: accuracy over readable replies, the unparsed rate, and which rule read each."""
    out = check_report(df).set_index(["model", "condition"])
    out["unparsed_rate"] = out["n_unparsed"] / out["n_prompts"]
    rules = pd.crosstab([df["model"], df["condition"]], df["check_rule"]).add_prefix("rule_")
    return out.join(rules).reset_index()


def diagnostics_v3(rows):
    """Per model x prompt x condition: dialogues, replies, truncation, and identical turn-1 replies across seeds."""
    df, cols = pd.DataFrame(rows), ["model", "prompt", "condition"]
    g = df.groupby(cols)
    out = pd.DataFrame({"n_replies": g.size(), "truncation_rate": g["truncated"].mean()})
    first = df[df["turn"] == 1]
    out["n_dialogues"] = first.groupby(cols).size()
    cell = first.groupby(cols + ["question_id", "phrasing"])["reply"].agg(["nunique", "size"])
    out["identical_across_seeds"] = ((cell["nunique"] == 1) & (cell["size"] > 1)).groupby(level=cols).mean()
    return out.reset_index()


def cohen_kappa(a, b):
    """Unweighted Cohen's kappa between two label lists (step b)."""
    a, b = np.asarray(a), np.asarray(b)
    observed = np.mean(a == b)
    chance = sum(np.mean(a == label) * np.mean(b == label) for label in np.union1d(a, b))
    return np.nan if chance == 1 else float((observed - chance) / (1 - chance))


# ---------- sensitivity (mock) ----------

SENSITIVITY_MEASURES = ("gave_by_turn2", "hint_level", "steps")


def sensitivity_sizes(cfg, measure):
    s = cfg["mock"]["sensitivity"][measure]
    return s["diffs"] if measure == "gave_by_turn2" else s["shifts"]


def primary_excludes_zero_v3(d, cfg, measure):
    p = cfg["analysis"]["primary"]
    return excludes_zero(d[d["prompt"] == p["prompt"]], p["a"], p["b"], measure, cfg)


def sensitivity_plan(cfg, bases):
    """(base, measure, size) per sweep: gave_by_turn2 once on the default base, hint_level and steps per base."""
    plan = [("default", "gave_by_turn2", s) for s in sensitivity_sizes(cfg, "gave_by_turn2")]
    return plan + [(b, m, s) for b in bases for m in SENSITIVITY_MEASURES[1:] for s in sensitivity_sizes(cfg, m)]


def sensitivity_v3(make_dataset, cfg, bases, plantable):
    """Share of simulated runs whose primary interval excludes zero, per base, measure and planted size.

    make_dataset(base, measure, size, run) returns scored tutor-reply rows of one model (mock.sensitivity_rows_v3).
    plantable(base, measure, size) is False when the base cannot carry the size; that row has no runs and NaN.
    """
    runs, rows = cfg["mock"]["sensitivity"]["runs"], []
    for base, measure, size in sensitivity_plan(cfg, bases):
        ok = plantable(base, measure, size)
        hits = [primary_excludes_zero_v3(dialogues_v3(make_dataset(base, measure, size, run)), cfg, measure)
                for run in range(runs if ok else 0)]
        rows.append({"base": base, "measure": measure, "size": size, "plantable": ok, "runs": len(hits),
                     "share_excluding_zero": float(np.mean(hits)) if ok else np.nan})
    return pd.DataFrame(rows)


def smallest_detectable_v3(table, power):
    """(base, measure) -> smallest planted size detected in at least `power` of runs (NaN if none)."""
    return {key: smallest_detectable(t.rename(columns={"size": "diff"}), power)
            for key, t in table.groupby(["base", "measure"])}


def min_detectable_by_model(found, base_of):
    """Model -> measure -> size. gave_by_turn2 comes from the shared default sweep, the others from the model's base."""
    return {model: {m: found.get(("default" if m == "gave_by_turn2" else base, m), np.nan) for m in SENSITIVITY_MEASURES}
            for model, base in base_of.items()}
