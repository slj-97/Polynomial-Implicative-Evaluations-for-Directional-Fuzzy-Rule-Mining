"""Reanalyse frozen tabular exports for Reviewer 1 Q3; never rewrite inputs."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import ROOT
from experiments.src.fuzzification import fuzzify_our_main
from experiments.src.matched_rules import pair_states, retention_contrast, retention_flags
from experiments.src.operators import eval_poly
from experiments.src.reproducibility import load_yaml

SOURCE = ROOT / "experiments/outputs/tabular"
OUTPUT = ROOT / "experiments/outputs/revision_r1_q3"
STATES = ["neither", "forward_only", "reverse_only", "both"]
DISPLAY = {"iris": "Iris", "wdbc": "WDBC", "vehicle": "Vehicle",
           "abalone": "Abalone", "magic": "MAGIC", "online_news": "Online News"}
PRIMARY_SIGMAS = [0.5, 1.0]
PRIMARY_RATIO = 0.5
PRIMARY_THRESHOLD = (0.01, 0.01, 0.5)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_matrix(frame: pd.DataFrame, quantity: str) -> np.ndarray:
    return frame[[f"{quantity}_forward", f"{quantity}_reverse"]].to_numpy(dtype=float)


def changed_rules(zero: pd.DataFrame, treatment: pd.DataFrame,
                  before: np.ndarray, after: np.ndarray,
                  meta: dict, thresholds: tuple[float, float, float]) -> pd.DataFrame:
    mask = (before != after).ravel()
    if not mask.any():
        return pd.DataFrame()
    cov = score_matrix(zero, "coverage").ravel()
    s0, s1 = score_matrix(zero, "support").ravel(), score_matrix(treatment, "support").ravel()
    c0, c1 = score_matrix(zero, "confidence").ravel(), score_matrix(treatment, "confidence").ravel()
    a = zero.index.get_level_values("item_a").to_numpy()
    c = zero.index.get_level_values("item_c").to_numpy()
    effective = np.maximum(thresholds[1], thresholds[2] * cov)
    data = dict(
        antecedent=np.column_stack([a, c]).ravel()[mask],
        consequent=np.column_stack([c, a]).ravel()[mask],
        direction=np.tile(["forward", "reverse"], len(zero))[mask],
        coverage=cov[mask], support_zero=s0[mask], support_treatment=s1[mask],
        confidence_zero=c0[mask], confidence_treatment=c1[mask],
        support_change=(s1-s0)[mask], confidence_change=(c1-c0)[mask],
        effective_support_threshold=effective[mask],
        baseline_margin=(s0-effective)[mask], treatment_margin=(s1-effective)[mask],
        selected_zero=before.ravel()[mask], selected_treatment=after.ravel()[mask],
        state_zero=np.repeat(pair_states(before), 2)[mask],
        state_treatment=np.repeat(pair_states(after), 2)[mask],
    )
    result = pd.DataFrame(data)
    for key, value in meta.items():
        result[key] = value
    return result


def markdown(frame: pd.DataFrame) -> str:
    rows = [[str(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(frame.columns) + " |",
                      "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
                     + ["| " + " | ".join(row) + " |" for row in rows])


def export_figures(matched: pd.DataFrame, datasets: list[str]) -> None:
    primary = matched[(matched.label_count == 3) & (matched.relative_delta == PRIMARY_RATIO)
                      & (matched.tau_support == .01) & (matched.tau_confidence == .5)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    for row, sigma in enumerate(PRIMARY_SIGMAS):
        block = primary[primary.sigma == sigma].set_index("dataset").loc[datasets]
        for col, sign in enumerate(["minus", "plus"]):
            ax = axes[row, col]
            x = np.arange(len(datasets))
            ax.bar(x-.18, block[f"zero_to_{sign}_added"], .36, label="Added", color="#286b9c")
            ax.bar(x+.18, -block[f"zero_to_{sign}_removed"], .36, label="Removed", color="#ba5b3b")
            ax.axhline(0, color="black", linewidth=.6)
            ax.set_title(f"sigma = {sigma:g}, delta = {'-' if sign == 'minus' else '+'}{sigma/2:g}")
            ax.set_xticks(x, [DISPLAY[d] for d in datasets], rotation=25, ha="right")
            ax.set_ylabel("Directed rules added / removed")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Matched retention changes relative to delta = 0 (three labels)")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUTPUT / "figures" / f"figure-01-matched-retention.{ext}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    columns = [(s, c) for s in [.01, .025, .05] for c in [.3, .5, .7]]
    part = matched[(matched.label_count == 3) & (matched.sigma == 1)
                   & (matched.relative_delta == PRIMARY_RATIO)]
    vmax = max(.01, 1 - part[["zero_to_minus_jaccard", "zero_to_plus_jaccard"]].min().min())
    for ax, sign in zip(axes, ["minus", "plus"]):
        matrix = 1 - part.pivot(index="dataset", columns=["tau_support", "tau_confidence"],
                                values=f"zero_to_{sign}_jaccard").reindex(index=datasets, columns=columns)
        im = ax.imshow(matrix.to_numpy(), aspect="auto", vmin=0, vmax=vmax, cmap="cividis")
        for i, j in itertools.product(range(len(datasets)), range(len(columns))):
            v = matrix.iloc[i, j]
            ax.text(j, i, f"{100*v:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if v < .45*vmax else "black")
        ax.set_title(f"sigma = 1, delta = {'-' if sign == 'minus' else '+'}0.5")
        ax.set_xticks(range(9), [f"{s:g}/{c:g}" for s, c in columns], rotation=60, ha="right")
        ax.set_yticks(range(6), [DISPLAY[d] for d in datasets])
        ax.set_xlabel("Support / confidence threshold")
    fig.subplots_adjust(bottom=.25, right=.84, wspace=.12)
    color_ax = fig.add_axes([.88, .25, .016, .63])
    fig.colorbar(im, cax=color_ax, label="1 - Jaccard relative to delta = 0")
    for ext in ["png", "pdf"]:
        fig.savefig(OUTPUT / "figures" / f"figure-02-threshold-sensitivity.{ext}", dpi=180)
    plt.close(fig)


def select_examples(primary_scores: pd.DataFrame) -> pd.DataFrame:
    selected, used = [], set()
    for category in ["orientation_flip", "addition", "removal", "unchanged"]:
        candidates = primary_scores[primary_scores[category]].copy()
        if candidates.empty:
            continue
        median = candidates.support_perturbation_abs.median()
        candidates["distance_to_category_median"] = abs(candidates.support_perturbation_abs-median)
        candidates = candidates.sort_values(["distance_to_category_median", "dataset", "sigma", "item_a", "item_c"])
        for _, row in candidates.iterrows():
            key = (row.dataset, row.sigma, row.item_a, row.item_c)
            if key not in used:
                used.add(key)
                selected.append((category, row))
                break
    records = []
    for number, (category, row) in enumerate(selected, 1):
        for setting, delta in [("zero", 0), ("minus", -row.sigma/2), ("plus", row.sigma/2)]:
            record = dict(example=number, category=category, dataset=row.dataset, label_count=3,
                          sigma=row.sigma, delta=delta, item_a=row.item_a, item_c=row.item_c,
                          gamma=row.gamma, coverage_forward=row.coverage_forward,
                          coverage_reverse=row.coverage_reverse,
                          tau_coverage=.01, tau_support=.01, tau_confidence=.5)
            for side in ["forward", "reverse"]:
                for quantity in ["support", "confidence", "selected"]:
                    record[f"{quantity}_{side}"] = row[f"{quantity}_{side}_{setting}"]
                record[f"baseline_margin_{side}"] = row[f"support_{side}_zero"] - max(.01, .5*row[f"coverage_{side}"])
                record[f"treatment_margin_{side}"] = record[f"support_{side}"] - max(.01, .5*row[f"coverage_{side}"])
            records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "figures").mkdir(exist_ok=True)
    config = load_yaml(ROOT / "experiments/configs/tabular.yaml")
    common = load_yaml(ROOT / "experiments/configs/common.yaml")
    provenance = json.loads((ROOT / "data/raw/tabular_firm/manifest.json").read_text())
    original = json.loads((SOURCE / "manifest.json").read_text())
    assert original["status"] == "complete" and original["smoke"] is False
    assert sha256(ROOT / "experiments/configs/tabular.yaml") == original["config_sha256"]
    assert sha256(ROOT / "experiments/configs/common.yaml") == original["common_sha256"]
    thresholds = list(itertools.product([float(common["thresholds"]["coverage"])],
                                       common["thresholds"]["support"], common["thresholds"]["confidence"]))
    inputs = {str(path): sha256(path) for path in [SOURCE / "manifest.json",
              ROOT / "experiments/configs/tabular.yaml", ROOT / "experiments/configs/common.yaml"]}
    counts, contrasts, transitions, changes, primary_scores = [], [], [], [], []
    flip_records = []
    errors, direct_errors = [], []
    margin_predicate_disagreements = 0
    pair_inventory = []
    for dataset in config["datasets"]:
        source = next(x for x in provenance if x["dataset"] == dataset)
        raw_path = ROOT / source["path"]
        assert sha256(raw_path) == source["sha256"]
        inputs[str(raw_path)] = source["sha256"]
        raw = pd.read_csv(raw_path)
        raw.columns = raw.columns.str.strip()
        numeric = raw.drop(columns=[config["targets"][dataset]]).select_dtypes(include=[np.number])
        split_path = SOURCE / f"{dataset}_split.npz"
        inputs[str(split_path)] = sha256(split_path)
        split = np.load(split_path)
        permutation = np.random.default_rng(config["seed"]).permutation(len(raw))
        cut = int(len(raw)*config["training_fraction"])
        np.testing.assert_array_equal(split["training"], permutation[:cut])
        np.testing.assert_array_equal(split["test"], permutation[cut:])
        for labels in config["label_counts"]:
            path = SOURCE / f"{dataset}_{labels}_pairs.parquet"
            inputs[str(path)] = sha256(path)
            all_results = pd.read_parquet(path)
            assert all_results.engine.eq("OUR-MAIN").all()
            assert all_results.protocol.eq("static-fixed-split").all()
            exported = all_results[all_results.evaluation.eq("proposed")].copy()
            assert not exported.duplicated(["sigma", "delta", "item_a", "item_c"]).any()
            _, memberships = fuzzify_our_main(numeric.iloc[split["training"]], numeric.iloc[split["test"]], labels=labels)
            universe = sorted((a, c) for a, c in itertools.combinations(memberships, 2)
                              if a.rsplit("::", 1)[0] != c.rsplit("::", 1)[0])
            expected = pd.MultiIndex.from_tuples(universe, names=["item_a", "item_c"])
            assert len(exported) == len(universe)*20
            pair_inventory.append(dict(dataset=dataset, label_count=labels, unordered_pairs=len(universe),
                                       directed_candidates=2*len(universe), evaluation_rows=len(split["test"])))
            for sigma in common["polynomial_grid"]["sigma"]:
                groups = {}
                for ratio in common["polynomial_grid"]["relative_delta"]:
                    delta = float(sigma*ratio)
                    group = exported[exported.sigma.eq(sigma) & exported.delta.eq(delta)].set_index(["item_a", "item_c"]).sort_index()
                    assert group.index.equals(expected)
                    groups[ratio] = group
                    errors.extend((group.support_forward-group.support_reverse-delta*group.gamma).abs())
                zero = groups[0]
                for ratio, group in groups.items():
                    delta = sigma*ratio
                    np.testing.assert_allclose(group.gamma, zero.gamma, rtol=0, atol=1e-15)
                    np.testing.assert_allclose(score_matrix(group, "coverage"), score_matrix(zero, "coverage"), rtol=0, atol=0)
                    expected_change = np.column_stack([delta*zero.gamma/2, -delta*zero.gamma/2])
                    np.testing.assert_allclose(score_matrix(group, "support")-score_matrix(zero, "support"), expected_change, rtol=0, atol=1e-12)
                    for i in np.random.default_rng(43).choice(len(group), min(3, len(group)), replace=False):
                        row = group.iloc[i]
                        a, c = memberships[row.name[0]], memberships[row.name[1]]
                        sf, sr = eval_poly(a, c, sigma, delta).mean(), eval_poly(c, a, sigma, delta).mean()
                        direct_errors.extend([abs(sf-row.support_forward), abs(sr-row.support_reverse),
                                              abs(a.mean()-row.coverage_forward), abs(c.mean()-row.coverage_reverse)])
                for tau in thresholds:
                    meta = dict(engine="OUR-MAIN", protocol="static-fixed-split-matched-retention", dataset=dataset,
                                label_count=labels, sigma=sigma, tau_coverage=tau[0], tau_support=tau[1], tau_confidence=tau[2])
                    flags = {ratio: retention_flags(group, *tau) for ratio, group in groups.items()}
                    for ratio, group in groups.items():
                        states = pair_states(flags[ratio])
                        suffix = f"cov{tau[0]:g}_supp{tau[1]:g}_conf{tau[2]:g}".replace(".", "p")
                        np.testing.assert_array_equal((states == 1) | (states == 2), group[f"SA_{suffix}"].to_numpy().astype(bool))
                        cov = score_matrix(group, "coverage")
                        effective = np.maximum(tau[1], tau[2]*cov)
                        effective_flags = (cov > 0) & (cov >= tau[0]) & (score_matrix(group, "support") >= effective)
                        margin_predicate_disagreements += int((effective_flags != flags[ratio]).sum())
                        counts.append(dict(meta, delta=sigma*ratio, relative_delta=ratio,
                                           retained_count=int(flags[ratio].sum()),
                                           **{STATES[i]: int((states == i).sum()) for i in range(4)}))
                    for ratio in [.5, 1.0]:
                        item = dict(meta, magnitude=sigma*ratio, relative_delta=ratio,
                                    n_zero=int(flags[0].sum()), n_minus=int(flags[-ratio].sum()), n_plus=int(flags[ratio].sum()),
                                    unordered_pairs=len(zero))
                        for name, left, right in [("zero_to_minus", 0, -ratio), ("zero_to_plus", 0, ratio), ("minus_to_plus", -ratio, ratio)]:
                            metrics, matrix = retention_contrast(flags[left], flags[right])
                            item.update({f"{name}_{key}": val for key, val in metrics.items()})
                            for i, j in itertools.product(range(4), repeat=2):
                                transitions.append(dict(meta, magnitude=sigma*ratio, comparison=name,
                                                        reference_state=STATES[i], treatment_state=STATES[j], count=int(matrix[i, j])))
                        perturbation = np.abs(sigma*ratio*zero.gamma.to_numpy()/2)
                        item.update(median_abs_support_gap=float(np.median(2*perturbation)),
                                    median_abs_support_perturbation=float(np.median(perturbation)),
                                    p95_abs_support_perturbation=float(np.quantile(perturbation, .95)),
                                    max_abs_support_perturbation=float(perturbation.max()))
                        coverages = score_matrix(zero, "coverage")
                        eligible = (coverages > 0) & (coverages >= tau[0])
                        conf_changes = np.divide(perturbation[:, None], coverages, out=np.zeros_like(coverages), where=coverages > 0)[eligible]
                        item.update(median_abs_confidence_perturbation=float(np.median(conf_changes)),
                                    p95_abs_confidence_perturbation=float(np.quantile(conf_changes, .95)),
                                    max_abs_confidence_perturbation=float(conf_changes.max()))
                        contrasts.append(item)
                        for sign in [-1, 1]:
                            change = changed_rules(zero, groups[sign*ratio], flags[0], flags[sign*ratio],
                                                   dict(meta, delta=sign*sigma*ratio, relative_delta=sign*ratio), tau)
                            if not change.empty:
                                changes.append(change)
                        flip_mask = ((pair_states(flags[-ratio]) == 1) & (pair_states(flags[ratio]) == 2)) | ((pair_states(flags[-ratio]) == 2) & (pair_states(flags[ratio]) == 1))
                        for i in np.flatnonzero(flip_mask):
                            event = len(flip_records)//3 + 1
                            for setting, r in [("zero", 0), ("minus", -ratio), ("plus", ratio)]:
                                z = groups[r].iloc[i]
                                record = dict(meta, flip_event=event, setting=setting,
                                              delta=sigma*r, magnitude=sigma*ratio,
                                              item_a=z.name[0], item_c=z.name[1], gamma=z.gamma)
                                for side, col in [("forward", 0), ("reverse", 1)]:
                                    for quantity in ["coverage", "support", "confidence"]:
                                        record[f"{quantity}_{side}"] = z[f"{quantity}_{side}"]
                                    record[f"selected_{side}"] = flags[r][i, col]
                                    effective = max(tau[1], tau[2]*z[f"coverage_{side}"])
                                    record[f"baseline_margin_{side}"] = zero.iloc[i][f"support_{side}"]-effective
                                    record[f"treatment_margin_{side}"] = z[f"support_{side}"]-effective
                                flip_records.append(record)
                        if labels == 3 and sigma in PRIMARY_SIGMAS and ratio == PRIMARY_RATIO and tau == PRIMARY_THRESHOLD:
                            p = zero[["coverage_forward", "coverage_reverse", "gamma"]].reset_index()
                            p["dataset"], p["sigma"], p["support_perturbation_abs"] = dataset, sigma, perturbation
                            for name, r in [("zero", 0), ("minus", -ratio), ("plus", ratio)]:
                                for side, col in [("forward", 0), ("reverse", 1)]:
                                    for quantity in ["support", "confidence"]:
                                        p[f"{quantity}_{side}_{name}"] = groups[r][f"{quantity}_{side}"].to_numpy()
                                    p[f"selected_{side}_{name}"] = flags[r][:, col]
                            p["addition"] = ((~flags[0]) & (flags[-ratio] | flags[ratio])).any(axis=1)
                            p["removal"] = (flags[0] & ~(flags[-ratio] & flags[ratio])).any(axis=1)
                            p["orientation_flip"] = ((pair_states(flags[-ratio]) == 1) & (pair_states(flags[ratio]) == 2)) | ((pair_states(flags[-ratio]) == 2) & (pair_states(flags[ratio]) == 1))
                            p["unchanged"] = (flags[0] == flags[-ratio]).all(axis=1) & (flags[0] == flags[ratio]).all(axis=1) & flags[0].any(axis=1) & (perturbation > 1e-12)
                            primary_scores.append(p)
            print(f"Verified and analysed {dataset}, labels={labels}, pairs={len(universe)}", flush=True)

    matched = pd.DataFrame(contrasts)
    pd.DataFrame(counts).to_csv(OUTPUT / "retention_counts.csv", index=False)
    matched.to_csv(OUTPUT / "matched_contrasts.csv", index=False)
    pd.DataFrame(transitions).to_csv(OUTPUT / "state_transitions.csv", index=False)
    changed = pd.concat(changes, ignore_index=True)
    assert (changed.selected_zero != changed.selected_treatment).all()
    changed.to_parquet(OUTPUT / "changed_rules.parquet", index=False)
    scores = pd.concat(primary_scores, ignore_index=True)
    scores.to_parquet(OUTPUT / "primary_pair_scores.parquet", index=False)
    examples = select_examples(scores)
    flip_frame = pd.DataFrame(flip_records)
    flip_frame.to_csv(OUTPUT / "orientation_flip_rules.csv", index=False)
    if not flip_frame.empty:
        candidates = flip_frame[flip_frame.setting.eq("zero")].copy()
        candidates["perturbation"] = abs(candidates.magnitude*candidates.gamma/2)
        candidates["median_distance"] = abs(candidates.perturbation-candidates.perturbation.median())
        candidate = candidates.sort_values(["median_distance", "dataset", "label_count", "sigma", "magnitude", "tau_support", "tau_confidence", "item_a", "item_c"]).iloc[0]
        flip_example = flip_frame[flip_frame.flip_event.eq(candidate.flip_event)].copy()
        flip_example["category"] = "orientation_flip"
        flip_example["example"] = int(examples.example.max()) + 1
        examples = pd.concat([examples, flip_example], ignore_index=True)
    examples.to_csv(OUTPUT / "representative_rules.csv", index=False)
    # Directly verify all selected illustrative rules, including near-threshold cases.
    memberships_cache = {}
    for row in examples.itertuples():
        key = (row.dataset, row.label_count)
        if key not in memberships_cache:
            frame = pd.read_csv(ROOT / f"data/raw/tabular_firm/{row.dataset}/{row.dataset}.csv")
            frame.columns = frame.columns.str.strip()
            numeric = frame.drop(columns=[config["targets"][row.dataset]]).select_dtypes(include=[np.number])
            split = np.load(SOURCE / f"{row.dataset}_split.npz")
            _, memberships_cache[key] = fuzzify_our_main(numeric.iloc[split["training"]], numeric.iloc[split["test"]], labels=int(row.label_count))
        a, c = memberships_cache[key][row.item_a], memberships_cache[key][row.item_c]
        sf, sr = eval_poly(a, c, row.sigma, row.delta).mean(), eval_poly(c, a, row.sigma, row.delta).mean()
        direct_errors.extend([abs(sf-row.support_forward), abs(sr-row.support_reverse),
                              abs(sf/a.mean()-row.confidence_forward), abs(sr/c.mean()-row.confidence_reverse),
                              abs((a*c*(1-a)*(1-c)*(a-c)).mean()-row.gamma)])
    primary = matched[(matched.label_count == 3) & matched.sigma.isin(PRIMARY_SIGMAS)
                      & (matched.relative_delta == PRIMARY_RATIO) & (matched.tau_support == .01) & (matched.tau_confidence == .5)].copy()
    primary.to_csv(OUTPUT / "primary_summary.csv", index=False)
    pd.DataFrame(pair_inventory).to_csv(OUTPUT / "pair_inventory.csv", index=False)
    assert len(matched) == 6*2*4*2*9
    assert max(errors) < 1e-12 and max(direct_errors) < 1e-12
    assert margin_predicate_disagreements == 0
    for path, digest in inputs.items():
        assert sha256(Path(path)) == digest, f"Input changed: {path}"
    manifest = dict(status="complete", seed=config["seed"], independent_splits=1,
                    input_sha256={str(Path(k).relative_to(ROOT)): v for k, v in inputs.items()},
                    comparison_rows=len(matched), changed_rule_records=len(changed),
                    scalar_direct_checks=len(direct_errors), max_direct_error=max(direct_errors),
                    max_identity_error=max(errors), margin_predicate_disagreements=margin_predicate_disagreements,
                    primary_sigmas=PRIMARY_SIGMAS, primary_relative_delta=PRIMARY_RATIO,
                    primary_thresholds=PRIMARY_THRESHOLD,
                    script_sha256=sha256(Path(__file__)),
                    analysis_module_sha256=sha256(ROOT / "experiments/src/matched_rules.py"),
                    primary_selection="Declared constants; no selection of parameters or thresholds from their results.",
                    examples_selection="For addition/removal/unchanged categories, choose the pair nearest the category median absolute support perturbation in the primary cells. For an exclusive-direction flip, choose the event nearest its category median in the full grid. Ties are lexical; disclose the distinct parameter/threshold scope of the flip example.",
                    scope="Descriptive matched retention contrasts on the original fixed splits; no inferential tests or repeated-split stability claims.")
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    export_figures(matched, config["datasets"])
    write_reports(primary, matched, changed, examples, manifest)


def write_reports(primary: pd.DataFrame, matched: pd.DataFrame, changed: pd.DataFrame,
                  examples: pd.DataFrame, manifest: dict) -> None:
    columns = ["dataset", "sigma", "n_zero", "n_minus", "n_plus", "zero_to_minus_added", "zero_to_minus_removed", "zero_to_plus_added", "zero_to_plus_removed", "zero_to_minus_jaccard", "zero_to_plus_jaccard", "minus_to_plus_orientation_flips"]
    display = primary[columns].copy()
    display["dataset"] = display.dataset.map(DISPLAY)
    for col in ["zero_to_minus_jaccard", "zero_to_plus_jaccard"]:
        display[col] = display[col].map(lambda v: f"{v:.6f}")
    flips = int(matched.minus_to_plus_orientation_flips.sum())
    minimum = min(matched.zero_to_minus_jaccard.min(), matched.zero_to_plus_jaccard.min())
    margin_max = float(changed.baseline_margin.abs().max())
    confidence_governed = float((changed.effective_support_threshold > changed.tau_support).mean())
    flip_frame = pd.read_csv(OUTPUT / "orientation_flip_rules.csv")
    flip_events = flip_frame[flip_frame.setting.eq("zero")]
    unique_flipped_pairs = len(flip_events.drop_duplicates(["dataset", "label_count", "item_a", "item_c"]))
    primary_minimum = min(primary.zero_to_minus_jaccard.min(), primary.zero_to_plus_jaccard.min())
    report = f"""# Reviewer 1 Q3: measured matched-rule effects

## Protocol and comparison unit

Original six processed FIRM datasets; saved 70/30 splits, seed 20260903;
training-only preprocessing; both three- and five-label partitions;
all four sigma values, both matched magnitudes d=sigma/2 and d=sigma,
and all nine original threshold combinations. Each contrast uses the same
unordered singleton item pairs and both directed rules. No new data split,
threshold tuning, candidate subsampling or parameter pooling was used.

Primary display: three labels, sigma in {{0.5, 1}}, delta in {{-sigma/2,0,+sigma/2}},
coverage/support/confidence thresholds (0.01,0.01,0.5). These display constants
were declared before the new full-grid analysis; the first inspection of
the uploaded artifacts used the same constants. The full-grid CSV preserves
every cell. Counts concern directed rules. Jaccard treats two empty sets as 1.

## Primary numeric table

{markdown(display)}

## Full-grid checks

- Matched comparison cells: {len(matched)} (6 datasets x 2 partitions x 4 sigma x 2 magnitudes x 9 thresholds).
- Exclusive forward-only <-> reverse-only changes between matched signs, summed over these cells: {flips}. Cells overlap, so this is not a unique-rule count.
- These flips occur in {len(flip_events.drop_duplicates(['dataset','label_count','sigma','magnitude','tau_support','tau_confidence']))} cells and involve {unique_flipped_pairs} distinct item pairs (dataset and partition identified). See `orientation_flip_rules.csv`.
- The primary display has zero exclusive-direction flips and minimum Jaccard {primary_minimum:.6f}.
- Minimum Jaccard to the matched symmetric member over the full grid: {minimum:.6f}.
- The minimum of 0.5 occurs for WDBC with three labels, sigma=1, tau_support=0.05 and tau_confidence=0.7: the baseline retains one rule and negative delta retains two. Low overlap here reflects a one-rule difference in a very small set.
- Maximum absolute baseline effective-threshold margin among changed directed-rule records: {margin_max:.9g}.
- Fraction of changed directed-rule records where confidence imposes a stricter support requirement than the nominal support threshold: {confidence_governed:.6f}. Records can recur across parameters and thresholds.

## Interpretation supported by the measurements

The antisymmetric component changes some retained rules even when the symmetric
component is fixed. The primary display has high overlap; stricter thresholds
can leave very small rule sets with much lower Jaccard values. A sign change in the analytical
support difference must be distinguished from an exclusive retained-direction
flip; use the recorded transition counts for the latter. Counts can coincide
despite different members. None of these quantities establishes superior rule
quality or stability across data splits.

For the same pair, S_forward(delta)=S0+delta*Gamma/2 and
S_reverse(delta)=S0-delta*Gamma/2. The per-rule change relative to delta=0 is
therefore half of the signed forward-reverse gap. Confidence changes by the
support change divided by that direction's fixed antecedent coverage.
For coverage-eligible rules, define t_eff=max(tau_support,tau_confidence*coverage).
The actual crossing depends on S0-t_eff, not only on the size of the gap
relative to tau_support. `changed_rules.parquet` records both margins.

## Representative rules

`representative_rules.csv` contains full scores and decisions for {examples.example.nunique()} examples,
each under zero and matched negative/positive delta. Addition, removal and
unchanged-retention examples use the primary display cells. The exclusive-direction
flip example uses its disclosed full-grid cell. In each category select the pair
or event nearest the median absolute support perturbation; break ties lexically
and avoid duplicate primary pairs.
This is a declared descriptive example-selection rule, not a utility criterion.

## Validation and limits

Input hashes and saved split indices checked. All original threshold-specific
pair disagreement flags reconstructed exactly. {manifest['scalar_direct_checks']} direct
record-level scalar checks had maximum discrepancy {manifest['max_direct_error']:.3g}.
Every stored per-rule support perturbation matched delta*Gamma/2 within 1e-12.
The original confidence predicate and effective-support-threshold predicate
gave identical decisions. Historical input artifacts were not modified.

There is one original split per dataset; item pairs share observations.
Results are conditional descriptive counts, not independent replications.
Repeated-split stability belongs to Q4. The manuscript and response letter
have not been edited by this analysis script.

## Claim candidates

- Keep: fixed-sigma antisymmetric perturbations can add or remove threshold-near rules. Evidence: `primary_summary.csv`, `changed_rules.parquet`.
- Keep: report the measured degree of overlap and actual state changes. Evidence: `matched_contrasts.csv`, `state_transitions.csv`.
- Reject: opposite signs necessarily reverse the retained orientation; a signed support identity alone cannot establish this.
- Reject: larger support gaps establish better rules, a learned direction, or cross-split stability.
"""
    (OUTPUT / "analysis-report.md").write_text(report, encoding="utf-8")
    (OUTPUT / "stats-appendix.md").write_text("""# Descriptive comparison definitions

The observational unit is a fixed unordered pair of singleton fuzzy items,
with two directed candidates. Directed-rule counts use a common universe.
Pair states: neither, forward only, reverse only, both. An orientation flip
counts only forward-only <-> reverse-only transitions. State-transition CSVs
include all sixteen cells for each of zero->minus, zero->plus, minus->plus.

Jaccard=intersection/union; Jaccard(empty,empty)=1 with both counts reported.
Added=treatment\\reference; removed=reference\\treatment. Changes in total
count alone can conceal different membership. Support perturbation summaries
use all unordered pairs in a cell; confidence perturbation summaries use
coverage-eligible directed rules. The p95 values are descriptive distribution
quantiles, not confidence limits. Rule-change records recur across cells.

No significance tests or confidence intervals: one original data split per
dataset, no independent repeated estimates, and label-pair dependence.
No causal, predictive-accuracy, superiority or resampling-stability claim.
Full parameter, partition and threshold cells are retained separately.
""", encoding="utf-8")
    (OUTPUT / "figure-catalog.md").write_text("""# Descriptive figure catalog

## figure-01-matched-retention
- Purpose: show directed rules added and removed relative to delta=0 at fixed sigma.
- Source: primary_summary.csv; six datasets, three labels, sigma=0.5/1, d=sigma/2; thresholds (0.01,0.01,0.5).
- Meaning: bars are exact counts on the original fixed split, with removal counts plotted below zero; no error bars or independent repeats.
- Interpretation: small net changes can hide additions and removals occurring together. Counts reflect candidate-space sizes as well as sensitivity.
- Check: identify both signs and each sigma separately; do not interpret a larger count as better rules.

## figure-02-threshold-sensitivity
- Purpose: show how retained-set overlap changes across the nine prespecified thresholds at fixed sigma=1 and d=0.5.
- Source: matched_contrasts.csv, three-label cells; each sign has its own panel.
- Meaning: color is 1-Jaccard; numeric labels are percentages. Both panels share a scale covering their full observed range; no sampling uncertainty encoded.
- Interpretation: the practical effect of the same evaluation perturbation depends on the retention thresholds.
- Check: these are descriptive fixed-split set contrasts, not significance or quality rankings. All other parameter and partition cells remain in the CSV.
""", encoding="utf-8")
    print(f"Completed {len(matched)} matched cells; full-grid exclusive flips={flips}; min J={minimum:.6f}", flush=True)


if __name__ == "__main__":
    main()
