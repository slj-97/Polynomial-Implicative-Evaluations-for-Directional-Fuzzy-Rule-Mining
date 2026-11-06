"""Verified descriptive tables and figures for the completed Q5 benchmark."""
from __future__ import annotations

import json
from pathlib import Path
import winreg

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import ROOT

OUT=ROOT/"experiments/outputs/revision_r1_q5"
DISPLAY={"exhaustive":"Exhaustive", "coverage_only":"Coverage only", "coverage_support":"Joint"}
MODES=list(DISPLAY)
COLORS=["#666666", "#286b9c", "#ba5b3b"]


def markdown(frame: pd.DataFrame) -> str:
    return "\n".join(["| "+" | ".join(frame.columns)+" |",
                       "| "+" | ".join(["---"]*len(frame.columns))+" |"]+
                      ["| "+" | ".join(map(str,r))+" |" for r in frame.itertuples(index=False,name=None)])


def main() -> None:
    manifest=json.loads((OUT/"manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"]=="complete" and manifest["search_runs"]==300
    runs=pd.read_csv(OUT/"measurements.csv")
    historical=pd.read_csv(ROOT/"experiments/outputs/raw_results/pruning_multivariate.csv")
    historical=historical[(historical.K==4)&historical.seed.isin(manifest["seeds"])]
    new=runs[(runs.variables==5)&(runs.delta==.375)]
    joined=new.merge(historical,on=["seed","mode","support_threshold"],suffixes=("_new","_old"))
    assert len(joined)==60
    for field in ["generated","scored","retained"]:
        assert np.array_equal(joined[field+"_new"],joined[field+"_old"])
    assert runs.retained_set_equal.all()
    assert (runs.elapsed_seconds>0).all() and (runs.peak_working_set_bytes>=runs.rss_before_bytes).all()
    assert runs.groupby(["variables","delta","support_threshold","mode"]).seed.nunique().eq(10).all()
    assert runs.groupby(["seed","variables"]).memberships_sha256.nunique().eq(1).all()
    coverage=runs[runs["mode"]=="coverage_only"]
    assert coverage.groupby(["seed","variables"])[["generated","scored"]].nunique().eq(1).all().all()
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
        cpu=winreg.QueryValueEx(key,"ProcessorNameString")[0].strip()
    (OUT/"validation.json").write_text(json.dumps(dict(historical_rows_checked=len(joined),
        historical_counts_exact=True,retained_set_agreement=True,search_runs=len(runs),
        memberships_identical_across_parameters_and_thresholds=True,coverage_candidate_counts_invariant=True,
        paired_comparisons=len(runs)//3,cpu_model=cpu,
        peak_memory_counter="Windows peak_wset snapshot immediately after search, before key-set construction/serialization"),indent=2),encoding="utf-8")
    scale=runs[runs.delta==.375]
    summary=[]
    for (variables,support),block in scale.groupby(["variables","support_threshold"]):
        row=dict(variables=variables,support_threshold=support)
        for mode in MODES:
            group=block[block["mode"]==mode]
            for metric in ["generated","scored","retained","elapsed_seconds","peak_mib","rss_before_mib",
                           "time_ratio_to_exhaustive","incremental_scored_reduction_vs_coverage"]:
                row[mode+"_"+metric]=float(group[metric].median())
        row.update(engine="OUR-MAIN",protocol=manifest["protocol"])
        summary.append(row)
    summary=pd.DataFrame(summary)
    summary.to_csv(OUT/"scale_display.csv",index=False)
    figs=OUT/"figures"
    figs.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,support in zip(axes,[.01,.05]):
        for mode,color in zip(MODES,COLORS):
            points=[]
            for v in [5,8,10]:
                a=scale.loc[(scale.variables==v)&(scale.support_threshold==support)&(scale["mode"]==mode),"elapsed_seconds"]
                points.append(a.quantile([.25,.5,.75]).to_numpy())
            points=np.asarray(points)
            ax.errorbar([5,8,10],points[:,1],yerr=np.array([points[:,1]-points[:,0],points[:,2]-points[:,1]]),
                        marker="o",capsize=3,label=DISPLAY[mode],color=color)
        ax.set_yscale("log")
        ax.set_xticks([5,8,10])
        ax.set_xlabel("Variables (three labels each)")
        ax.set_ylabel("Search time (seconds; logarithmic scale)")
        ax.set_title(f"Support threshold = {support:g}")
    axes[0].legend(frameon=False)
    fig.suptitle("K = 4, sigma = 0.75, delta = 0.375; median and IQR over 10 seeds")
    fig.tight_layout()
    for ext in ["png","pdf"]: fig.savefig(figs/f"figure-01-search-time.{ext}",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,support in zip(axes,[.01,.05]):
        for offset,(mode,color) in enumerate(zip(MODES,COLORS)):
            data=[scale.loc[(scale.variables==v)&(scale.support_threshold==support)&(scale["mode"]==mode),"peak_mib"].to_numpy() for v in [5,8,10]]
            for i,values in enumerate(data):
                ax.scatter(i+(offset-1)*.2+np.linspace(-.045,.045,len(values)),values,s=15,color=color,alpha=.55)
            ax.plot(np.arange(3)+(offset-1)*.2,[np.median(a) for a in data],marker="_",ls="",color=color,markersize=12,label=DISPLAY[mode])
        ax.set_xticks(range(3),[5,8,10])
        ax.set_xlabel("Variables")
        ax.set_title(f"Support threshold = {support:g}")
        ax.set_ylabel("Whole-worker peak working set (MiB)")
    axes[0].legend(frameon=False)
    fig.suptitle("OS lifetime peak through search; includes interpreter, preprocessing and returned rules")
    fig.tight_layout()
    for ext in ["png","pdf"]: fig.savefig(figs/f"figure-02-peak-memory.{ext}",dpi=180)
    plt.close(fig)
    display=summary[["variables","support_threshold"]+[m+"_"+q for q in ["scored","elapsed_seconds","peak_mib"] for m in MODES]].copy()
    for col in display.columns:
        if col not in ["variables","support_threshold"]: display[col]=display[col].map(lambda v:f"{v:.4f}")
    report="# Reviewer 1 Q5 computational extension\n\n"
    report+="100 paired seed/configuration comparisons, 300 sequential isolated searches. Ten fixed seeds; scale 5/8/10 variables at K=4; sigma=.75, delta=.375; support thresholds .01/.05. The 8-variable parameter analysis also uses delta=-.375 and0, reusing the measured positive setting.\n\n"
    report+=markdown(display)+"\n\n"
    report+="## Validation and measurement\n\nAll 100 comparisons have exactly equal retained key sets across the three search modes. Exhaustive counts match the combinatorial formula. All 60 new five-variable seed/mode/threshold rows reproduce historical generated/scored/retained counts. Search source unchanged. CPU: "+cpu+". Timing covers mine_rules and result sorting, excludes startup/preprocessing and serialization. Memory is the OS lifetime peak through return from search; pre-search RSS and peak are recorded. It includes process/import/preprocessing overhead, so it is not a search-only allocation peak. Modes run sequentially in fresh workers, with cyclic ordering and one BLAS thread.\n\n"
    report+="## Claim candidates\n\nCandidate reductions and paired elapsed-time ratios quantify this implementation on controlled membership data. Any speedup claim must give its measured settings and retained-set equality. Whole-worker memory measurements may be similar despite fewer candidates because input, interpreter and retained output remain resident. The experiment does not establish real-data scaling or a smaller combinatorial worst case. Sources: measurements.csv, summary.csv, validation.json.\n"
    (OUT/"analysis-report.md").write_text(report,encoding="utf-8")
    (OUT/"stats-appendix.md").write_text("""# Q5 descriptive statistics

Ten fixed seeds per configuration. Medians and 25th/75th empirical percentiles
use linear interpolation. Time ratios are computed per paired seed, then
summarized; a ratio of medians is not substituted for the median ratio.
Incremental support-score reduction is 1-C_J/C_C per seed. Parameter settings
are separate configurations, not pooled. The positive 8-variable setting is
shared between the scale and directional-parameter displays, not independently
remeasured. No significance tests or confidence intervals are claimed. Fresh
process peaks include worker startup, preprocessing and retained output, with
pre-search RSS/peak saved separately. Timing excludes those startup stages.
""",encoding="utf-8")
    (OUT/"figure-catalog.md").write_text("""# Figure catalog

1. figure-01-search-time: scale subset of measurements.csv; log-time curves
by three search modes; median/IQR across ten seeds. Shows empirical runtime
growth and whether joint pruning saves time beyond coverage-only pruning.
Error bars are descriptive quartiles, not confidence intervals.
2. figure-02-peak-memory: same scale subset; each worker's OS peak working
set with median marks. Shows full-process memory separately from runtime.
These peaks include startup/input/output overhead and must not be interpreted
as search-only allocations or inferred solely from candidate counts.
""",encoding="utf-8")
    print(summary.to_string(index=False))


if __name__=="__main__": main()
