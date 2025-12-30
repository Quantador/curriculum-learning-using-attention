# analyze_ablation_results.py
# Analyze and plot results from eval_curriculum_ablation.py

import os
import json
from pathlib import Path
import matplotlib.pyplot as plt

RESULTS_DIR = "results_ablation"
OUT_DIR = "results_ablation/plots"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

def load_all_metrics(results_dir):
    metrics = {}
    for fname in os.listdir(results_dir):
        if fname.endswith("_metrics.json"):
            exp_name = fname.replace("_metrics.json", "")
            with open(os.path.join(results_dir, fname), "r") as f:
                data = json.load(f)
            metrics[exp_name] = data
    return metrics

def plot_metric_over_time(metrics_dict, metric_name, out_path, experiments=None, ylabel=None, use_step=True):
    """
    metrics_dict: dict(exp_name -> metrics_json)
    metric_name: key inside metrics_json to plot
    experiments: list of experiment names to include (if None: all)
    """
    plt.figure(figsize=(8, 5))
    any_plotted = False

    for exp_name, hist in metrics_dict.items():
        if experiments is not None and exp_name not in experiments:
            continue
        if metric_name not in hist:
            continue

        ys = hist[metric_name]
        if not ys:
            continue

        # Try to use 'step' if lengths match, otherwise fall back to simple index
        if use_step and "step" in hist and len(hist["step"]) == len(ys):
            xs = hist["step"]
        else:
            xs = list(range(len(ys)))

        label = exp_name
        plt.plot(xs, ys, label=label)
        any_plotted = True

    if not any_plotted:
        print(f"[WARN] No data to plot for metric '{metric_name}'. Nothing saved.")
        plt.close()
        return

    plt.xlabel("Step" if use_step else "Index")
    plt.ylabel(ylabel or metric_name)
    plt.title(metric_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")


def summarize_experiments(metrics_dict):
    summary = {}
    for exp_name, hist in metrics_dict.items():
        s = {}
        # Final and best val_ppl
        val_ppl = hist.get("val_ppl", [])
        if val_ppl:
            s["final_val_ppl"] = val_ppl[-1]
            s["best_val_ppl"] = min(val_ppl)
        else:
            s["final_val_ppl"] = None
            s["best_val_ppl"] = None
        
        # Curriculum metrics: early vs late easy_ratio
        easy_ratio = hist.get("easy_ratio", [])
        if easy_ratio:
            n = len(easy_ratio)
            third = max(n // 3, 1)
            early = sum(easy_ratio[:third]) / third
            late = sum(easy_ratio[-third:]) / third
            s["early_easy_ratio"] = early
            s["late_easy_ratio"] = late
            s["curriculum_shift"] = early - late
        else:
            s["early_easy_ratio"] = None
            s["late_easy_ratio"] = None
            s["curriculum_shift"] = None
        
        # Coverage: average over last 5 points
        coverage = hist.get("coverage", [])
        if len(coverage) >= 5:
            s["avg_coverage_last5"] = sum(coverage[-5:]) / 5
        elif coverage:
            s["avg_coverage_last5"] = sum(coverage) / len(coverage)
        else:
            s["avg_coverage_last5"] = None
        
        # Entropy: average over last 5 points
        entropy = hist.get("entropy", [])
        if len(entropy) >= 5:
            s["avg_entropy_last5"] = sum(entropy[-5:]) / 5
        elif entropy:
            s["avg_entropy_last5"] = sum(entropy) / len(entropy)
        else:
            s["avg_entropy_last5"] = None
        
        summary[exp_name] = s
    return summary

def save_summary_text(summary, out_path_txt):
    lines = []
    header = [
        "experiment",
        "final_val_ppl",
        "best_val_ppl",
        "early_easy_ratio",
        "late_easy_ratio",
        "curriculum_shift",
        "avg_coverage_last5",
        "avg_entropy_last5",
    ]
    lines.append("\t".join(header))
    
    for exp_name, s in summary.items():
        row = [
            exp_name,
            f"{s['final_val_ppl']:.3f}" if s["final_val_ppl"] is not None else "NA",
            f"{s['best_val_ppl']:.3f}" if s["best_val_ppl"] is not None else "NA",
            f"{s['early_easy_ratio']:.3f}" if s["early_easy_ratio"] is not None else "NA",
            f"{s['late_easy_ratio']:.3f}" if s["late_easy_ratio"] is not None else "NA",
            f"{s['curriculum_shift']:.3f}" if s["curriculum_shift"] is not None else "NA",
            f"{s['avg_coverage_last5']:.3f}" if s["avg_coverage_last5"] is not None else "NA",
            f"{s['avg_entropy_last5']:.3f}" if s["avg_entropy_last5"] is not None else "NA",
        ]
        lines.append("\t".join(row))
    
    with open(out_path_txt, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved summary table (txt): {out_path_txt}")

def main():
    print("Loading metrics from:", RESULTS_DIR)
    metrics = load_all_metrics(RESULTS_DIR)
    print(f"Found {len(metrics)} experiments:", list(metrics.keys()))
    
    # You can restrict to a subset here if you want
    # e.g. only baseline + full router + a couple ablations
    FOCUS_EXPERIMENTS = [
        "baseline_uniform",
        "router_full_improvement",
        "router_random_reward",
        "router_text_stats_only",
        "router_hidden_only",
        "router_mlp_router",
        "router_temp_0_5",
        "router_temp_2_0",
        "router_no_entropy",
        "router_high_entropy",
        "router_curriculum_bias",
    ]
    
    # Filter to experiments that exist
    focus_existing = [e for e in FOCUS_EXPERIMENTS if e in metrics]
    print("Focusing on experiments:", focus_existing)
    
    # --- Plots ---
    # Validation perplexity
    plot_metric_over_time(
        metrics,
        metric_name="val_ppl",
        out_path=os.path.join(OUT_DIR, "val_ppl_vs_step.png"),
        experiments=focus_existing,
        ylabel="Validation Perplexity",
        use_step=True,
    )
    
    # Easy ratio (only routers will have it)
    plot_metric_over_time(
        metrics,
        metric_name="easy_ratio",
        out_path=os.path.join(OUT_DIR, "easy_ratio_vs_step.png"),
        experiments=focus_existing,
        ylabel="Easy sample ratio",
        use_step=True,
    )
    
    # Coverage
    plot_metric_over_time(
        metrics,
        metric_name="coverage",
        out_path=os.path.join(OUT_DIR, "coverage_vs_step.png"),
        experiments=focus_existing,
        ylabel="Coverage",
        use_step=True,
    )
    
    # Entropy
    plot_metric_over_time(
        metrics,
        metric_name="entropy",
        out_path=os.path.join(OUT_DIR, "entropy_vs_step.png"),
        experiments=focus_existing,
        ylabel="Selection entropy",
        use_step=True,
    )
    
    # --- Summary table ---
    summary = summarize_experiments(metrics)
    out_json = os.path.join(OUT_DIR, "summary_table.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary table (json): {out_json}")
    
    out_txt = os.path.join(OUT_DIR, "summary_table.txt")
    save_summary_text(summary, out_txt)

if __name__ == "__main__":
    main()
