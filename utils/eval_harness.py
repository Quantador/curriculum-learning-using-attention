"""Benchmark evaluation harness for OPUS-comparable scoring.

Reproduces the benchmark suite OPUS (arXiv:2602.05400) reports in Table 3 (in-domain) and
Table 5 (out-of-distribution), via lm-evaluation-harness, against a trained model.

Task-count note: OPUS's paper publishes 16 benchmark scores (10 in-domain columns in Table 3,
6 out-of-distribution columns in Table 5). The lm-eval-harness task-name lists below hold 22
entries (12 in IN_DOMAIN_TASKS, 10 in OOD_TASKS) covering those same 16 benchmarks, because
some of them are split into several separately-registered lm-eval task names: ANLI is 3 tasks
(anli_r1/r2/r3) and BBH is 6 (one per curated subset). So "16" describes OPUS's published
columns; 12/10 describe this file's registry-task lists. See suite_averages() for how the
per-task scores are aggregated (and note ANLI's 3 sub-tasks therefore carry 3x the weight of
any other single benchmark in in_domain_avg).

Scale note: every score produced here is a FRACTION in [0, 1] (lm-eval-harness's native
output), never a percentage -- see run_eval_suite()/suite_averages(). compare_to_opus.py
multiplies by 100 at its print site to line up with OPUS's published 0-100 figures.

Task-name caveat: exact lm-eval-harness task-registry names can shift across versions. Verify
ALL_TASKS against your installed version with:
    python -c "import lm_eval, lm_eval.tasks; print(sorted(lm_eval.tasks.TaskManager().all_tasks))" | less
and adjust names below if any don't match -- this list was written against lm-eval-harness's
standard task registry as of this project's pyproject.toml pin (lm-eval>=0.4.5).

Verified against the installed lm-eval-harness 0.4.12 (2026-08-18); two names below needed
adjusting for that version:
  - `import lm_eval.tasks` is required explicitly -- 0.4.12's `lm_eval/__init__.py` only
    lazy-loads `evaluate`/`simple_evaluate` via `__getattr__`, so `lm_eval.tasks` is not
    auto-exposed as a package attribute the way older versions did.
  - "story_cloze_2016" -> "storycloze_2016" (no underscore between "story" and "cloze") --
    that's the name actually registered by lm_eval/tasks/storycloze/storycloze_2016.yaml.
    Its dataset (`LSDSem/story_cloze`) is still gated on the HF Hub, so this task is still
    expected to degrade to nan without manual dataset access approval; only the *name* changed.
  - "super_glue_axb" / "super_glue_axg" are kept as-is even though lm-eval-harness 0.4.12 does
    not register any AX-b/AX-g task at all (checked: no task name containing "ax" resembling
    AX-b/AX-g anywhere in `TaskManager().all_tasks`, and `lm_eval/tasks/super_glue/` on disk has
    no axb/axg subdirectory -- SuperGLUE's diagnostic sets were dropped from the bundled
    registry, likely because their HF dataset config still relies on a script-based loader that
    modern `datasets` versions no longer support). There is no equivalent task elsewhere in the
    registry to substitute, so these two names are left as documented placeholders that will hit
    the "unknown task" branch below and score nan -- functionally identical to the gated-dataset
    case, just for a different underlying reason. Revisit if a future lm-eval-harness release
    restores them.
"""
from __future__ import annotations

import math
from pathlib import Path

import lm_eval
import lm_eval.tasks
from lm_eval.models.huggingface import HFLM

CUSTOM_TASKS_DIR = Path(__file__).resolve().parent.parent / "eval_tasks"

# OPUS Table 3 in-domain suite.
IN_DOMAIN_TASKS = [
    "mmlu_full_answer",  # custom task, see eval_tasks/mmlu_full_answer.yaml
    "anli_r1", "anli_r2", "anli_r3",
    "hellaswag",
    "piqa",
    "social_iqa",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "commonsense_qa",
    "wsc273",
]

# OPUS Table 5 out-of-distribution suite. BBH subset matches the paper's own curated list
# (Section 6.1: "Tracking Shuffled Objects, Reasoning about Colored Objects, Logical
# Deduction, Disambiguation QA, Penguins in a Table, and Sports Understanding").
OOD_TASKS = [
    "bbh_cot_fewshot_tracking_shuffled_objects_three_objects",
    "bbh_cot_fewshot_reasoning_about_colored_objects",
    "bbh_cot_fewshot_logical_deduction_three_objects",
    "bbh_cot_fewshot_disambiguation_qa",
    "bbh_cot_fewshot_penguins_in_a_table",
    "bbh_cot_fewshot_sports_understanding",
    "race",
    "super_glue_axb",  # AX-b -- unregistered in lm-eval-harness 0.4.12, see module docstring
    "super_glue_axg",  # AX-g -- unregistered in lm-eval-harness 0.4.12, see module docstring
    "storycloze_2016",  # gated dataset -- may be unavailable, see run_eval_suite() docstring
]

ALL_TASKS = IN_DOMAIN_TASKS + OOD_TASKS

_BBH_FEWSHOT = 3  # OPUS: BBH is evaluated 3-shot; everything else is zero-shot.

# Preference order for picking a task's headline metric out of lm-eval-harness's per-task
# results dict. Keys look like "<metric>,<filter>" (e.g. "acc,none", "acc_norm,none") -- the
# filter suffix varies by task/version, so we match on the metric name (the part before the
# first comma), not the full key string or a startswith() prefix (startswith("acc") would
# wrongly match "acc_norm,..." too, which is exactly the bug this ordering fixes).
#
# acc_norm (length-normalized accuracy) is preferred over raw acc wherever both are reported --
# hellaswag/piqa/arc_easy/arc_challenge in particular declare acc before acc_norm in their
# metric_list, so a naive "first match" picks the weaker, length-biased acc instead of the
# acc_norm figure OPUS (and virtually every published leaderboard) actually reports for these.
# exact_match covers BBH's generate_until tasks, which report no acc/acc_norm at all. Plain acc
# is the fallback for tasks (e.g. the custom mmlu_full_answer) that only ever report it.
_METRIC_PREFERENCE = ("acc_norm", "exact_match", "acc")


def _pick_metric_key(task_results: dict) -> str:
    """Return the results-dict key for a task's headline metric, per _METRIC_PREFERENCE."""
    for preferred in _METRIC_PREFERENCE:
        for key in task_results:
            if key.split(",", 1)[0] == preferred:
                return key
    raise KeyError(
        f"none of {_METRIC_PREFERENCE} found in task results: {sorted(task_results)}"
    )


def run_eval_suite(
    model, tokenizer, cfg, batch_size: int = 16, tasks: list[str] | None = None,
) -> dict[str, float]:
    """Run `tasks` (default: the full ALL_TASKS suite) against `model` (a
    transformers.PreTrainedModel -- e.g. HFCausalLM.hf, not the HFCausalLM wrapper) and return
    {task_name: score}.

    Scores are FRACTIONS in [0, 1] (lm-eval-harness's native accuracy scale), NOT percentages:
    a score of 0.35 means 35% accuracy. Anything comparing these against OPUS's published
    0-100 table figures must scale by 100 first (compare_to_opus.py does this at its print
    site). tests/test_eval_harness.py asserts this range.

    `tasks` exists so a caller can request a smaller subset without touching ALL_TASKS itself --
    e.g. tests/test_eval_harness.py's evaluate_checkpoint.py integration test passes a 2-task
    subset so it finishes in about a minute on a CPU dev machine instead of the ~30-90 minutes
    the real ALL_TASKS suite (22 tasks) takes there; production callers (the default, and every
    other call site in this codebase) are unaffected and still get every task.

    A task that fails to load or evaluate (e.g. StoryCloze's gated dataset access, or a task
    name no longer present in the installed lm-eval-harness's registry) is logged as a warning
    and scored `float("nan")` rather than failing the whole suite, so one unavailable benchmark
    doesn't lose every other result from a real (expensive) training run.
    """
    if tasks is None:
        tasks = ALL_TASKS
    # NOTE: `device=` is inert on this call path -- when HFLM is given a pre-built model
    # instance (rather than a `pretrained="<hf-name>"` string) it ignores the argument and
    # reads `self._model.device` instead. The real device placement is therefore the caller's
    # `.to(cfg.device)` on the model BEFORE it is passed in (see evaluate_checkpoint.py); this
    # argument is kept only for clarity/forward-compatibility.
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=cfg.device)
    task_manager = lm_eval.tasks.TaskManager(include_path=str(CUSTOM_TASKS_DIR))

    scores: dict[str, float] = {}
    for task in tasks:
        num_fewshot = _BBH_FEWSHOT if task.startswith("bbh_") else 0
        try:
            results = lm_eval.simple_evaluate(
                model=lm, tasks=[task], num_fewshot=num_fewshot, task_manager=task_manager,
            )
            task_results = results["results"][task]
            metric_key = _pick_metric_key(task_results)
            scores[task] = float(task_results[metric_key])
        except Exception as exc:
            print(f"[eval_harness] WARNING: task {task!r} failed ({exc!r}); scoring as nan.")
            scores[task] = float("nan")
    return scores


def suite_averages(scores: dict[str, float]) -> dict[str, float]:
    """Mean of IN_DOMAIN_TASKS and mean of OOD_TASKS, skipping nan (unavailable) tasks --
    matches how OPUS reports its Table 3 (in-domain) and Table 5 (OOD) averages.

    Averages are FRACTIONS in [0, 1], on the same scale as run_eval_suite()'s per-task scores,
    NOT percentages (see this module's docstring).

    Also returns `in_domain_n` / `ood_n`: how many tasks actually contributed to each mean.
    These matter because skipping nan makes the sample size invisible otherwise -- on a real
    run most of OOD_TASKS is expected to be nan (6 BBH tasks exceed GPT-2's 1024-token context,
    AX-b/AX-g are unregistered in lm-eval 0.4.12, StoryCloze's dataset is gated), so `ood_avg`
    can easily be a 1-task average printed next to OPUS's genuine 6-benchmark figure.
    compare_to_opus.py prints these counts so that is visible rather than implied.
    """
    def _mean(task_names: list[str]) -> float:
        vals = [scores[t] for t in task_names if t in scores and not math.isnan(scores[t])]
        return sum(vals) / len(vals) if vals else float("nan")

    def _n(task_names: list[str]) -> int:
        return sum(1 for t in task_names if t in scores and not math.isnan(scores[t]))

    return {
        "in_domain_avg": _mean(IN_DOMAIN_TASKS),
        "ood_avg": _mean(OOD_TASKS),
        "in_domain_n": _n(IN_DOMAIN_TASKS),
        "ood_n": _n(OOD_TASKS),
    }
