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

Verified against the installed lm-eval-harness 0.4.12 (2026-08-18):
  - `import lm_eval.tasks` is required explicitly -- 0.4.12's `lm_eval/__init__.py` only
    lazy-loads `evaluate`/`simple_evaluate` via `__getattr__`, so `lm_eval.tasks` is not
    auto-exposed as a package attribute the way older versions did.
  - "story_cloze_2016" -> "storycloze_2016" (no underscore between "story" and "cloze") --
    that's the name actually registered by lm_eval/tasks/storycloze/storycloze_2016.yaml.

A first full run (2026-08-20, GPT-2 XL) scored 11 of the 22 tasks nan. Three unrelated causes,
and what was done about each:

  1. datasets >= 4.0 removed dataset-script support entirely ("RuntimeError: Dataset scripts
     are no longer supported"), and this project pins datasets==4.4.2. That killed social_iqa
     (allenai/social_i_qa -> social_i_qa.py), wsc273 (winograd_wsc -> winograd_wsc.py) and
     storycloze_2016 (LSDSem/story_cloze -> story_cloze.py). The first two are FIXED by the
     script-free mirrors in eval_tasks/social_iqa_parquet.yaml and eval_tasks/wsc273_parquet.yaml
     (identical schemas and row counts; see those files). storycloze_2016 has no script-free
     mirror -- LSDSem/story_cloze is also gated -- so it is still expected to score nan until
     someone requests Hub access AND a non-script copy exists. It is the one task here with no
     code-side fix.
  2. All 6 BBH tasks hit "requested max tokens to generate (1024) must be less than model's
     maximum sequence length (1024)". FIXED via _BBH_MAX_GEN_TOKS below -- see that constant.
  3. super_glue_axb / super_glue_axg are registered by no lm-eval 0.4.12 task (its
     lm_eval/tasks/super_glue/ has no axb/axg directory). FIXED by defining them locally in
     eval_tasks/super_glue_ax{b,g}.yaml -- `aps/super_glue` still serves both configs, script
     free and with real (non-hidden) entailment labels, so they are scorable after all.
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
    "social_iqa_parquet",  # custom task; stock `social_iqa` is script-loaded, see module docstring
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "commonsense_qa",
    "wsc273_parquet",  # custom task; stock `wsc273` is script-loaded, see module docstring
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
    "super_glue_axb",  # custom task, eval_tasks/super_glue_axb.yaml -- see module docstring
    "super_glue_axg",  # custom task, eval_tasks/super_glue_axg.yaml -- see module docstring
    "storycloze_2016",  # gated AND script-loaded; no fix available, see module docstring
]

ALL_TASKS = IN_DOMAIN_TASKS + OOD_TASKS

_BBH_FEWSHOT = 3  # OPUS: BBH is evaluated 3-shot; everything else is zero-shot.

# Generation budget for the BBH tasks, which are the suite's only generate_until tasks.
#
# lm-eval splits a model's context window between prompt and generation:
#     max_ctx_len = model.max_length - max_gen_toks
# and asserts max_ctx_len > 0. BBH's _cot_fewshot_template_yaml hardcodes
# generation_kwargs.max_gen_toks = 1024, which on a GPT-2-family model (max_length =
# 1024 exactly) leaves 0 tokens of context and fails that assertion -- every BBH task
# scored nan. Overriding it here is what makes them scorable at all.
#
# 128 is a deliberate compromise, not a tuned value. Measured 3-shot CoT prompt lengths
# under the GPT-2 tokenizer (p50/max tokens): sports_understanding 240/246,
# logical_deduction 797/809, tracking_shuffled_objects 857/867, colored_objects 917/937,
# penguins_in_a_table 889/952, disambiguation_qa 993/1006. Leaving 1024-128 = 896 for
# context therefore fits four of the six subsets outright and left-truncates the other
# two by roughly 40-110 tokens, which costs part of the FIRST few-shot exemplar while
# preserving the actual question (it sits at the end of the prompt).
#
# Raising this is counterproductive: every extra generated token comes straight out of
# the prompt. At the degenerate end, max_gen_toks=1023 satisfies the assertion but
# leaves a 1-token prompt, so the model generates unconditioned text and scores ~0.
# Lowering it buys prompt room but eventually truncates the chain-of-thought answer
# before it can emit "the answer is X", which is the string BBH's regex filter extracts.
#
# Caveat worth carrying into any comparison: BBH 3-shot CoT does not really fit a
# 1024-token window, so these scores are NOT strictly protocol-identical to OPUS's,
# which evaluated the same benchmark without this constraint.
_BBH_MAX_GEN_TOKS = 128

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
        is_bbh = task.startswith("bbh_")
        num_fewshot = _BBH_FEWSHOT if is_bbh else 0
        # simple_evaluate merges gen_kwargs into the task's own generation_kwargs
        # (set_config(..., update=True)), so this replaces only max_gen_toks and leaves
        # BBH's `until`/`do_sample`/`temperature` intact. Passed for BBH alone: it is the
        # only generate_until task group here, and generation_kwargs is inert for the
        # multiple_choice tasks anyway.
        gen_kwargs = {"max_gen_toks": _BBH_MAX_GEN_TOKS} if is_bbh else None
        try:
            results = lm_eval.simple_evaluate(
                model=lm, tasks=[task], num_fewshot=num_fewshot, task_manager=task_manager,
                gen_kwargs=gen_kwargs,
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
