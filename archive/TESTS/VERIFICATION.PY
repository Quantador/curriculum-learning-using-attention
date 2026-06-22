# verification_experiments.py
# Comprehensive verification suite to validate curriculum learning results
# 
# Experiments:
# 1. Multiple random seeds (check consistency)
# 2. Ablation studies (remove each component)
# 3. Reverse curriculum (hard→easy, should be worse)
# 4. Fixed schedules (compare to learned curriculum)
# 5. Statistical significance tests

import math, torch, random, json
from torch import nn
from pathlib import Path
from collections import defaultdict
import numpy as np

# Import from your main file (assumes it's in same directory)
import sys
sys.path.append('/home/claude')
from evalRouter2 import *

# -------------------------
# Experiment 1: Multiple Random Seeds
# -------------------------
def experiment_multiple_seeds(train_ds, val_ds, n_seeds=5):
    """Run both baseline and router with multiple random seeds"""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Multiple Random Seeds")
    print("="*70)
    print(f"Running {n_seeds} independent trials with different random seeds...")
    
    baseline_ppls = []
    router_ppls = []
    
    for seed in range(n_seeds):
        print(f"\n--- Trial {seed+1}/{n_seeds} (seed={seed}) ---")
        
        # Set seed
        random.seed(seed)
        torch.manual_seed(seed)
        
        # Baseline
        print("Running baseline...")
        baseline_metrics = MetricsTracker()
        baseline_diversity = DiversityTracker(len(train_ds))
        baseline_model = train_baseline(train_ds, val_ds, baseline_metrics, baseline_diversity)
        baseline_ppl = baseline_metrics.get_final_ppl()
        baseline_ppls.append(baseline_ppl)
        print(f"Baseline PPL: {baseline_ppl:.2f}")
        
        # Reset seed
        random.seed(seed)
        torch.manual_seed(seed)
        
        # Router
        print("Running router...")
        router_metrics = MetricsTracker()
        router_diversity = DiversityTracker(len(train_ds))
        router_model, router_net = train_router(train_ds, val_ds, router_metrics, router_diversity)
        router_ppl = router_metrics.get_final_ppl()
        router_ppls.append(router_ppl)
        print(f"Router PPL: {router_ppl:.2f}")
        print(f"Improvement: {((baseline_ppl - router_ppl) / baseline_ppl * 100):.2f}%")
    
    # Statistical analysis
    baseline_mean = np.mean(baseline_ppls)
    baseline_std = np.std(baseline_ppls)
    router_mean = np.mean(router_ppls)
    router_std = np.std(router_ppls)
    improvements = [(b - r) / b * 100 for b, r in zip(baseline_ppls, router_ppls)]
    improvement_mean = np.mean(improvements)
    improvement_std = np.std(improvements)
    
    # Paired t-test
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(baseline_ppls, router_ppls)
    
    print("\n" + "="*70)
    print("RESULTS - Multiple Seeds")
    print("="*70)
    print(f"Baseline PPL:  {baseline_mean:.2f} ± {baseline_std:.2f}")
    print(f"Router PPL:    {router_mean:.2f} ± {router_std:.2f}")
    print(f"Improvement:   {improvement_mean:.2f}% ± {improvement_std:.2f}%")
    print(f"\nPaired t-test: t={t_stat:.3f}, p={p_value:.4f}")
    
    if p_value < 0.05 and improvement_mean > 0:
        print("✓✓ STATISTICALLY SIGNIFICANT improvement (p < 0.05)")
    elif p_value < 0.10 and improvement_mean > 0:
        print("✓ Marginally significant improvement (p < 0.10)")
    else:
        print("✗ NOT statistically significant - may be random")
    
    return {
        'baseline_ppls': baseline_ppls,
        'router_ppls': router_ppls,
        'p_value': p_value,
        'improvement_mean': improvement_mean
    }

# -------------------------
# Experiment 2: Ablation Studies
# -------------------------
def experiment_ablations(train_ds, val_ds):
    """Test removing each component to see what matters"""
    print("\n" + "="*70)
    print("EXPERIMENT 2: Ablation Studies")
    print("="*70)
    
    results = {}
    
    # Save original config
    orig_temp = cfg.temp
    orig_ent = cfg.lambda_ent
    
    # 1. No curriculum bias
    print("\n--- Ablation 1: Remove curriculum bias ---")
    random.seed(0)
    torch.manual_seed(0)
    metrics = MetricsTracker()
    diversity = DiversityTracker(len(train_ds))
    model, router = train_router_ablation(train_ds, val_ds, metrics, diversity, 
                                          use_curriculum_bias=False)
    results['no_curriculum_bias'] = metrics.get_final_ppl()
    print(f"PPL without curriculum bias: {results['no_curriculum_bias']:.2f}")
    
    # 2. No improvement clipping
    print("\n--- Ablation 2: Remove improvement clipping ---")
    random.seed(0)
    torch.manual_seed(0)
    metrics = MetricsTracker()
    diversity = DiversityTracker(len(train_ds))
    model, router = train_router_ablation(train_ds, val_ds, metrics, diversity,
                                          clip_improvement=False)
    results['no_clipping'] = metrics.get_final_ppl()
    print(f"PPL without clipping: {results['no_clipping']:.2f}")
    
    # 3. Mean pooling instead of hierarchical
    print("\n--- Ablation 3: Mean pooling (not hierarchical) ---")
    random.seed(0)
    torch.manual_seed(0)
    metrics = MetricsTracker()
    diversity = DiversityTracker(len(train_ds))
    model, router = train_router_ablation(train_ds, val_ds, metrics, diversity,
                                          use_hierarchical=False)
    results['mean_pooling'] = metrics.get_final_ppl()
    print(f"PPL with mean pooling: {results['mean_pooling']:.2f}")
    
    # 4. Full system
    print("\n--- Full system (all components) ---")
    random.seed(0)
    torch.manual_seed(0)
    metrics = MetricsTracker()
    diversity = DiversityTracker(len(train_ds))
    model, router = train_router(train_ds, val_ds, metrics, diversity)
    results['full_system'] = metrics.get_final_ppl()
    print(f"PPL with full system: {results['full_system']:.2f}")
    
    # Baseline for reference
    print("\n--- Baseline ---")
    random.seed(0)
    torch.manual_seed(0)
    metrics = MetricsTracker()
    diversity = DiversityTracker(len(train_ds))
    model = train_baseline(train_ds, val_ds, metrics, diversity)
    results['baseline'] = metrics.get_final_ppl()
    print(f"Baseline PPL: {results['baseline']:.2f}")
    
    # Restore config
    cfg.temp = orig_temp
    cfg.lambda_ent = orig_ent
    
    print("\n" + "="*70)
    print("ABLATION RESULTS")
    print("="*70)
    baseline_ppl = results['baseline']
    for name, ppl in results.items():
        if name != 'baseline':
            improvement = (baseline_ppl - ppl) / baseline_ppl * 100
            print(f"{name:25s}: {ppl:8.2f}  ({improvement:+6.2f}%)")
    print(f"{'baseline':25s}: {baseline_ppl:8.2f}  (  0.00%)")
    
    return results

# -------------------------
# Experiment 3: Reverse Curriculum (Hard→Easy)
# -------------------------
def experiment_reverse_curriculum(train_ds, val_ds):
    """Force hard→easy curriculum (should be worse)"""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Reverse Curriculum (Hard→Easy)")
    print("="*70)
    print("Training with REVERSED curriculum (hard first, easy later)")
    print("This should perform WORSE than normal curriculum\n")
    
    random.seed(0)
    torch.manual_seed(0)
    
    metrics = MetricsTracker()
    diversity = DiversityTracker(len(train_ds))
    model, router = train_router_reverse(train_ds, val_ds, metrics, diversity)
    reverse_ppl = metrics.get_final_ppl()
    
    # Get normal curriculum for comparison
    random.seed(0)
    torch.manual_seed(0)
    metrics_normal = MetricsTracker()
    diversity_normal = DiversityTracker(len(train_ds))
    model_normal, router_normal = train_router(train_ds, val_ds, metrics_normal, diversity_normal)
    normal_ppl = metrics_normal.get_final_ppl()
    
    # Get baseline
    random.seed(0)
    torch.manual_seed(0)
    metrics_baseline = MetricsTracker()
    diversity_baseline = DiversityTracker(len(train_ds))
    model_baseline = train_baseline(train_ds, val_ds, metrics_baseline, diversity_baseline)
    baseline_ppl = metrics_baseline.get_final_ppl()
    
    print("\n" + "="*70)
    print("REVERSE CURRICULUM RESULTS")
    print("="*70)
    print(f"Baseline (uniform):       {baseline_ppl:.2f}")
    print(f"Normal curriculum:        {normal_ppl:.2f}  ({(baseline_ppl-normal_ppl)/baseline_ppl*100:+.2f}%)")
    print(f"REVERSED curriculum:      {reverse_ppl:.2f}  ({(baseline_ppl-reverse_ppl)/baseline_ppl*100:+.2f}%)")
    print()
    
    if normal_ppl < reverse_ppl:
        print("✓ Normal curriculum is BETTER than reversed (as expected)")
        print("  This validates that easy→hard order matters!")
    else:
        print("✗ Reversed curriculum is not worse - curriculum order may not matter")
        print("  Results might be accidental")
    
    return {
        'baseline': baseline_ppl,
        'normal': normal_ppl,
        'reverse': reverse_ppl
    }

# -------------------------
# Experiment 4: Fixed Schedules
# -------------------------
def experiment_fixed_schedules(train_ds, val_ds):
    """Compare learned curriculum to fixed schedules"""
    print("\n" + "="*70)
    print("EXPERIMENT 4: Fixed Schedules vs Learned Curriculum")
    print("="*70)
    
    results = {}
    
    # 1. Always easy
    print("\n--- Schedule 1: Always easy (100% easy throughout) ---")
    random.seed(0)
    torch.manual_seed(0)
    metrics = MetricsTracker()
    diversity = DiversityTracker(len(train_ds))
    model = train_fixed_schedule(train_ds, val_ds, metrics, diversity, schedule='always_easy')
    results['always_easy'] = metrics.get_final_ppl()
    print(f"Always easy PPL: {results['always_easy']:.2f}")
    
    # 2. Always hard
    print("\n--- Schedule 2: Always hard (100% hard throughout) ---")
    random.seed(0)
    torch.manual_seed(0)
    metrics = MetricsTracker()
    diversity = DiversityTracker(len(train_ds))
    model = train_fixed_schedule(train_ds, val_ds, metrics, diversity, schedule='always_hard')
    results['always_hard'] = metrics.get_final_ppl()
    print(f"Always hard PPL: {results['always_hard']:.2f}")
    
    # 3. Linear schedule (80%→20% easy)
    print("\n--- Schedule 3: Linear schedule (80%→20% easy) ---")
    random.seed(0)
    torch.manual_seed(0)
    metrics = MetricsTracker()
    diversity = DiversityTracker(len(train_ds))
    model = train_fixed_schedule(train_ds, val_ds, metrics, diversity, schedule='linear')
    results['linear_schedule'] = metrics.get_final_ppl()
    print(f"Linear schedule PPL: {results['linear_schedule']:.2f}")
    
    # 4. Learned curriculum
    print("\n--- Learned curriculum (router) ---")
    random.seed(0)
    torch.manual_seed(0)
    metrics = MetricsTracker()
    diversity = DiversityTracker(len(train_ds))
    model, router = train_router(train_ds, val_ds, metrics, diversity)
    results['learned_curriculum'] = metrics.get_final_ppl()
    print(f"Learned curriculum PPL: {results['learned_curriculum']:.2f}")
    
    # 5. Baseline
    random.seed(0)
    torch.manual_seed(0)
    metrics = MetricsTracker()
    diversity = DiversityTracker(len(train_ds))
    model = train_baseline(train_ds, val_ds, metrics, diversity)
    results['baseline'] = metrics.get_final_ppl()
    
    print("\n" + "="*70)
    print("FIXED SCHEDULE RESULTS")
    print("="*70)
    baseline_ppl = results['baseline']
    
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    print("Ranking (best to worst):")
    for rank, (name, ppl) in enumerate(sorted_results, 1):
        improvement = (baseline_ppl - ppl) / baseline_ppl * 100
        marker = "★" if name == 'learned_curriculum' else " "
        print(f"{rank}. {marker} {name:20s}: {ppl:8.2f}  ({improvement:+6.2f}%)")
    
    if sorted_results[0][0] == 'learned_curriculum':
        print("\n✓✓ Learned curriculum is BEST - router actually learning!")
    elif 'learned_curriculum' in [x[0] for x in sorted_results[:2]]:
        print("\n✓ Learned curriculum in top 2 - competitive with best fixed schedule")
    else:
        print("\n✗ Learned curriculum not in top 2 - fixed schedules work better")
        print("  Router may not be learning effectively")
    
    return results

# -------------------------
# Helper: Ablation trainer
# -------------------------
def train_router_ablation(train_ds, val_ds, metrics, diversity, 
                         use_curriculum_bias=True,
                         clip_improvement=True,
                         use_hierarchical=True):
    """Modified trainer for ablations"""
    model = TinyGPT(vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff, cfg.block).to(cfg.device)
    d_router_input = cfg.n_chunks * cfg.d_model + 4 if use_hierarchical else cfg.d_model + 4
    router = AttentionRouter(d_input=d_router_input, d_k=64).to(cfg.device)
    
    opt_lm = torch.optim.AdamW(model.parameters(), lr=cfg.lr_lm)
    opt_router = torch.optim.AdamW(router.parameters(), lr=cfg.lr_router)
    loss_fn = nn.CrossEntropyLoss()
    
    global_step = 0
    
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        router.train()
        idx_loader = make_index_loader(len(train_ds), cfg.pool)
        
        for pool_indices in idx_loader:
            if len(pool_indices) < cfg.batch:
                continue
            global_step += 1
            
            batch_data = [train_ds[i] for i in pool_indices]
            xs, ys, diffs = zip(*batch_data)
            X = torch.stack(xs, 0).to(cfg.device)
            Y = torch.stack(ys, 0).to(cfg.device)
            M = X.size(0)
            
            # Feature extraction (hierarchical or mean)
            if use_hierarchical:
                feats = extract_hierarchical_features(model, X)
            else:
                with torch.no_grad():
                    h = model.forward_to_hidden(X)
                    mean_feats = h.mean(dim=1)
                    text_stats = compute_text_statistics(X)
                    feats = torch.cat([mean_feats, text_stats], dim=-1)
            
            scores = router(feats)
            
            # Curriculum bias (optional)
            if use_curriculum_bias:
                training_progress = global_step / (len(train_ds) // cfg.pool * cfg.epochs)
                curriculum_strength = max(0, 1.0 - training_progress)
                if curriculum_strength > 0:
                    difficulty_tensor = torch.tensor([diffs[i] for i in range(M)], 
                                                     device=scores.device, dtype=torch.float32)
                    curriculum_bonus = curriculum_strength * 2.0 * (1 - difficulty_tensor)
                    scores = scores + curriculum_bonus
            
            probs = torch.softmax(scores / cfg.temp, dim=0)
            k = cfg.batch
            topk_idx = torch.topk(probs, k=k, dim=0).indices
            
            X_sel = X[topk_idx]
            Y_sel = Y[topk_idx]
            
            # Loss improvement
            with torch.no_grad():
                loss_before = compute_loss_per_sample(model, X_sel, Y_sel, loss_fn)
            
            logits = model(X_sel)
            loss_lm = loss_fn(logits.reshape(-1, logits.size(-1)), Y_sel.reshape(-1))
            
            opt_lm.zero_grad(set_to_none=True)
            loss_lm.backward()
            opt_lm.step()
            
            with torch.no_grad():
                loss_after = compute_loss_per_sample(model, X_sel, Y_sel, loss_fn)
            
            improvement = loss_before - loss_after
            
            # Clipping (optional)
            if clip_improvement:
                improvement = improvement.clamp(min=0.0)
            
            baseline = improvement.mean()
            
            sel_probs = probs[topk_idx].clamp_min(1e-12)
            reinforce = -((improvement - baseline) * torch.log(sel_probs)).mean()
            
            ent = (probs * torch.log(probs.clamp_min(1e-12))).sum()
            loss_router = reinforce + cfg.lambda_ent * ent
            
            opt_router.zero_grad(set_to_none=True)
            loss_router.backward()
            opt_router.step()
            
            selected_indices = [pool_indices[i] for i in topk_idx.cpu().tolist()]
            selected_diffs = [diffs[i] for i in topk_idx.cpu().tolist()]
            diversity.update(selected_indices, selected_diffs)
        
        val_loss, val_ppl = evaluate(model, val_ds, loss_fn)
        metrics.log(epoch=epoch, step=global_step, val_loss=val_loss, val_ppl=val_ppl)
    
    return model, router

# -------------------------
# Helper: Reverse curriculum trainer
# -------------------------
def train_router_reverse(train_ds, val_ds, metrics, diversity):
    """Train with REVERSED curriculum (hard→easy)"""
    model = TinyGPT(vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff, cfg.block).to(cfg.device)
    d_router_input = cfg.n_chunks * cfg.d_model + 4
    router = AttentionRouter(d_input=d_router_input, d_k=64).to(cfg.device)
    
    opt_lm = torch.optim.AdamW(model.parameters(), lr=cfg.lr_lm)
    opt_router = torch.optim.AdamW(router.parameters(), lr=cfg.lr_router)
    loss_fn = nn.CrossEntropyLoss()
    
    global_step = 0
    
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        router.train()
        idx_loader = make_index_loader(len(train_ds), cfg.pool)
        
        for pool_indices in idx_loader:
            if len(pool_indices) < cfg.batch:
                continue
            global_step += 1
            
            batch_data = [train_ds[i] for i in pool_indices]
            xs, ys, diffs = zip(*batch_data)
            X = torch.stack(xs, 0).to(cfg.device)
            Y = torch.stack(ys, 0).to(cfg.device)
            M = X.size(0)
            
            feats = extract_hierarchical_features(model, X)
            scores = router(feats)
            
            # REVERSED curriculum: hard first, easy later
            training_progress = global_step / (len(train_ds) // cfg.pool * cfg.epochs)
            curriculum_strength = max(0, 1.0 - training_progress)
            
            if curriculum_strength > 0:
                difficulty_tensor = torch.tensor([diffs[i] for i in range(M)], 
                                                 device=scores.device, dtype=torch.float32)
                # REVERSED: Boost hard samples early (opposite of normal)
                curriculum_bonus = curriculum_strength * 2.0 * difficulty_tensor  # Hard=+2, Easy=0
                scores = scores + curriculum_bonus
            
            probs = torch.softmax(scores / cfg.temp, dim=0)
            k = cfg.batch
            topk_idx = torch.topk(probs, k=k, dim=0).indices
            
            X_sel = X[topk_idx]
            Y_sel = Y[topk_idx]
            
            with torch.no_grad():
                loss_before = compute_loss_per_sample(model, X_sel, Y_sel, loss_fn)
            
            logits = model(X_sel)
            loss_lm = loss_fn(logits.reshape(-1, logits.size(-1)), Y_sel.reshape(-1))
            
            opt_lm.zero_grad(set_to_none=True)
            loss_lm.backward()
            opt_lm.step()
            
            with torch.no_grad():
                loss_after = compute_loss_per_sample(model, X_sel, Y_sel, loss_fn)
            
            improvement = (loss_before - loss_after).clamp(min=0.0)
            baseline = improvement.mean()
            
            sel_probs = probs[topk_idx].clamp_min(1e-12)
            reinforce = -((improvement - baseline) * torch.log(sel_probs)).mean()
            
            ent = (probs * torch.log(probs.clamp_min(1e-12))).sum()
            loss_router = reinforce + cfg.lambda_ent * ent
            
            opt_router.zero_grad(set_to_none=True)
            loss_router.backward()
            opt_router.step()
            
            selected_indices = [pool_indices[i] for i in topk_idx.cpu().tolist()]
            selected_diffs = [diffs[i] for i in topk_idx.cpu().tolist()]
            diversity.update(selected_indices, selected_diffs)
        
        val_loss, val_ppl = evaluate(model, val_ds, loss_fn)
        metrics.log(epoch=epoch, step=global_step, val_loss=val_loss, val_ppl=val_ppl)
    
    return model, router

# -------------------------
# Helper: Fixed schedule trainer
# -------------------------
def train_fixed_schedule(train_ds, val_ds, metrics, diversity, schedule='linear'):
    """Train with fixed difficulty schedule"""
    model = TinyGPT(vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff, cfg.block).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_lm)
    loss_fn = nn.CrossEntropyLoss()
    
    global_step = 0
    total_steps = len(train_ds) // cfg.pool * cfg.epochs
    
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        idx_loader = make_index_loader(len(train_ds), cfg.pool)
        
        for pool_indices in idx_loader:
            if len(pool_indices) < cfg.batch:
                continue
            global_step += 1
            
            # Get difficulties
            batch_data = [train_ds[i] for i in pool_indices]
            diffs = [d for _, _, d in batch_data]
            
            # Select based on schedule
            progress = global_step / total_steps
            
            if schedule == 'always_easy':
                # Always prefer easy
                selected = sorted(range(len(diffs)), key=lambda i: diffs[i])[:cfg.batch]
            elif schedule == 'always_hard':
                # Always prefer hard
                selected = sorted(range(len(diffs)), key=lambda i: -diffs[i])[:cfg.batch]
            elif schedule == 'linear':
                # Linear: 80% easy at start → 20% easy at end
                easy_ratio = 0.8 * (1 - progress) + 0.2 * progress
                n_easy = int(cfg.batch * easy_ratio)
                n_hard = cfg.batch - n_easy
                
                easy_idx = [i for i in range(len(diffs)) if diffs[i] == 0]
                hard_idx = [i for i in range(len(diffs)) if diffs[i] == 1]
                
                selected = (random.sample(easy_idx, min(n_easy, len(easy_idx))) +
                           random.sample(hard_idx, min(n_hard, len(hard_idx))))
                if len(selected) < cfg.batch:
                    selected = random.sample(range(len(diffs)), cfg.batch)
            
            # Train on selected
            selected_indices = [pool_indices[i] for i in selected]
            batch_data = [train_ds[i] for i in selected_indices]
            xs, ys, diffs_sel = zip(*batch_data)
            X = torch.stack(xs, 0).to(cfg.device)
            Y = torch.stack(ys, 0).to(cfg.device)
            
            logits = model(X)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            
            diversity.update(selected_indices, diffs_sel)
        
        val_loss, val_ppl = evaluate(model, val_ds, loss_fn)
        metrics.log(epoch=epoch, step=global_step, val_loss=val_loss, val_ppl=val_ppl)
    
    return model

# -------------------------
# Main verification suite
# -------------------------
def main():
    print("="*70)
    print("CURRICULUM LEARNING VERIFICATION SUITE")
    print("="*70)
    print("\nThis will run 4 verification experiments:")
    print("1. Multiple random seeds (statistical significance)")
    print("2. Ablation studies (which components matter)")
    print("3. Reverse curriculum (hard→easy should be worse)")
    print("4. Fixed schedules (vs learned curriculum)")
    print("\nWarning: This will take a while (~2-3 hours)!")
    print()
    
    input("Press Enter to continue or Ctrl+C to cancel...")
    
    # Load data once
    print("\nLoading dataset...")
    train_chunks = make_mixed_chunks("train")
    val_chunks = make_mixed_chunks("validation")
    train_ds = MixedLMDataset(train_chunks)
    val_ds = MixedLMDataset(val_chunks)
    print(f"✓ Train: {len(train_ds)} samples, Val: {len(val_ds)} samples\n")
    
    Path(cfg.save_dir).mkdir(exist_ok=True)
    
    all_results = {}
    
    # Experiment 1: Multiple seeds
    seed_results = experiment_multiple_seeds(train_ds, val_ds, n_seeds=5)
    all_results['multiple_seeds'] = seed_results
    
    # Experiment 2: Ablations
    ablation_results = experiment_ablations(train_ds, val_ds)
    all_results['ablations'] = ablation_results
    
    # Experiment 3: Reverse curriculum
    reverse_results = experiment_reverse_curriculum(train_ds, val_ds)
    all_results['reverse_curriculum'] = reverse_results
    
    # Experiment 4: Fixed schedules
    schedule_results = experiment_fixed_schedules(train_ds, val_ds)
    all_results['fixed_schedules'] = schedule_results
    
    # Save all results
    with open(f"{cfg.save_dir}/verification_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Final summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    print("\n✓ Experiment 1 - Multiple Seeds:")
    if seed_results['p_value'] < 0.05:
        print(f"  ✓✓ PASSED: Statistically significant (p={seed_results['p_value']:.4f})")
    else:
        print(f"  ✗ FAILED: Not significant (p={seed_results['p_value']:.4f})")
    
    print("\n✓ Experiment 2 - Ablations:")
    full_ppl = ablation_results['full_system']
    baseline_ppl = ablation_results['baseline']
    if full_ppl < min(ablation_results['no_curriculum_bias'], 
                      ablation_results['no_clipping'],
                      ablation_results['mean_pooling']):
        print("  ✓✓ PASSED: Full system beats all ablations")
    else:
        print("  ~ PARTIAL: Some ablations work as well or better")
    
    print("\n✓ Experiment 3 - Reverse Curriculum:")
    if reverse_results['normal'] < reverse_results['reverse']:
        print("  ✓✓ PASSED: Normal curriculum beats reversed")
    else:
        print("  ✗ FAILED: Reversed is not worse - order may not matter")
    
    print("\n✓ Experiment 4 - Fixed Schedules:")
    sorted_schedules = sorted(schedule_results.items(), key=lambda x: x[1])
    if sorted_schedules[0][0] == 'learned_curriculum':
        print("  ✓✓ PASSED: Learned curriculum is best")
    elif 'learned_curriculum' in [x[0] for x in sorted_schedules[:2]]:
        print("  ✓ PARTIAL: Learned curriculum in top 2")
    else:
        print("  ✗ FAILED: Fixed schedules work better")
    
    print("\n" + "="*70)
    print(f"✓ All results saved to {cfg.save_dir}/verification_results.json")
    print("="*70)

if __name__ == "__main__":
    main()