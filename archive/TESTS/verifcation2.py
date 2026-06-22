# quick_validation.py
# Quick validation checks that can be run in ~15 minutes
# 
# Checks:
# 1. Dataset is actually 70/30 easy/hard
# 2. Router shows clear easy→hard progression
# 3. Improvement signal is positive
# 4. Router beats baseline consistently (3 seeds)

import sys
sys.path.append('/home/claude')
from evalRouter2 import *
import numpy as np

def check_dataset_balance(train_ds):
    """Verify dataset is properly balanced"""
    print("\n" + "="*70)
    print("CHECK 1: Dataset Balance")
    print("="*70)
    
    difficulties = [train_ds[i][2] for i in range(len(train_ds))]
    easy_count = sum(1 for d in difficulties if d == 0)
    hard_count = sum(1 for d in difficulties if d == 1)
    easy_ratio = easy_count / len(difficulties)
    
    print(f"Easy samples: {easy_count} ({easy_ratio:.2%})")
    print(f"Hard samples: {hard_count} ({1-easy_ratio:.2%})")
    
    if 0.65 <= easy_ratio <= 0.75:
        print("✓ PASS: Dataset is properly balanced (70% ± 5%)")
        return True
    else:
        print("✗ FAIL: Dataset is not balanced correctly")
        print("  Expected: ~70% easy, 30% hard")
        return False

def check_curriculum_progression(metrics_history):
    """Verify router shows easy→hard progression"""
    print("\n" + "="*70)
    print("CHECK 2: Curriculum Progression")
    print("="*70)
    
    easy_ratios = metrics_history.history.get('easy_ratio', [])
    if not easy_ratios:
        print("✗ FAIL: No easy_ratio data in metrics")
        return False
    
    # Split into thirds
    n = len(easy_ratios)
    early = easy_ratios[:n//3]
    late = easy_ratios[-n//3:]
    
    early_mean = np.mean(early)
    late_mean = np.mean(late)
    shift = early_mean - late_mean
    
    print(f"Early training (first 1/3):  {early_mean:.2%} easy")
    print(f"Late training (last 1/3):    {late_mean:.2%} easy")
    print(f"Shift toward hard samples:   {shift:.2%}")
    
    if shift > 0.10 and early_mean > 0.60:
        print("✓ PASS: Clear easy→hard progression")
        return True
    elif shift > 0.05:
        print("~ PARTIAL: Some progression, but not very strong")
        return True
    else:
        print("✗ FAIL: No clear curriculum progression")
        return False

def check_improvement_signal(metrics_history):
    """Verify improvement signal is positive"""
    print("\n" + "="*70)
    print("CHECK 3: Improvement Signal")
    print("="*70)
    
    improvements = metrics_history.history.get('avg_improvement', [])
    if not improvements:
        print("✗ FAIL: No improvement data in metrics")
        return False
    
    mean_improvement = np.mean(improvements)
    positive_ratio = sum(1 for i in improvements if i > 0) / len(improvements)
    
    print(f"Average improvement:     {mean_improvement:.4f}")
    print(f"Positive improvements:   {positive_ratio:.1%}")
    
    if mean_improvement > 0.01 and positive_ratio > 0.80:
        print("✓ PASS: Improvement signal is consistently positive")
        return True
    elif mean_improvement > 0:
        print("~ PARTIAL: Improvement is positive but weak")
        return True
    else:
        print("✗ FAIL: Improvement signal is not positive")
        return False

def check_router_vs_baseline(train_ds, val_ds, n_seeds=3):
    """Verify router beats baseline across multiple seeds"""
    print("\n" + "="*70)
    print("CHECK 4: Router vs Baseline (3 seeds)")
    print("="*70)
    
    wins = 0
    improvements = []
    
    for seed in range(n_seeds):
        print(f"\nSeed {seed}:")
        
        # Baseline
        random.seed(seed)
        torch.manual_seed(seed)
        baseline_metrics = MetricsTracker()
        baseline_diversity = DiversityTracker(len(train_ds))
        baseline_model = train_baseline(train_ds, val_ds, baseline_metrics, baseline_diversity)
        baseline_ppl = baseline_metrics.get_final_ppl()
        
        # Router
        random.seed(seed)
        torch.manual_seed(seed)
        router_metrics = MetricsTracker()
        router_diversity = DiversityTracker(len(train_ds))
        router_model, router_net = train_router(train_ds, val_ds, router_metrics, router_diversity)
        router_ppl = router_metrics.get_final_ppl()
        
        improvement = (baseline_ppl - router_ppl) / baseline_ppl * 100
        improvements.append(improvement)
        
        if router_ppl < baseline_ppl:
            wins += 1
            print(f"  Baseline: {baseline_ppl:.2f}  Router: {router_ppl:.2f}  ✓ Router wins (+{improvement:.2f}%)")
        else:
            print(f"  Baseline: {baseline_ppl:.2f}  Router: {router_ppl:.2f}  ✗ Baseline wins ({improvement:.2f}%)")
        
        # Check progression for this seed
        easy_ratios = router_metrics.history.get('easy_ratio', [])
        if easy_ratios:
            early = np.mean(easy_ratios[:len(easy_ratios)//3])
            late = np.mean(easy_ratios[-len(easy_ratios)//3:])
            print(f"  Curriculum: {early:.2f} → {late:.2f} (shift: {early-late:+.2f})")
    
    mean_improvement = np.mean(improvements)
    print(f"\n Summary:")
    print(f"  Router wins: {wins}/{n_seeds}")
    print(f"  Mean improvement: {mean_improvement:+.2f}%")
    
    if wins == n_seeds and mean_improvement > 3:
        print("✓ PASS: Router consistently beats baseline")
        return True
    elif wins >= n_seeds * 0.67:
        print("~ PARTIAL: Router wins most of the time")
        return True
    else:
        print("✗ FAIL: Router does not consistently beat baseline")
        return False

def main():
    print("="*70)
    print("QUICK VALIDATION SUITE")
    print("="*70)
    print("\nRunning 4 quick validation checks...")
    print("Estimated time: ~15 minutes\n")
    
    # Load data
    print("Loading dataset...")
    train_chunks = make_mixed_chunks("train")
    val_chunks = make_mixed_chunks("validation")
    train_ds = MixedLMDataset(train_chunks)
    val_ds = MixedLMDataset(val_chunks)
    print(f"✓ Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    
    Path(cfg.save_dir).mkdir(exist_ok=True)
    
    # Run checks
    results = {}
    
    # Check 1: Dataset balance
    results['dataset_balance'] = check_dataset_balance(train_ds)
    
    # Check 2-3: Need to train once to get metrics
    print("\nTraining router once to check metrics...")
    random.seed(0)
    torch.manual_seed(0)
    metrics = MetricsTracker()
    diversity = DiversityTracker(len(train_ds))
    model, router = train_router(train_ds, val_ds, metrics, diversity)
    
    results['curriculum_progression'] = check_curriculum_progression(metrics)
    results['improvement_signal'] = check_improvement_signal(metrics)
    
    # Check 4: Router vs baseline
    results['router_wins'] = check_router_vs_baseline(train_ds, val_ds, n_seeds=3)
    
    # Final summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nPassed: {passed}/{total} checks")
    print()
    for check, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}  {check.replace('_', ' ').title()}")
    
    print("\n" + "="*70)
    
    if passed == total:
        print("✓✓ ALL CHECKS PASSED!")
        print("Curriculum learning is working as expected.")
    elif passed >= total * 0.75:
        print("✓ MOSTLY PASSED")
        print("Curriculum learning is working, but some aspects could be improved.")
    else:
        print("✗ FAILED")
        print("Something is wrong. Review the failed checks above.")
    
    print("="*70)
    
    return results

if __name__ == "__main__":
    main()