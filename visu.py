#!/usr/bin/env python3
"""
Visualization module for curriculum learning experiments.
Generates all must-have plots to validate router effectiveness.

Usage:
    python visualize_results.py --results_dir results_improved
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

class MetricsVisualizer:
    """Visualizer for curriculum learning metrics"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "plots"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load metrics
        self.baseline_metrics = self._load_metrics("baseline_metrics.json")
        self.router_metrics = self._load_metrics("router_metrics.json")
        
        print(f"✓ Loaded metrics from {results_dir}")
        print(f"  - Baseline: {len(self.baseline_metrics.get('step', []))} steps")
        print(f"  - Router: {len(self.router_metrics.get('step', []))} steps")
        print(f"  - Plots will be saved to: {self.output_dir}\n")
    
    def _load_metrics(self, filename: str) -> Dict:
        """Load metrics from JSON file"""
        path = self.results_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Metrics file not found: {path}")
        
        with open(path, 'r') as f:
            return json.load(f)
    
    def _moving_average(self, data: List[float], window: int = 50) -> np.ndarray:
        """Compute moving average for smoothing"""
        if len(data) < window:
            return np.array(data)
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    def plot_1_validation_ppl(self):
        """
        MUST-HAVE PLOT 1: Validation Perplexity over Epochs
        Shows main performance metric comparison
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data - validation metrics are logged at end of each epoch
        # So we need to align epochs with val_ppl values
        baseline_epochs_all = self.baseline_metrics.get('epoch', [])
        baseline_ppl_all = self.baseline_metrics.get('val_ppl', [])
        router_epochs_all = self.router_metrics.get('epoch', [])
        router_ppl_all = self.router_metrics.get('val_ppl', [])
        
        # Filter to only validation points (where val_ppl is actually present)
        def get_validation_points(epochs, ppls):
            """Extract validation points - val_ppl is only logged at epoch end"""
            if not epochs or not ppls or len(epochs) != len(ppls):
                return [], []
            
            # Find indices where val_ppl is actually logged (not None/placeholder)
            # In the JSON, val_ppl appears in the list at validation time
            # Group by epoch and keep the last (most recent) value per epoch
            epoch_to_ppl = {}
            for e, p in zip(epochs, ppls):
                if p is not None and p > 0:  # Valid perplexity value
                    epoch_to_ppl[e] = p
            
            if not epoch_to_ppl:
                return [], []
            
            sorted_epochs = sorted(epoch_to_ppl.keys())
            sorted_ppls = [epoch_to_ppl[e] for e in sorted_epochs]
            return sorted_epochs, sorted_ppls
        
        baseline_epochs, baseline_ppl = get_validation_points(baseline_epochs_all, baseline_ppl_all)
        router_epochs, router_ppl = get_validation_points(router_epochs_all, router_ppl_all)
        
        # Debug: print number of validation points found
        if baseline_epochs and router_epochs:
            print(f"  Found {len(baseline_epochs)} baseline validation points (epochs: {min(baseline_epochs)}-{max(baseline_epochs)})")
            print(f"  Found {len(router_epochs)} router validation points (epochs: {min(router_epochs)}-{max(router_epochs)})")
        
        # Plot
        ax.plot(baseline_epochs, baseline_ppl, 'o-', label='Baseline (Uniform)', 
                linewidth=2, markersize=8, color='#1f77b4')
        ax.plot(router_epochs, router_ppl, 's-', label='Router (Curriculum)', 
                linewidth=2, markersize=8, color='#ff7f0e')
        
        # Annotations
        if baseline_ppl and router_ppl:
            final_baseline = baseline_ppl[-1]
            final_router = router_ppl[-1]
            improvement = ((final_baseline - final_router) / final_baseline) * 100
            
            # Add text box with improvement
            textstr = f'Final PPL:\nBaseline: {final_baseline:.2f}\nRouter: {final_router:.2f}\nImprovement: {improvement:+.2f}%'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', bbox=props)
            
            # Highlight final points
            ax.scatter([baseline_epochs[-1]], [final_baseline], s=200, 
                      color='#1f77b4', zorder=5, edgecolors='black', linewidths=2)
            ax.scatter([router_epochs[-1]], [final_router], s=200, 
                      color='#ff7f0e', zorder=5, edgecolors='black', linewidths=2)
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Validation Perplexity', fontweight='bold')
        ax.set_title('Validation Perplexity Comparison: Router vs Baseline', 
                    fontweight='bold', fontsize=14)
        ax.legend(loc='upper right', framealpha=0.95)
        ax.grid(True, alpha=0.3)
        
        # Set y-axis to start from reasonable value
        if baseline_ppl and router_ppl:
            min_ppl = min(min(baseline_ppl), min(router_ppl))
            ax.set_ylim(bottom=min_ppl * 0.95)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '1_validation_ppl.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: 1_validation_ppl.png")
    
    def plot_2_difficulty_ratio(self):
        """
        MUST-HAVE PLOT 2: Easy vs Hard Ratio over Training
        Shows curriculum progression (easy → hard)
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract router data only (baseline doesn't track this)
        steps = self.router_metrics.get('step', [])
        easy_ratio = self.router_metrics.get('easy_ratio', [])
        hard_ratio = self.router_metrics.get('hard_ratio', [])
        epochs = self.router_metrics.get('epoch', [])
        
        if not easy_ratio or not hard_ratio:
            print("⚠ No difficulty ratio data found. Skipping plot 2.")
            return
        
        # Smooth for readability
        window = min(50, len(easy_ratio) // 10)
        if window > 1:
            easy_smooth = self._moving_average(easy_ratio, window)
            hard_smooth = self._moving_average(hard_ratio, window)
            steps_smooth = steps[window-1:]
        else:
            easy_smooth = easy_ratio
            hard_smooth = hard_ratio
            steps_smooth = steps
        
        # Plot both ratios
        ax.plot(steps_smooth, easy_smooth, '-', label='Easy Samples Ratio', 
                linewidth=2.5, color='#2ca02c', alpha=0.9)
        ax.plot(steps_smooth, hard_smooth, '-', label='Hard Samples Ratio', 
                linewidth=2.5, color='#d62728', alpha=0.9)
        
        # Add raw data as scatter (semi-transparent)
        ax.scatter(steps, easy_ratio, s=10, color='#2ca02c', alpha=0.2)
        ax.scatter(steps, hard_ratio, s=10, color='#d62728', alpha=0.2)
        
        # Mark training phases
        if steps:
            total_steps = steps[-1]
            ax.axvline(total_steps * 0.33, color='gray', linestyle='--', 
                      alpha=0.5, linewidth=1.5, label='Training phases')
            ax.axvline(total_steps * 0.67, color='gray', linestyle='--', 
                      alpha=0.5, linewidth=1.5)
            
            # Phase labels
            ax.text(total_steps * 0.17, 0.95, 'Early', ha='center', 
                   transform=ax.get_xaxis_transform(), fontsize=10, style='italic')
            ax.text(total_steps * 0.50, 0.95, 'Middle', ha='center', 
                   transform=ax.get_xaxis_transform(), fontsize=10, style='italic')
            ax.text(total_steps * 0.83, 0.95, 'Late', ha='center', 
                   transform=ax.get_xaxis_transform(), fontsize=10, style='italic')
        
        # Compute curriculum shift
        if len(easy_ratio) > 100:
            early_easy = np.mean(easy_ratio[:len(easy_ratio)//3])
            late_easy = np.mean(easy_ratio[-len(easy_ratio)//3:])
            shift = early_easy - late_easy
            
            textstr = f'Curriculum Shift:\nEarly Easy: {early_easy:.2f}\nLate Easy: {late_easy:.2f}\nShift: {shift:.2f}'
            if shift > 0.05:
                textstr += '\n✓ Clear easy→hard progression'
            else:
                textstr += '\n✗ Weak curriculum signal'
            
            props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
            ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='bottom', horizontalalignment='right', bbox=props)
        
        ax.set_xlabel('Training Step', fontweight='bold')
        ax.set_ylabel('Selection Ratio', fontweight='bold')
        ax.set_title('Curriculum Progression: Easy vs Hard Sample Selection', 
                    fontweight='bold', fontsize=14)
        ax.legend(loc='upper left', framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '2_difficulty_ratio.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: 2_difficulty_ratio.png")
    
    def plot_3_loss_improvement(self):
        """
        MUST-HAVE PLOT 3: Average Improvement per Step
        Shows router is selecting samples with good learning signal
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        steps = self.router_metrics.get('step', [])
        improvements = self.router_metrics.get('avg_improvement', [])
        
        if not improvements:
            print("⚠ No improvement data found. Skipping plot 3.")
            return
        
        # Smooth
        window = min(50, len(improvements) // 10)
        if window > 1:
            improvements_smooth = self._moving_average(improvements, window)
            steps_smooth = steps[window-1:]
        else:
            improvements_smooth = improvements
            steps_smooth = steps
        
        # Plot
        ax.plot(steps_smooth, improvements_smooth, '-', linewidth=2.5, 
               color='#9467bd', label='Avg Improvement (smoothed)')
        ax.scatter(steps, improvements, s=10, color='#9467bd', alpha=0.15, 
                  label='Raw values')
        
        # Zero line
        ax.axhline(0, color='red', linestyle='--', linewidth=1.5, 
                  alpha=0.7, label='Zero (no learning)')
        
        # Statistics
        mean_improvement = np.mean(improvements)
        positive_pct = (np.array(improvements) > 0).mean() * 100
        
        textstr = f'Statistics:\nMean: {mean_improvement:.4f}\nPositive: {positive_pct:.1f}%'
        if mean_improvement > 0:
            textstr += '\n✓ Positive learning signal'
        else:
            textstr += '\n✗ Poor learning signal'
        
        props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', horizontalalignment='right', bbox=props)
        
        ax.set_xlabel('Training Step', fontweight='bold')
        ax.set_ylabel('Average Loss Improvement', fontweight='bold')
        ax.set_title('Loss Improvement per Step (Router)', 
                    fontweight='bold', fontsize=14)
        ax.legend(loc='upper left', framealpha=0.95)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '3_loss_improvement.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: 3_loss_improvement.png")
    
    def plot_4_coverage_entropy(self):
        """
        MUST-HAVE PLOT 4: Coverage and Entropy over Training
        Shows sample diversity comparison
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # === Plot 4a: Coverage ===
        baseline_steps = self.baseline_metrics.get('step', [])
        baseline_coverage = self.baseline_metrics.get('coverage', [])
        router_steps = self.router_metrics.get('step', [])
        router_coverage = self.router_metrics.get('coverage', [])
        
        if baseline_coverage and router_coverage:
            # Smooth
            window = min(50, len(baseline_coverage) // 10)
            if window > 1:
                baseline_cov_smooth = self._moving_average(baseline_coverage, window)
                router_cov_smooth = self._moving_average(router_coverage, window)
                baseline_steps_smooth = baseline_steps[window-1:]
                router_steps_smooth = router_steps[window-1:]
            else:
                baseline_cov_smooth = baseline_coverage
                router_cov_smooth = router_coverage
                baseline_steps_smooth = baseline_steps
                router_steps_smooth = router_steps
            
            ax1.plot(baseline_steps_smooth, baseline_cov_smooth, '-', 
                    label='Baseline', linewidth=2.5, color='#1f77b4')
            ax1.plot(router_steps_smooth, router_cov_smooth, '-', 
                    label='Router', linewidth=2.5, color='#ff7f0e')
            
            # Final values
            final_baseline_cov = baseline_coverage[-1] if baseline_coverage else 0
            final_router_cov = router_coverage[-1] if router_coverage else 0
            
            textstr = f'Final Coverage:\nBaseline: {final_baseline_cov:.3f}\nRouter: {final_router_cov:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax1.text(0.98, 0.02, textstr, transform=ax1.transAxes, fontsize=11,
                    verticalalignment='bottom', horizontalalignment='right', bbox=props)
        
        ax1.set_xlabel('Training Step', fontweight='bold')
        ax1.set_ylabel('Dataset Coverage', fontweight='bold')
        ax1.set_title('Dataset Coverage: % of Samples Seen', 
                     fontweight='bold', fontsize=13)
        ax1.legend(loc='upper left', framealpha=0.95)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # === Plot 4b: Entropy ===
        baseline_entropy = self.baseline_metrics.get('entropy', [])
        router_entropy = self.router_metrics.get('entropy', [])
        
        if baseline_entropy and router_entropy:
            # Smooth
            if window > 1:
                baseline_ent_smooth = self._moving_average(baseline_entropy, window)
                router_ent_smooth = self._moving_average(router_entropy, window)
            else:
                baseline_ent_smooth = baseline_entropy
                router_ent_smooth = router_entropy
            
            ax2.plot(baseline_steps_smooth, baseline_ent_smooth, '-', 
                    label='Baseline', linewidth=2.5, color='#1f77b4')
            ax2.plot(router_steps_smooth, router_ent_smooth, '-', 
                    label='Router', linewidth=2.5, color='#ff7f0e')
            
            # Final values
            final_baseline_ent = baseline_entropy[-1] if baseline_entropy else 0
            final_router_ent = router_entropy[-1] if router_entropy else 0
            
            textstr = f'Final Entropy:\nBaseline: {final_baseline_ent:.3f}\nRouter: {final_router_ent:.3f}'
            textstr += '\n(Lower = more focused)'
            props = dict(boxstyle='round', facecolor='lightcyan', alpha=0.8)
            ax2.text(0.98, 0.98, textstr, transform=ax2.transAxes, fontsize=11,
                    verticalalignment='top', horizontalalignment='right', bbox=props)
        
        ax2.set_xlabel('Training Step', fontweight='bold')
        ax2.set_ylabel('Selection Entropy', fontweight='bold')
        ax2.set_title('Selection Entropy: Sample Diversity Measure', 
                     fontweight='bold', fontsize=13)
        ax2.legend(loc='upper right', framealpha=0.95)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '4_coverage_entropy.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: 4_coverage_entropy.png")
    
    def plot_5_final_performance(self):
        """
        MUST-HAVE PLOT 5: Final Performance Bar Chart
        Clear comparison of final validation perplexity
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Get final PPL
        baseline_ppl_all = self.baseline_metrics.get('val_ppl', [])
        router_ppl_all = self.router_metrics.get('val_ppl', [])
        
        if not baseline_ppl_all or not router_ppl_all:
            print("⚠ No validation PPL data found. Skipping plot 5.")
            return
        
        baseline_ppl = baseline_ppl_all[-1]
        router_ppl = router_ppl_all[-1]
        improvement = ((baseline_ppl - router_ppl) / baseline_ppl) * 100
        
        # Bar chart
        methods = ['Baseline\n(Uniform)', 'Router\n(Curriculum)']
        ppls = [baseline_ppl, router_ppl]
        colors = ['#1f77b4', '#ff7f0e']
        
        bars = ax.bar(methods, ppls, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add value labels on bars
        for i, (bar, ppl) in enumerate(zip(bars, ppls)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{ppl:.2f}',
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Add improvement annotation
        if router_ppl < baseline_ppl:
            # Arrow showing improvement
            ax.annotate('', xy=(1, router_ppl), xytext=(1, baseline_ppl),
                       arrowprops=dict(arrowstyle='<->', color='green', lw=3))
            
            mid_point = (baseline_ppl + router_ppl) / 2
            ax.text(1.15, mid_point, f'{improvement:+.2f}%\nimprovement',
                   ha='left', va='center', fontsize=12, fontweight='bold',
                   color='green',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        ax.set_ylabel('Validation Perplexity', fontweight='bold', fontsize=13)
        ax.set_title('Final Performance Comparison', fontweight='bold', fontsize=15)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis to start from reasonable value
        min_ppl = min(ppls)
        ax.set_ylim(bottom=min_ppl * 0.85)
        
        # Add verdict
        if improvement > 5:
            verdict = "✓✓ Significant improvement"
            verdict_color = 'darkgreen'
        elif improvement > 2:
            verdict = "✓ Meaningful improvement"
            verdict_color = 'green'
        elif improvement > 0:
            verdict = "~ Modest improvement"
            verdict_color = 'orange'
        else:
            verdict = "✗ No improvement"
            verdict_color = 'red'
        
        ax.text(0.5, 0.98, verdict, transform=ax.transAxes,
               ha='center', va='top', fontsize=13, fontweight='bold',
               color=verdict_color,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                        edgecolor=verdict_color, linewidth=2))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '5_final_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: 5_final_performance.png")
    
    def generate_all_must_have_plots(self):
        """Generate all must-have plots"""
        print("\n" + "="*70)
        print("GENERATING MUST-HAVE PLOTS")
        print("="*70 + "\n")
        
        self.plot_1_validation_ppl()
        self.plot_2_difficulty_ratio()
        self.plot_3_loss_improvement()
        self.plot_4_coverage_entropy()
        self.plot_5_final_performance()
        
        print("\n" + "="*70)
        print(f"✓ All must-have plots generated successfully!")
        print(f"✓ Saved to: {self.output_dir}")
        print("="*70 + "\n")
    
    def generate_summary_report(self):
        """Generate a text summary report"""
        report_path = self.output_dir / "summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CURRICULUM LEARNING EXPERIMENT - SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")
            
            # 1. Final Performance
            baseline_ppl_all = self.baseline_metrics.get('val_ppl', [])
            router_ppl_all = self.router_metrics.get('val_ppl', [])
            
            if baseline_ppl_all and router_ppl_all:
                baseline_ppl = baseline_ppl_all[-1]
                router_ppl = router_ppl_all[-1]
                improvement = ((baseline_ppl - router_ppl) / baseline_ppl) * 100
                
                f.write("1. FINAL PERFORMANCE\n")
                f.write("-" * 70 + "\n")
                f.write(f"Baseline (Uniform):      {baseline_ppl:.4f}\n")
                f.write(f"Router (Curriculum):     {router_ppl:.4f}\n")
                f.write(f"Improvement:             {improvement:+.2f}%\n\n")
            
            # 2. Curriculum Progression
            easy_ratio = self.router_metrics.get('easy_ratio', [])
            if len(easy_ratio) > 100:
                early_easy = np.mean(easy_ratio[:len(easy_ratio)//3])
                late_easy = np.mean(easy_ratio[-len(easy_ratio)//3:])
                shift = early_easy - late_easy
                
                f.write("2. CURRICULUM PROGRESSION\n")
                f.write("-" * 70 + "\n")
                f.write(f"Early training (easy ratio):  {early_easy:.3f}\n")
                f.write(f"Late training (easy ratio):   {late_easy:.3f}\n")
                f.write(f"Shift toward hard samples:    {shift:.3f}\n")
                f.write(f"Clear progression:            {'Yes' if shift > 0.05 else 'No'}\n\n")
            
            # 3. Learning Signal
            improvements = self.router_metrics.get('avg_improvement', [])
            if improvements:
                mean_imp = np.mean(improvements)
                positive_pct = (np.array(improvements) > 0).mean() * 100
                
                f.write("3. LEARNING SIGNAL (Router)\n")
                f.write("-" * 70 + "\n")
                f.write(f"Mean improvement:        {mean_imp:.6f}\n")
                f.write(f"Positive improvement:    {positive_pct:.1f}%\n\n")
            
            # 4. Sample Diversity
            baseline_coverage = self.baseline_metrics.get('coverage', [])
            router_coverage = self.router_metrics.get('coverage', [])
            
            if baseline_coverage and router_coverage:
                f.write("4. SAMPLE DIVERSITY\n")
                f.write("-" * 70 + "\n")
                f.write(f"Baseline coverage:       {baseline_coverage[-1]:.3f}\n")
                f.write(f"Router coverage:         {router_coverage[-1]:.3f}\n\n")
            
            # 5. Verdict
            f.write("5. VERDICT\n")
            f.write("-" * 70 + "\n")
            if baseline_ppl_all and router_ppl_all:
                if improvement > 5:
                    f.write("✓✓ SIGNIFICANT IMPROVEMENT (>5%)\n")
                    f.write("    Router provides clear benefits over baseline.\n")
                elif improvement > 2:
                    f.write("✓ MEANINGFUL IMPROVEMENT (2-5%)\n")
                    f.write("   Router shows consistent benefits.\n")
                elif improvement > 0:
                    f.write("~ MODEST IMPROVEMENT (<2%)\n")
                    f.write("  Router shows slight benefits.\n")
                else:
                    f.write("✗ NO IMPROVEMENT\n")
                    f.write("  Router does not outperform baseline.\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"✓ Generated: summary_report.txt")
        return report_path


def main():
    parser = argparse.ArgumentParser(description='Visualize curriculum learning results')
    parser.add_argument('--results_dir', type=str, default='results_improved',
                       help='Directory containing metrics JSON files')
    args = parser.parse_args()
    
    # Check if results directory exists
    if not Path(args.results_dir).exists():
        print(f"✗ Error: Results directory not found: {args.results_dir}")
        print(f"  Please run the training script first to generate metrics.")
        return
    
    # Create visualizer
    visualizer = MetricsVisualizer(args.results_dir)
    
    # Generate all plots
    visualizer.generate_all_must_have_plots()
    
    # Generate summary report
    visualizer.generate_summary_report()
    
    print("\n📊 Visualization complete!")
    print(f"📁 All outputs saved to: {visualizer.output_dir}/")
    print("\nGenerated files:")
    print("  - 1_validation_ppl.png")
    print("  - 2_difficulty_ratio.png")
    print("  - 3_loss_improvement.png")
    print("  - 4_coverage_entropy.png")
    print("  - 5_final_performance.png")
    print("  - summary_report.txt")


if __name__ == "__main__":
    main()