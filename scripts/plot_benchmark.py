import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from pathlib import Path

def load_results(json_path):
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Plot Benchmark Comparisons (Baseline vs MoE)")
    parser.add_argument("--baseline", type=str, required=True, help="Path to baseline results JSON")
    parser.add_argument("--moe", type=str, required=True, help="Path to MoE results JSON")
    parser.add_argument("--output", type=str, default="benchmark_comparison.png", help="Output image filename")
    args = parser.parse_args()

    print(f"Loading baseline: {args.baseline}")
    print(f"Loading MoE:      {args.moe}")

    base_data = load_results(args.baseline)
    moe_data = load_results(args.moe)

    # Mapping corresponds to the JSON structure from bench.py
    metrics_map = [
        ("WER (%)", "wer", "test-clean", "wer_percent"),
        ("Latency P50 (ms)", "latency", "test-clean", "p50_ms"),
        ("Throughput (utt/s)", "throughput", "test-clean", "throughput_utt_per_sec")
    ]

    labels = []
    base_vals = []
    moe_vals = []

    for label, cat, split, key in metrics_map:
        try:
            if cat == "wer":
                b_val = base_data.get("wer", {}).get(split, {}).get(key, 0)
                m_val = moe_data.get("wer", {}).get(split, {}).get(key, 0)
            else:
                b_val = base_data.get(cat, {}).get(split, {}).get(key, 0)
                m_val = moe_data.get(cat, {}).get(split, {}).get(key, 0)
        except AttributeError:
            print(f"Warning: Data missing for {label}")
            b_val, m_val = 0, 0

        labels.append(label)
        base_vals.append(b_val)
        moe_vals.append(m_val)

    # --- Start Plotting ---
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, base_vals, width, label='Dense Baseline', color='gray', alpha=0.7)
    rects2 = ax.bar(x + width/2, moe_vals, width, label='MoE (Ours)', color='#1f77b4')

    ax.set_ylabel('Scores')
    ax.set_title('Performance Comparison: Dense vs MoE (OWSM-CTC)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"\n[Success] Comparison plot saved to: {args.output}")

if __name__ == "__main__":
    main()