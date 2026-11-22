
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Create output folder
Path("outputs").mkdir(exist_ok=True)

# Read model comparison results
data = pd.read_csv("outputs/metrics_summary.csv")

# Metrics to compare
metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1_score"
]

# Plot 1: Performance comparison
data.set_index("model")[metrics].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Champion vs Challenger - Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.legend(title="Metrics")
plt.tight_layout()

plt.savefig("outputs/performance_comparison.png")
plt.show()

# Plot 2: Inference latency comparison
data.plot(
    x="model",
    y="avg_latency_ms",
    kind="bar",
    legend=False,
    figsize=(8, 5)
)

plt.title("Model Inference Latency Comparison")
plt.xlabel("Model")
plt.ylabel("Average Latency (ms)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("outputs/latency_comparison.png")
plt.show()

print("Both comparison graphs saved successfully!")
