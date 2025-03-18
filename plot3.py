import matplotlib.pyplot as plt
import numpy as np

# Data from the table
n_values = [1, 2, 4, 8, 16]
metrics = {
    "Doubling Time": [3.602, 4.009, 3.594, 3.567, 3.735],
    "Convolution Time": [19.852, 10.991, 5.721, 3.340, 1.504],
    "Gather Time": [3.230, 4.567, 3.874, 3.768, 3.804],
    "Global Time": [29.721, 23.647, 17.261, 15.416, 14.712],
}

# Calculate speedup for each metric
speedup_metrics = {
    metric: [values[0] / value for value in values]
    for metric, values in metrics.items()
}

# Plot each metric's speedup
plt.figure(figsize=(10, 6))
for metric, speedup_values in speedup_metrics.items():
    plt.plot(n_values, speedup_values, marker='o', label=metric)

# Formatting the plot
plt.xlabel("Number of Processes (n)")
plt.ylabel("Speedup")
plt.title("Performance Speedup vs Number of Processes")
plt.xscale("log", base=2)  # Log scale for better visualization
plt.xticks(n_values, labels=[str(n) for n in n_values])
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Show plot
plt.show()
