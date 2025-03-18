import matplotlib.pyplot as plt
import numpy as np

# Data from the table
n_values = [1, 2, 4, 8, 16]
metrics = {
    "Doubling Time": [3.934, 3.973, 3.691, 3.853, 4.057],
    "Convolution Time": [33.523, 13.687, 7.320, 4.308, 1.864],
    "Gather Time": [3.331, 4.697, 3.533, 3.745, 3.980],
    "Global Time": [44.175, 26.147, 18.569, 16.679, 15.612],
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
