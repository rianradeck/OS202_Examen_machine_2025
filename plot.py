import matplotlib.pyplot as plt

# Data for n values and corresponding speedups
n_values = [2, 4, 8, 16]
speedup_values = [2.18, 3.52, 5.13, 4.01]

plt.figure(figsize=(8, 6))

plt.plot(n_values, speedup_values, marker='o', linestyle='-', color='b', label='Actual Speedup')

plt.plot(n_values, n_values, marker='x', linestyle='--', color='r', label='Ideal Speedup')

plt.xlabel('Number of Processes (n)')
plt.ylabel('Speedup')
plt.title('Speedup of movie_filter.py')

plt.ylim(2, 5.5)
plt.grid(True)
plt.xticks(n_values)
plt.legend()
plt.show()
