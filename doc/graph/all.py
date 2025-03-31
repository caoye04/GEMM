import matplotlib.pyplot as plt
import numpy as np

# Data preparation
matrix_sizes = [31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129, 159, 160, 161,
                191, 192, 193, 223, 224, 225, 255, 256, 257, 287, 288, 289, 319,
                320, 321, 384, 392, 452, 472, 496, 511, 512, 513, 528, 575, 576,
                577, 640, 641, 767, 768, 769, 895, 896, 897, 1023, 1024, 1025]

# 第一个图: 不同步骤在各矩阵大小上的性能
step0 = [3.02, 2.99, 2.94, 2.92, 2.92, 2.91, 2.80, 2.81, 2.81, 2.75, 2.75, 2.76, 2.78, 2.77, 2.77,
         2.81, 2.80, 2.80, 2.78, 2.79, 2.80, 2.75, 2.75, 2.76, 2.76, 2.75, 2.75, 2.78, 2.77, 2.77,
         2.76, 2.78, 2.75, 2.77, 2.76, 2.69, 2.54, 2.69, 2.76, 2.75, 2.75, 2.75, 2.76, 2.76, 2.75,
         2.74, 2.75, 2.76, 2.75, 2.75, 2.63, 2.46, 2.63]

step1 = [4.12, 6.88, 6.73, 5.08, 5.03, 4.97, 5.18, 5.16, 5.16, 5.24, 5.18, 5.18, 5.55, 5.52, 5.49,
         5.15, 5.15, 5.12, 5.35, 5.35, 5.33, 5.30, 5.23, 5.29, 5.39, 5.34, 5.34, 5.49, 5.50, 5.48,
         5.32, 5.31, 5.35, 5.34, 5.34, 5.27, 4.00, 5.26, 5.41, 5.33, 5.34, 5.32, 5.31, 5.31, 5.27,
         5.19, 5.26, 5.39, 5.39, 5.39, 4.89, 2.74, 4.88]

step2 = [8.66, 14.20, 8.86, 8.64, 8.41, 11.80, 8.69, 8.91, 8.90, 9.80, 9.97, 9.54, 10.70, 11.10, 10.90,
         10.40, 10.50, 10.40, 10.10, 10.30, 10.20, 9.85, 9.44, 9.92, 11.00, 10.70, 10.60, 11.40, 10.80, 10.80,
         10.20, 10.50, 10.70, 10.60, 10.60, 10.30, 9.78, 10.40, 11.00, 10.80, 10.80, 10.80, 10.70, 10.80, 10.80,
         10.60, 10.90, 10.90, 10.90, 11.00, 10.60, 9.15, 10.70]

step3 = [9.23, 16.3, 9.86, 15.6, 23.9, 16.2, 18.4, 23.6, 18.6, 18.8, 18.4, 19.1, 20.1, 20, 20.6,
         19, 18.4, 18.5, 19.6, 19.3, 19.4, 19.3, 19.6, 19.3, 20.2, 20.4, 20.4, 19.6, 19.5, 19.3,
         20.1, 20, 19.9, 20.4, 20.3, 19.4, 20, 19.5, 20.5, 20.1, 20, 20.2, 19.9, 20.3, 19.3,
         20.2, 19.4, 20.3, 19.8, 20.4, 19.1, 16.1, 19.2]

step4 = [8.66, 14.6, 8.9, 16.7, 18.9, 16.8, 23, 21.9, 22.2, 19.2, 18.6, 18.9, 20, 19, 19.5,
         20.2, 19.4, 20, 20.4, 19.6, 18, 18.9, 17.5, 17.6, 16.8, 18.2, 17, 18.8, 18.6, 18.8,
         18.1, 17.6, 18.5, 18.3, 18.6, 18.4, 17.4, 17.5, 18.8, 18.6, 18.2, 18.6, 18.8, 17.9, 18.7,
         17.6, 18.1, 18.4, 18, 18, 18.3, 15.7, 17.9]

step5 = [8.65, 77.7, 18.7, 16.7, 80.2, 30.7, 23.1, 75.3, 38.3, 19.2, 70.3, 43.1, 20.6, 75.5, 48,
         23.3, 73.7, 50.7, 25.5, 75, 53.5, 27.1, 63.4, 51.6, 29.6, 74.7, 57.1, 31.3, 72.7, 57.8,
         68, 47.7, 52.4, 49.7, 73.3, 36.4, 54.4, 51.9, 73, 41.1, 69.9, 62.5, 66.2, 61.8, 43.5,
         61.4, 58.4, 47.4, 65.1, 63.1, 37.4, 42.6, 43.6]

step6 = [57.8, 126, 36.6, 81.6, 140, 69, 87.7, 137, 69.1, 90.9, 111, 91.5, 106, 122, 88.2,
         101, 114, 96.8, 98.6, 115, 87.1, 98.1, 113, 99.5, 103, 114, 93.2, 99.8, 114, 102,
         112, 104, 106, 101, 99.2, 93.1, 111, 105, 105, 98.4, 110, 105, 110, 105, 98.9,
         108, 106, 100, 108, 106, 96.3, 62.6, 103]

# Performance metrics for different steps
max_perf = [3.02, 6.88, 14.20, 23.9, 23, 80.2, 140]
avg_perf = [2.726, 5.12, 10.52, 18.78, 17.6, 52.64, 97.39]

# Set up color scheme
colors = {
    'step0': '#8B4513',  # Brown
    'step1': '#228B22',  # ForestGreen
    'step2': '#1E90FF',  # DodgerBlue
    'step3': '#FF8C00',  # DarkOrange
    'step4': '#9932CC',  # DarkOrchid
    'step5': '#FF1493',  # DeepPink
    'step6': '#000000',  # Black
}

# Figure 1: Matrix Size vs Performance
plt.figure(figsize=(14, 8))

# Plot lines connecting data points
plt.plot(matrix_sizes, step0, '-', color=colors['step0'], linewidth=1.5, label='Step 0: Base Implementation')
plt.plot(matrix_sizes, step1, '-', color=colors['step1'], linewidth=1.5, label='Step 1: Compilation Optimization')
plt.plot(matrix_sizes, step2, '-', color=colors['step2'], linewidth=1.5, label='Step 2: Loop Refactorization')
plt.plot(matrix_sizes, step3, '-', color=colors['step3'], linewidth=1.5, label='Step 3: Single-layer Blocking')
plt.plot(matrix_sizes, step4, '-', color=colors['step4'], linewidth=1.5, label='Step 4: Two-level Blocking')
plt.plot(matrix_sizes, step5, '-', color=colors['step5'], linewidth=1.5, label='Step 5: Vectorization')
plt.plot(matrix_sizes, step6, '-', color=colors['step6'], linewidth=1.5, label='Step 6: Multi-kernel Design')

# Add title and labels
plt.title('Performance Across Different Matrix Sizes and Optimization Steps', fontsize=16)
plt.xlabel('Matrix Size', fontsize=14)
plt.ylabel('Performance (Gflop/s)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=10)

# Highlight key sizes
key_sizes = [32, 64, 96, 128, 256, 512, 1024]
plt.xticks(key_sizes, [str(size) for size in key_sizes])

# Set y-axis range
plt.ylim(0, 150)

plt.tight_layout()
plt.savefig('optimization_by_matrix_size.png', dpi=300)

# Figure 2: Performance metrics across steps
plt.figure(figsize=(10, 6))

# Create x-axis for steps
steps = list(range(7))
step_names = ['Step 0', 'Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Step 6']

# Plot max and average performance
plt.plot(steps, max_perf, 'o-', color='#FF4500', linewidth=2, markersize=8, label='Maximum Performance')
plt.plot(steps, avg_perf, 's-', color='#4169E1', linewidth=2, markersize=8, label='Weighted Average Performance')

# Add data labels
for i, (max_val, avg_val) in enumerate(zip(max_perf, avg_perf)):
    plt.text(i, max_val + 5, f'{max_val:.1f}', ha='center', fontsize=9)
    plt.text(i, avg_val - 5, f'{avg_val:.1f}', ha='center', fontsize=9)

# Add title and labels
plt.title('Performance Metrics Across Optimization Steps', fontsize=16)
plt.xlabel('Optimization Step', fontsize=14)
plt.ylabel('Performance (Gflop/s)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)

# Set x-ticks to step names
plt.xticks(steps, step_names)

# Set y-axis range
plt.ylim(0, 160)

plt.tight_layout()
plt.savefig('performance_by_step.png', dpi=300)

plt.show()