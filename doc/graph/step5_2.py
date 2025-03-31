import matplotlib.pyplot as plt
import numpy as np

# Data preparation
methods = ['64/16 No Unroll', '256/16 No Unroll', '64/16 C-Reg Unroll', '256/16 C-Reg Unroll', '64/16 Full Unroll', '256/16 Full Unroll']
avg_perf = [50.37, 48.59, 46.08, 46.07, 48.83, 49.17]
max_perf = [80.20, 78.90, 68.20, 68.40, 72.70, 72.20]
min_perf = [8.63, 8.64, 8.03, 8.65, 8.65, 8.65]
pow2_perf = [60.89, 59.78, 56.05, 56.77, 58.74, 59.12]

# Set up bar chart
fig, ax = plt.subplots(figsize=(12, 8))

# Set bar width and position
width = 0.2
x = np.arange(len(methods))

# Draw multiple bar groups
ax.bar(x - width*1.5, avg_perf, width, label='Weighted Average', color='#4472C4')
ax.bar(x - width/2, max_perf, width, label='Maximum', color='#70AD47')
ax.bar(x + width/2, min_perf, width, label='Minimum', color='#FFC000')
ax.bar(x + width*1.5, pow2_perf, width, label='Power-of-2 Average', color='#ED7D31')

# Add title and labels
ax.set_title('Performance Comparison of Vectorization and Loop Unrolling Methods', fontsize=16)
ax.set_ylabel('Performance (Gflops)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15, ha='right')
ax.legend()

# Add value labels
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

# Display chart
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('step5.2.png', dpi=300)
plt.show()