import matplotlib.pyplot as plt
import numpy as np

# Data preparation
matrix_sizes = [31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129, 159, 160, 161,
                191, 192, 193, 223, 224, 225, 255, 256, 257, 287, 288, 289, 319,
                320, 321, 384, 392, 452, 472, 496, 511, 512, 513, 528, 575, 576,
                577, 640, 641, 767, 768, 769, 895, 896, 897, 1023, 1024, 1025]

# Performance data for different implementations
base_version = [57.8, 126, 36.6, 81.6, 140, 58.5, 87.7, 136, 69.1, 90.9, 111, 90, 106, 119, 88.2,
                101, 114, 87.3, 98.6, 114, 87.1, 98.1, 112, 98.3, 103, 107, 93.2, 99.8, 105, 91,
                103, 99.9, 98.6, 95.8, 96.4, 93.1, 92.8, 94.3, 97.3, 98.4, 101, 93.8, 102, 98.7,
                98.9, 104, 101, 100, 104, 102, 96.3, 62.3, 100]

add_96x4 = [34.9, 121, 20.3, 44.9, 138, 51.1, 63.9, 137, 45.3, 55.8, 86.3, 72.9, 67.6, 103, 54,
            59.7, 90.7, 68.3, 71.3, 92.1, 60.1, 64.2, 87.5, 73.1, 72.8, 88.3, 63.6, 65.2, 82.6, 71.6,
            79.2, 75, 85, 72.8, 80.5, 67.8, 69.7, 72.3, 75.7, 69.1, 77.8, 73.7, 77.9, 74.4, 72,
            78.1, 75.8, 73, 77.9, 76.4, 71.9, 41, 74.5]

add_16x1 = [36.6, 121, 20.7, 44.6, 138, 51.2, 64.1, 135, 45.4, 55.7, 109, 75.3, 77.1, 122, 59.7,
            64.6, 114, 84.2, 82.8, 114, 67.8, 71.7, 112, 88.6, 84.7, 107, 71.5, 72.8, 106, 87.7,
            104, 96.3, 98.1, 92.2, 90.5, 79.6, 100, 90.5, 96.7, 81.6, 102, 92.9, 103, 94.8, 87.6,
            104, 97.4, 89.9, 104, 99.0, 89.7, 61.7, 97.4]

add_32x1 = [36.9, 121, 20.7, 44.7, 138, 51.1, 63.8, 135, 45.4, 55.6, 109, 75.4, 77.0, 122, 59.7,
            64.6, 114, 84.2, 83.0, 115, 67.9, 71.7, 112, 88.4, 84.7, 107, 71.5, 72.9, 106, 87.7,
            104, 96.3, 98.3, 92.5, 90.4, 79.2, 98.7, 89.8, 95.4, 80.1, 99.0, 90.4, 99.2, 91.2, 84.9,
            99.8, 93.9, 87.6, 101, 96.2, 88.0, 60.9, 95.3]

add_64x1 = [35.9, 121, 20.9, 44.8, 138, 69.0, 63.8, 135, 45.3, 55.7, 109, 91.5, 77.2, 122, 59.7,
            64.7, 114, 96.8, 83.1, 115, 67.8, 72.0, 113, 99.5, 87.6, 114, 74.1, 75.6, 114, 102,
            112, 104, 106, 101, 99.2, 84.7, 111, 105, 105, 85.7, 110, 105, 110, 105, 90.7, 108,
            106, 92.6, 108, 106, 91.8, 62.6, 103]

without_48x8 = [35.0, 119, 20.1, 44.7, 138, 50.8, 63.4, 137, 45.4, 55.8, 86.5, 73.9, 67.7, 103, 54,
               59.7, 91.2, 68.4, 71.5, 91.8, 60.4, 64.6, 88, 73.6, 75.1, 91.1, 65.4, 67.6, 87.6, 76.2,
               87.2, 81.9, 94.5, 79.8, 78.9, 71.8, 83.8, 78.1, 81.1, 72.5, 84.1, 79, 83.1, 78.8, 74.5,
               81.5, 79, 75.2, 80.8, 78.9, 73.5, 41.1, 76.4]

# Use soft colors
colors = {
    'base': '#000000',       # Black
    'add_96x4': '#70AD47',   # Green
    'add_16x1': '#FFC000',   # Gold
    'add_32x1': '#ED7D31',   # Orange
    'add_64x1': '#5B9BD5',   # Light blue
    'without_48x8': '#A5A5A5' # Gray
}

# Set up chart
plt.figure(figsize=(14, 8))
plt.plot(matrix_sizes, base_version, 'o-', label='Base Version (16-16, 32-8, 48-8, 64-4)', 
         color=colors['base'], markersize=3, alpha=0.8)
plt.plot(matrix_sizes, add_96x4, 'o-', label='Add 96×4', 
         color=colors['add_96x4'], markersize=3, alpha=0.8)
plt.plot(matrix_sizes, add_16x1, 'o-', label='Add 16×1', 
         color=colors['add_16x1'], markersize=3, alpha=0.8)
plt.plot(matrix_sizes, add_32x1, 'o-', label='Add 32×1', 
         color=colors['add_32x1'], markersize=3, alpha=0.8)
plt.plot(matrix_sizes, add_64x1, 'o-', label='Add 64×1', 
         color=colors['add_64x1'], markersize=3, alpha=0.8)
plt.plot(matrix_sizes, without_48x8, 'o-', label='Without 48×8 (16-16, 32-8, 64-4)', 
         color=colors['without_48x8'], markersize=3, alpha=0.8)

# Add title and labels
plt.title('Performance Comparison of Matrix Multiplication Implementations', fontsize=16)
plt.xlabel('Matrix Size', fontsize=14)
plt.ylabel('Performance (Gflop/s)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=10)

# Adjust x-axis ticks
key_sizes = [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
plt.xticks(key_sizes, [str(size) for size in key_sizes])

# Set y-axis range
y_min = 0
y_max = 145  # Slightly above maximum performance value
plt.ylim(y_min, y_max)

# Display chart
plt.tight_layout()
plt.savefig('step6.png', dpi=300)
plt.show()