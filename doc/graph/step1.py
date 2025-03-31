import matplotlib.pyplot as plt
import numpy as np

# 数据准备
matrix_sizes = [31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129, 159, 160, 161, 
                191, 192, 193, 223, 224, 225, 255, 256, 257, 287, 288, 289, 319, 
                320, 321, 384, 392, 452, 472, 496, 511, 512, 513, 528, 575, 576, 
                577, 640, 641, 767, 768, 769, 895, 896, 897, 1023, 1024, 1025]

blocked = [3.02, 2.99, 2.94, 2.92, 2.92, 2.91, 2.80, 2.81, 2.81, 2.75, 2.75, 2.76, 
           2.78, 2.77, 2.77, 2.81, 2.80, 2.80, 2.78, 2.79, 2.80, 2.75, 2.75, 2.76, 
           2.76, 2.75, 2.75, 2.78, 2.77, 2.77, 2.76, 2.78, 2.75, 2.77, 2.76, 2.69, 
           2.54, 2.69, 2.76, 2.75, 2.75, 2.75, 2.76, 2.76, 2.75, 2.74, 2.75, 2.76, 
           2.75, 2.75, 2.63, 2.46, 2.63]

o3_optimization = [4.12, 4.21, 4.11, 4.17, 4.16, 4.19, 4.15, 4.17, 4.16, 4.18, 4.14, 
                   4.19, 4.26, 4.26, 4.26, 4.23, 4.24, 4.24, 4.23, 4.23, 4.23, 4.17, 
                   4.14, 4.16, 4.26, 4.23, 4.22, 4.25, 4.23, 4.24, 4.17, 4.22, 4.21, 
                   4.21, 4.20, 3.97, 3.20, 3.98, 4.22, 4.19, 4.18, 4.19, 4.16, 4.21, 
                   4.16, 4.12, 4.16, 4.22, 4.17, 4.21, 3.83, 2.64, 3.83]

fast_optimization = [3.96, 6.88, 6.73, 5.08, 5.03, 4.97, 5.18, 5.16, 5.16, 5.24, 5.18, 
                     5.18, 5.55, 5.52, 5.49, 5.15, 5.15, 5.12, 5.35, 5.35, 5.33, 5.30, 
                     5.23, 5.29, 5.39, 5.34, 5.34, 5.49, 5.50, 5.48, 5.32, 5.31, 5.35, 
                     5.34, 5.34, 5.27, 4.00, 5.26, 5.41, 5.33, 5.34, 5.32, 5.31, 5.31, 
                     5.27, 5.19, 5.26, 5.39, 5.39, 5.39, 4.89, 2.74, 4.88]

# 使用更温和的颜色
colors = {
    'blocked': '#5b9bd5',      # 温和的蓝色
    'o3': '#70ad47',           # 温和的绿色
    'fast': '#ed7d31'          # 温和的橙色
}

# 设置图表
plt.figure(figsize=(14, 8))
plt.plot(matrix_sizes, blocked, 'o-', label='Blocked', color=colors['blocked'], markersize=3, alpha=0.8)
plt.plot(matrix_sizes, o3_optimization, 'o-', label='O3 Optimization', color=colors['o3'], markersize=3, alpha=0.8)
plt.plot(matrix_sizes, fast_optimization, 'o-', label='Fast Optimization', color=colors['fast'], markersize=3, alpha=0.8)

# 添加英文标题和标签
plt.title('Performance Comparison of Different Optimization Methods across Matrix Sizes\nStep1:Compilation optimization', fontsize=16)
plt.xlabel('Matrix Size', fontsize=14)
plt.ylabel('Performance (Gflops)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)

# 调整x轴刻度，确保能够显示关键点
key_sizes = [32, 64, 128, 256, 512, 768, 1024]
plt.xticks(key_sizes, [str(size) for size in key_sizes])

# 设置y轴范围，确保数据点清晰可见并留出空白
y_min = min(min(blocked), min(o3_optimization), min(fast_optimization)) - 0.5
y_max = max(max(blocked), max(o3_optimization), max(fast_optimization)) + 0.5
plt.ylim(y_min, y_max)

# 添加水平参考线
plt.axhline(y=np.mean(blocked), color=colors['blocked'], linestyle='--', alpha=0.3)
plt.axhline(y=np.mean(o3_optimization), color=colors['o3'], linestyle='--', alpha=0.3)
plt.axhline(y=np.mean(fast_optimization), color=colors['fast'], linestyle='--', alpha=0.3)

# 显示图表
plt.tight_layout()
plt.savefig('step1.png', dpi=300)
plt.show()