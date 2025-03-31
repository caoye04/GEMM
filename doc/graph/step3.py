import matplotlib.pyplot as plt
import numpy as np

# 数据
block_sizes = [21, 41, 61, 91, 121, 151, 191, 221, 251, 253, 255, 256, 257, 259, 
               261, 263, 265, 266, 267, 271, 275, 279, 281, 286]
weighted_avg = [10.67, 14.67, 17.45, 17.39, 15.92, 15.53, 15.45, 15.78, 17.65, 17.88, 17.88, 18.08, 
                18.42, 18.07, 18.51, 18.26, 18.49, 18.78, 18.27, 18.22, 18.16, 18.19, 18.39, 18.41]

# 创建图表
plt.figure(figsize=(12, 7))

# 使用更温和的颜色 - 柔和的蓝绿色
plt.plot(block_sizes, weighted_avg, 'o-', color='#5B9BD5', linewidth=2, markersize=6)

# 添加标题和标签
plt.title('Performance Comparison of Different block size of single-layer matrix blocks in Matrix Multiplication\nStep3:Cache blocking1', 
          fontsize=14, pad=20)  # 增加标题的上边距
plt.xlabel('Block size of single-layer matrix blocks', fontsize=12)
plt.ylabel('Weighted Average Performance (Gflops)', fontsize=12)

# 设置y轴范围
plt.ylim(10, 21)  # 调整纵坐标最大值为21

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 标记最佳性能点
best_idx = np.argmax(weighted_avg)
best_block_size = block_sizes[best_idx]
best_performance = weighted_avg[best_idx]

# 将注释放在下方，避免与标题重叠
plt.annotate(f'Best: {best_performance:.3f} Gflops at block size {best_block_size}',
             xy=(best_block_size, best_performance),
             xytext=(best_block_size-40, best_performance+1.8),  
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=10)

# 优化显示
plt.tight_layout()
plt.savefig('step3.png', dpi=300)
# 显示图表
plt.show()