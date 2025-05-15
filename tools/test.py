import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

def simulate_and_compare_distributions(
    n1=100, mean1=170, std1=5, label1="Class A",
    n2=80, mean2=175, std2=6, label2="Class B"):
    """
    模拟两个班级的身高分布，计算它们之间的Wasserstein距离，并可视化分布。

    参数:
    n1 (int): 班级1的学生数量
    mean1 (float): 班级1的平均身高
    std1 (float): 班级1的身高标准差
    label1 (str): 班级1的标签
    n2 (int): 班级2的学生数量
    mean2 (float): 班级2的平均身高
    std2 (float): 班级2的身高标准差
    label2 (str): 班级2的标签
    """

    # 1. 模拟数据
    # 设置随机种子以保证结果可复现
    np.random.seed(42)
    heights1 = np.random.normal(loc=mean1, scale=std1, size=n1)
    heights2 = np.random.normal(loc=mean2, scale=std2, size=n2)

    # 2. 计算Wasserstein距离
    # scipy.stats.wasserstein_distance可以直接处理原始样本数据
    # 对于一维数据，它计算的是第一个Wasserstein距离，也称为Earth Mover's Distance (EMD)
    # W_1(P, Q) = \int_{-\infty}^{\infty} |F_P(x) - F_Q(x)| dx
    # 其中 F_P 和 F_Q 是两个分布的累积分布函数 (CDF)
    w_distance = wasserstein_distance(heights1, heights2)

    print(f"{label1} 学生数量: {n1}, 平均身高 (模拟): {np.mean(heights1):.2f} cm, 身高标准差 (模拟): {np.std(heights1):.2f} cm")
    print(f"{label2} 学生数量: {n2}, 平均身高 (模拟): {np.mean(heights2):.2f} cm, 身高标准差 (模拟): {np.std(heights2):.2f} cm")
    print(f"\n两个身高分布之间的Wasserstein距离: {w_distance:.4f}")

    # 3. 可视化分布
    plt.style.use('seaborn-v0_8-whitegrid') # 使用一种美观的样式
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # 3a. 绘制直方图比较概率密度
    # 为了更好地比较不同样本量的分布，使用density=True将直方图归一化，使其面积为1
    bins = np.linspace(min(np.min(heights1), np.min(heights2)), max(np.max(heights1), np.max(heights2)), 30)
    axs[0].hist(heights1, bins=bins, alpha=0.7, label=f"{label1} (N={n1})", density=True, color='skyblue')
    axs[0].hist(heights2, bins=bins, alpha=0.7, label=f"{label2} (N={n2})", density=True, color='salmon')
    axs[0].set_title("Height Distribution Histogram (Probability Density)")
    axs[0].set_xlabel("Height (cm)")
    axs[0].set_ylabel("Probability Density")
    axs[0].legend()
    axs[0].grid(True)

    # 3b. 绘制经验累积分布函数 (ECDF)
    # ECDF可以很好地展示Wasserstein距离的含义（两个ECDF之间的面积）
    def ecdf(data):
        x = np.sort(data)
        y = np.arange(1, len(data) + 1) / len(data)
        return x, y

    x1_ecdf, y1_ecdf = ecdf(heights1)
    x2_ecdf, y2_ecdf = ecdf(heights2)

    axs[1].plot(x1_ecdf, y1_ecdf, marker='.', linestyle='none', label=f"{label1} ECDF", color='blue')
    axs[1].plot(x2_ecdf, y2_ecdf, marker='.', linestyle='none', label=f"{label2} ECDF", color='red')

    # 为了更清晰地展示ECDF，可以画出阶梯状的ECDF
    # (scipy.stats.ecdf 也可以用来生成ECDF对象，但这里手动绘制以便理解)
    # 为了填充，我们需要合并x轴坐标并重新计算y值
    all_x = np.sort(np.unique(np.concatenate([x1_ecdf, x2_ecdf])))
    y1_interp = np.interp(all_x, x1_ecdf, y1_ecdf, left=0.0, right=1.0)
    y2_interp = np.interp(all_x, x2_ecdf, y2_ecdf, left=0.0, right=1.0)

    axs[1].plot(all_x, y1_interp, label=f"{label1} ECDF (Steps)", color='cornflowerblue', drawstyle='steps-post')
    axs[1].plot(all_x, y2_interp, label=f"{label2} ECDF (Steps)", color='lightcoral', drawstyle='steps-post')

    # 可选：填充两个ECDF之间的面积，直观表示Wasserstein距离
    axs[1].fill_between(all_x, y1_interp, y2_interp, color='gray', alpha=0.3, step='post', label='|ECDF1 - ECDF2|')


    axs[1].set_title(f"Empirical Cumulative Distribution Function (ECDF) of Heights, Wasserstein Distance: {w_distance:.4f}")
    axs[1].set_xlabel("Height (cm)")
    axs[1].set_ylabel("Cumulative Probability")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"{label1}_vs_{label2}_distribution_comparison.png")
    plt.close()

# --- 运行模拟 ---
# 场景1: 两个班级身高分布相似，但学生数量不同
print("--- 场景 1: 分布相似 ---")
simulate_and_compare_distributions(
    n1=1200, mean1=172, std1=5, label1="Class A",
    n2=50, mean2=171.5, std2=5.5, label2="Class B"
)

print("\n--- 场景 2: 两个班级身高分布差异较大 ---")
simulate_and_compare_distributions(
    n1=100, mean1=165, std1=4, label1="Class C",
    n2=150, mean2=180, std2=6, label2="Class D"
)

print("\n--- 场景 3: 平均身高相似，但离散程度（标准差）不同 ---")
simulate_and_compare_distributions(
    n1=110, mean1=175, std1=3, label1="Class E",
    n2=110, mean2=175, std2=8, label2="Class F"
)