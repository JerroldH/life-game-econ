#!/usr/bin/env python
"""
analysis.py

此脚本加载存储在 batch_results.csv 中的实验结果，
并对以下关键指标进行分析：
  - Final GDP
  - Final Gini 系数
  - Final Extinctions（族群灭绝数）

我们以冲突成本 (conflict_cost)、技术外溢 (tech_spill) 和资源总量 (resource) 为自变量，
分析它们对宏观产出、不平等程度的影响。

初步观察：
  1. 当冲突成本较低 (0.01) 时，系统总体 GDP 较高，且 Gini 系数较低；而随着冲突成本上升，
     GDP 显著下降，Gini 系数趋向于升高。
  2. 技术外溢对 GDP 和不平等的影响存在一定的非线性：适度外溢时可降低不平等，
     但过高时对 GDP 的促进作用不稳定。
  3. 在资源充足时 (resource=2000)，总体产出明显提升，而在资源较少时则有明显下降。
  4. 本次实验所有组合下族群灭绝数均为 0，说明模型参数范围内未触发族群“灭绝”现象。

请根据后续实验继续验证和讨论这些现象。
"""

import pandas as pd
import matplotlib.pyplot as plt

# 设置 matplotlib 样式，使图表更美观
# plt.style.use("seaborn-whitegrid")

# 读取 CSV 数据
df = pd.read_csv("batch_results.csv")

# 查看数据基本情况
print("数据预览：")
print(df.head())
print("\n描述统计：")
print(df.describe())

# 提取参数的唯一值，便于后续绘图
conflict_costs = sorted(df["conflict_cost"].unique())
tech_spills = sorted(df["tech_spill"].unique())
resources = sorted(df["resource"].unique())

# ---------------------------
# 图1：Final GDP vs Conflict Cost（不同资源水平下，不同技术外溢水平的曲线）
# ---------------------------
fig, axs = plt.subplots(1, len(resources), figsize=(16, 5), sharey=True)
for i, res in enumerate(resources):
    subset = df[df["resource"] == res]
    for ts in tech_spills:
        data = subset[subset["tech_spill"] == ts].sort_values("conflict_cost")
        axs[i].plot(
            data["conflict_cost"],
            data["final_gdp"],
            marker="o",
            label=f"Tech Spill {ts}",
        )
    axs[i].set_title(f"Resource = {res}")
    axs[i].set_xlabel("Conflict Cost")
    if i == 0:
        axs[i].set_ylabel("Final GDP")
    axs[i].legend()
fig.suptitle("Final GDP vs Conflict Cost (by Resource & Tech Spill)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ---------------------------
# 图2：Final Gini vs Conflict Cost（不同资源水平下，不同技术外溢水平的曲线）
# ---------------------------
fig, axs = plt.subplots(1, len(resources), figsize=(16, 5), sharey=True)
for i, res in enumerate(resources):
    subset = df[df["resource"] == res]
    for ts in tech_spills:
        data = subset[subset["tech_spill"] == ts].sort_values("conflict_cost")
        axs[i].plot(
            data["conflict_cost"],
            data["final_gini"],
            marker="s",
            label=f"Tech Spill {ts}",
        )
    axs[i].set_title(f"Resource = {res}")
    axs[i].set_xlabel("Conflict Cost")
    if i == 0:
        axs[i].set_ylabel("Final Gini")
    axs[i].legend()
fig.suptitle("Final Gini vs Conflict Cost (by Resource & Tech Spill)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ---------------------------
# 图3：散点图展示 Final GDP 与 Final Gini 的关系
# ---------------------------
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    df["final_gdp"],
    df["final_gini"],
    c=df["conflict_cost"],
    cmap="viridis",
    s=100,
    alpha=0.8,
)
ax.set_xlabel("Final GDP")
ax.set_ylabel("Final Gini")
ax.set_title("Scatter: Final GDP vs Final Gini (color = Conflict Cost)")
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Conflict Cost")
plt.tight_layout()
plt.show()

# ---------------------------
# 图4：条形图展示不同资源水平下的平均 Final GDP 与 Final Gini
# ---------------------------
# 分组计算均值
grouped = df.groupby("resource").agg({"final_gdp": "mean", "final_gini": "mean"}).reset_index()
fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()
width = 0.4
ax1.bar(grouped["resource"] - width/2, grouped["final_gdp"], width=width, color="skyblue", label="Final GDP")
ax2.bar(grouped["resource"] + width/2, grouped["final_gini"], width=width, color="salmon", label="Final Gini")
ax1.set_xlabel("Resource")
ax1.set_ylabel("Average Final GDP", color="skyblue")
ax2.set_ylabel("Average Final Gini", color="salmon")
ax1.set_title("Average Final GDP & Gini by Resource Level")
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.tight_layout()
plt.show()

# ---------------------------
# 保存分析结果到图片文件（可选）
# ---------------------------
fig.savefig("analysis_summary.png", dpi=300)

print("分析完成！请查看生成的图表。")
