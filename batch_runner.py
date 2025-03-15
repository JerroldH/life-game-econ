"""
batch_runner.py

示范批量实验脚本：遍历多组 (冲突成本, 技术外溢, 资源总量) 参数，
收集最终的 Gini、灭绝数、GDP，并可在中间记录时序。

使用:
  python batch_runner.py
结果会打印在终端，你也可将其写入 CSV 文件等。
"""

import itertools
import control
import numpy as np

def run_batch_experiments():
    # 参数范围示例
    conflict_cost_factors = [0.01, 0.05, 0.1, 0.15, 0.2]
    tech_spillovers = [0.0, 0.05, 0.1, 0.2, 0.3]
    resources = [500, 1000, 2000, 5000, 10000]

    steps = 1500   # 每次模拟 1500 代
    grid_size = (30, 30)  # 网格规模


    # 创建输出文件(可选)，或仅print
    output_file = "batch_results.csv"
    with open(output_file, "w") as f:
        f.write("conflict_cost,tech_spill,resource,final_gdp,final_gini,final_extinctions\n")

    # 笛卡尔乘积遍历
    for cc, ts, rs in itertools.product(conflict_cost_factors, tech_spillovers, resources):
        print("---- RUNNING EXPERIMENT ----")
        print(f"ConflictCost={cc}, TechSpill={ts}, Resource={rs}")

        history = control.simulate_economy(
            steps=steps,
            grid_size=grid_size,
            conflict_cost=cc,
            tech_spill=ts,
            global_resource=rs,
            # 其他参数可采用默认，也可自行指定
        )

        # 提取最终值(也可提取最大或平均值)
        final_gdp = history["gdp"][-1]
        final_gini = history["gini"][-1]
        final_extinctions = history["extinctions"][-1]

        print(f"Final GDP={final_gdp:.2f}, Gini={final_gini:.3f}, Extinctions={final_extinctions}")

        # 也可以查看时序中位数或平均：
        # mean_gdp = np.mean(history["gdp"])
        # ...
        # 若要写入CSV：
        csv_line = f"{cc},{ts},{rs},{final_gdp:.2f},{final_gini:.3f},{final_extinctions}\n"
        with open(output_file, "a") as f:
            f.write(csv_line)
        # f.write(csv_line)

if __name__ == "__main__":
    run_batch_experiments()
