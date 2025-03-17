# main.py
import pygame
import matplotlib.pyplot as plt
import control
import display

def compute_nash_deviation():
    """计算当前各族群人口份额与均分状态的平均绝对偏差"""
    total_pop = sum(clan.population() for clan in control.CLANS.values())
    surviving = [clan for clan in control.CLANS.values() if clan.population() > 0]
    if total_pop == 0 or len(surviving) == 0:
        return 0
    equal_share = 1.0 / len(surviving)
    deviation_sum = 0.0
    for clan in surviving:
        share = clan.population() / total_pop
        deviation_sum += abs(share - equal_share)
    return deviation_sum / len(surviving)

def main():
    pygame.init()
    clock = display.startClock()
    frame_rate = 30
    frame_count = 0

    # 初始化网格
    grid_size = (80, 80)
    g = control.Grid(grid_size, control.initialize_clans)
    display.initialize_screen(g)

    # 初始化实时绘图（采用2行2列布局）
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_wealth = axes[0,0]
    ax_population = axes[0,1]
    ax_nash = axes[1,0]
    # ax_extra 可以用于其他指标，如单位价格；这里暂不显示
    ax_extra = axes[1,1]
    ax_extra.set_visible(False)  # 隐藏备用图

    ax_wealth.set_title("Wealth Evolution")
    ax_population.set_title("Population Evolution")
    ax_nash.set_title("Nash Deviation Evolution")
    for ax in [ax_wealth, ax_population, ax_nash]:
        ax.set_xlabel("Generation")

    # 记录模拟历史数据
    simulation_history = {
        "generation": [],
        "wealth": {cid: [] for cid in control.CLANS},
        "population": {cid: [] for cid in control.CLANS},
        "nash_deviation": []
    }

    update_interval = 5

    while True:
        display.handleInputEvents()
        display.draw(g)
        control.conflict_evolve(g, control.neighborSquare)
        control.apply_depreciation()
        display.draw_leaderboard()
        display.orientate()

        if frame_count % update_interval == 0:
            gen = g.generations
            simulation_history["generation"].append(gen)
            for cid, clan in control.CLANS.items():
                simulation_history["wealth"][cid].append(clan.wealth)
                simulation_history["population"][cid].append(clan.population())
            # 计算并记录纳什偏差
            nash_dev = compute_nash_deviation()
            simulation_history["nash_deviation"].append(nash_dev)

            gens = simulation_history["generation"]

            # 更新 Wealth Evolution 图
            ax_wealth.clear()
            ax_wealth.set_title("Wealth Evolution")
            ax_wealth.set_xlabel("Generation")
            for cid, clan in control.CLANS.items():
                # 转换颜色为 0~1 之间
                ccolor = [c/255 for c in clan.color]
                ax_wealth.plot(gens, simulation_history["wealth"][cid],
                               label=f"Clan {cid}", color=ccolor)
            ax_wealth.legend()

            # 更新 Population Evolution 图
            ax_population.clear()
            ax_population.set_title("Population Evolution")
            ax_population.set_xlabel("Generation")
            for cid, clan in control.CLANS.items():
                ax_population.plot(gens, simulation_history["population"][cid],
                                   label=f"Clan {cid}")
            ax_population.legend()

            # 更新 Nash Deviation Evolution 图
            ax_nash.clear()
            ax_nash.set_title("Nash Deviation Evolution")
            ax_nash.set_xlabel("Generation")
            ax_nash.plot(gens, simulation_history["nash_deviation"],
                         label="Avg Nash Deviation", color="purple")
            ax_nash.legend()

            plt.tight_layout()
            plt.pause(0.001)

        if control.SAVE_ANIMATION:
            display.updateAnimation(frame_count, frame_rate)
        frame_count += 1
        clock.tick(frame_rate)

if __name__ == "__main__":
    main()
