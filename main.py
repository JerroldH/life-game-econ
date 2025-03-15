"""
main.py

图形化运行入口：
1. 初始化pygame
2. 建立网格 (control.Grid)
3. 在循环中执行冲突、折旧、生产、技术外溢等逻辑
4. 实时绘制网格与右侧排行榜
5. 可选择将动画保存

可用来直接演示“新生命游戏”模型的动态演化过程。
"""

import pygame
import control
import display

def main():
    pygame.init()
    clock = display.startClock()
    frame_count = 0
    frame_rate = 50

    grid_size = (150, 150)  # 可调小一点方便可视化

    # 初始化网格
    g = control.Grid(grid_size, control.initialize_clans)
    display.initialize_screen(g)

    while True:
        display.handleInputEvents()

        # 绘制网格
        display.draw(g)

        # 演化：冲突 -> 折旧 -> 投资生产 -> 技术外溢
        control.conflict_evolve(g, control.neighborSquare)
        control.apply_depreciation()
        control.update_investment_and_production()
        control.technology_spillover()

        # 绘制排行榜
        display.draw_leaderboard()
        display.orientate()

        if control.SAVE_ANIMATION:
            display.updateAnimation(frame_count, frame_rate)

        frame_count += 1
        clock.tick(frame_rate)

if __name__ == "__main__":
    main()
