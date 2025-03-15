"""
display.py

负责图形界面可视化，包括：
1. 初始化窗口
2. 在网格上绘制各族群颜色
3. 绘制右侧排行榜 (含人口、财富、资本、技术等)
4. Pygame 事件处理 (退出)
"""

import pygame
import sys
from PIL import Image

CELL_SIZE = 5
CELL_PAD = CELL_SIZE // 5

ANIMATION_FRAME_COUNT = 200
ANIMATION_FILENAME = 'new_economic_life_game.gif'
ANIMATION_FRAMES = []

WINDOW_SIZE = None
SCREEN = None

def initialize_screen(g):
    global WINDOW_SIZE, SCREEN
    rows, cols = g.gridSize
    # 额外留200像素展示数据
    WINDOW_SIZE = ((CELL_SIZE + CELL_PAD) * cols + CELL_PAD + 200,
                   (CELL_SIZE + CELL_PAD) * rows + CELL_PAD)
    pygame.display.set_caption("New Life Game - Extended Economic Model")
    SCREEN = pygame.display.set_mode(WINDOW_SIZE)


def draw_block(x, y, color):
    pixel_x = (CELL_SIZE + CELL_PAD) * x + CELL_PAD
    pixel_y = (CELL_SIZE + CELL_PAD) * y + CELL_PAD
    pygame.draw.rect(SCREEN, color, [pixel_x, pixel_y, CELL_SIZE, CELL_SIZE])


def draw(g):
    import control
    rows, cols = g.data.shape
    for y in range(rows):
        for x in range(cols):
            clan_id = g.data[y, x]
            if clan_id in control.CLANS:
                color = control.CLANS[clan_id].color
            else:
                color = (0, 0, 0)
            draw_block(x, y, color)


def draw_leaderboard():
    import control
    font = pygame.font.SysFont(None, 22)
    x_offset = WINDOW_SIZE[0] - 190
    y_offset = 10

    pygame.draw.rect(SCREEN, (200, 200, 200), [x_offset - 10, 0, 210, WINDOW_SIZE[1]])

    # 只显示活着的族群
    living_clans = [c for c in control.CLANS.values() if c.population() > 0]
    sorted_clans = sorted(living_clans, key=lambda c: c.wealth, reverse=True)

    for clan in sorted_clans:
        pop_text = f"Pop={clan.population()}  W={clan.wealth:.1f}"
        cap_text = f"K={clan.capital:.1f} T={clan.tech:.2f} UP={clan.unit_price():.2f}"
        img1 = font.render(f"Clan {clan.id}: {pop_text}", True, clan.color)
        img2 = font.render(cap_text, True, clan.color)
        SCREEN.blit(img1, (x_offset, y_offset))
        SCREEN.blit(img2, (x_offset, y_offset+20))
        y_offset += 50


def orientate():
    pygame.display.flip()


def handleInputEvents():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit(0)


def startClock():
    return pygame.time.Clock()


def updateAnimation(frame_count, frame_rate):
    global ANIMATION_FRAMES
    if frame_count < ANIMATION_FRAME_COUNT:
        ANIMATION_FRAMES.append(Image.frombytes(
            'RGB',
            SCREEN.get_size(),
            pygame.image.tobytes(SCREEN, 'RGB')
        ))
    else:
        ANIMATION_FRAMES[0].save(
            ANIMATION_FILENAME, format='GIF',
            append_images=ANIMATION_FRAMES[1:],
            duration=1000/frame_rate,
            save_all=True,
            loop=0
        )
        sys.exit(0)
