# display.py
import pygame
import sys
from PIL import Image
import control

SCREEN = None
WINDOW_SIZE = (800, 600)
ANIMATION_FRAME_COUNT = 200
ANIMATION_FILENAME = 'clan_conflict.gif'
ANIMATION_FRAMES = []

def initialize_screen(g):
    global SCREEN, WINDOW_SIZE
    pygame.display.set_caption("Clan Conflict Automata")
    SCREEN = pygame.display.set_mode(WINDOW_SIZE)

def draw_block(x, y, acolor):
    pixel_x = x * control.CELL_SIZE + control.CELL_PAD
    pixel_y = y * control.CELL_SIZE + control.CELL_PAD
    pygame.draw.rect(SCREEN, acolor, [pixel_x, pixel_y, control.CELL_SIZE, control.CELL_SIZE])

def draw(g):
    rows, cols = g.rows, g.cols
    for y in range(rows):
        for x in range(cols):
            clan_id = g.data[y, x]
            if clan_id in control.CLANS:
                c = control.CLANS[clan_id].color
            else:
                c = (0, 0, 0)
            draw_block(x, y, c)

def draw_leaderboard():
    font = pygame.font.SysFont(None, 24)
    x_offset = 420
    y_offset = 10
    w, h = 400, 450
    pygame.draw.rect(SCREEN, (200, 200, 200), [x_offset, 0, w, h])
    living_clans = [c for c in control.CLANS.values() if c.population() > 0]
    sorted_clans = sorted(living_clans, key=lambda c: c.wealth, reverse=True)[:5]
    for clan in sorted_clans:
        pop_text = f"Pop={clan.population()}  W={clan.wealth:.1f}"
        price_text = f"UP={clan.unit_price():.2f}"
        img1 = font.render(f"Clan {clan.id}: {pop_text}", True, clan.color)
        img2 = font.render(price_text, True, clan.color)
        SCREEN.blit(img1, (x_offset, y_offset))
        SCREEN.blit(img2, (x_offset, y_offset + 22))
        y_offset += 44

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
        img = Image.frombytes('RGB', SCREEN.get_size(),
                              pygame.image.tobytes(SCREEN, 'RGB'))
        ANIMATION_FRAMES.append(img)
    else:
        ANIMATION_FRAMES[0].save(
            ANIMATION_FILENAME,
            format='GIF',
            append_images=ANIMATION_FRAMES[1:],
            duration=1000/frame_rate,
            save_all=True,
            loop=0
        )
        sys.exit(0)
