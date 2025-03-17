# control.py
import numpy as np
import random

# -----------------------------
# 全局可调参数
# -----------------------------
NUM_CLANS = 5
INITIAL_WEALTH = 100.0
DEPRECIATION_RATE = 0.1
CONFLICT_COST_FACTOR = 0.1  # 冲突成本系数
TRANSACTION_SCALE = 1.0     # 交易比例
SAVE_ANIMATION = False

CLANS = {}  # 存储所有 Clan 实例
CELL_SIZE = 5
CELL_PAD = 0

class Clan:
    """
    表示一个族群对象，包含财富、网格单元（人口）、颜色等属性
    """
    def __init__(self, clan_id, color, init_wealth):
        self.id = clan_id
        self.color = color
        self.wealth = init_wealth
        self.cells = set()

    def population(self):
        return len(self.cells)

    def unit_price(self):
        if self.population() == 0:
            return 0.0
        return self.wealth / self.population()


class Grid:
    """
    网格类，存储网格大小、数据矩阵和演化步数
    """
    def __init__(self, size, setup_func):
        self.generations = 0
        self.gridSize = size
        self.rows, self.cols = size
        self.data = setup_func(size)


def initialize_clans(size):
    """
    初始化网格与族群，随机分配网格单元并更新各族群的单元集合
    """
    global CLANS
    CLANS.clear()

    rows, cols = size
    default_colors = [
        (255, 0, 0),    # 红
        (0, 255, 0),    # 绿
        (0, 0, 255),    # 蓝
        (255, 255, 0),  # 黄
        (255, 0, 255),  # 品红
    ]
    for clan_id in range(1, NUM_CLANS + 1):
        c = default_colors[(clan_id - 1) % len(default_colors)]
        CLANS[clan_id] = Clan(clan_id, c, INITIAL_WEALTH)

    data_arr = np.empty(size, dtype=int)
    for y in range(rows):
        for x in range(cols):
            cid = random.choice(list(CLANS.keys()))
            data_arr[y, x] = cid
            CLANS[cid].cells.add((x, y))
    return data_arr


def neighborSquare(x, y):
    """
    返回 (x, y) 的8邻域偏移量
    """
    return [(-1, 0), (-1, -1), (0, -1), (0, 1),
            (-1, 1), (1, 1), (1, -1), (1, 0)]


def conflict_evolve(g: Grid, neighbor_function):
    """
    执行冲突演化：每个单元与其邻居对抗，若成功则转移单元
    """
    rows, cols = g.rows, g.cols
    cell_positions = [(x, y) for y in range(rows) for x in range(cols)]
    random.shuffle(cell_positions)
    
    for (x, y) in cell_positions:
        attacker_id = g.data[y, x]
        if attacker_id not in CLANS:
            continue
        attacker = CLANS[attacker_id]
        if attacker.population() == 0:
            continue

        conflict_cost = CONFLICT_COST_FACTOR * attacker.wealth
        if attacker.wealth < conflict_cost:
            continue

        enemies = []
        for dx, dy in neighbor_function(x, y):
            nx, ny = x + dx, y + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                defender_id = g.data[ny, nx]
                if defender_id != attacker_id and defender_id in CLANS:
                    if CLANS[defender_id].population() > 0:
                        enemies.append((nx, ny, defender_id))
        if enemies:
            nx, ny, enemy_id = random.choice(enemies)
            defender = CLANS[enemy_id]
            attacker_price = attacker.unit_price()
            defender_price = defender.unit_price()
            ransom = TRANSACTION_SCALE * defender_price
            attacker.wealth -= conflict_cost
            if ransom > attacker_price:
                pass
            else:
                reward = 0.5 * defender_price
                attacker.wealth += reward
                g.data[ny, nx] = attacker_id
                defender.cells.discard((nx, ny))
                attacker.cells.add((nx, ny))
    g.generations += 1


def apply_depreciation():
    """对每个族群的财富进行折旧"""
    for clan in CLANS.values():
        clan.wealth *= (1 - DEPRECIATION_RATE)
