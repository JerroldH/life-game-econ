"""
control.py

此文件实现一个重新设计的“演化经济学+生命游戏”模型，具有以下特征：
1. 多族群 (Clan)：每个族群拥有人口(网格单元集合)、财富、资本、技术等属性
2. 冲突成本：当一个族群尝试击杀或掠夺另一个族群的单元，需要付出一定军费
3. 资源限制：通过对生产函数添加一个全局有限资源R，或更高的边际递减
4. 技术外溢：每代各族群的技术水平均会与全局平均值接轨一部分
5. Gini系数与灭绝统计：每代可计算分配不平等程度，以及统计灭绝族群数量

通过该模型，我们可在无再分配的条件下，检验冲突、投资、技术外溢对长期增长与结构演化的影响。
"""

import numpy as np
import random

# -----------------------------
# 全局可调参数 (默认值，可被simulate_economy覆盖)
# -----------------------------

#: 族群数量
NUM_CLANS = 5

#: 初始财富
INITIAL_WEALTH = 100

#: 初始资本
INITIAL_CAPITAL = 50

#: 初始技术
INITIAL_TECH = 1.0

#: 冲突成本系数（攻击方需支付的军费 = conflict_cost_factor * attacker_wealth）
CONFLICT_COST_FACTOR = 0.05

#: 技术外溢系数 (0~1)，每代使技术向全局平均水平靠拢
TECH_SPILLOVER = 0.1

#: 资源限制相关：生产函数中再引入一个有限资源R
#: 若资源很大则几乎不受限，若资源有限则增加强边际递减
GLOBAL_RESOURCE = 1000.0

#: 生产函数参数
INVESTMENT_RATE = 0.2         # 投资率
CAPITAL_DEPRECIATION_RATE = 0.05
PRODUCTION_ALPHA = 0.3        # 资本弹性
RESOURCE_BETA = 0.2           # 资源弹性：产出中资源占的比重
INNOVATION_RATE = 0.01        # 基础技术进步率

#: 财富折旧率
WEALTH_DEPRECIATION = 0.01

#: 冲突交易倍数
TRANSACTION_SCALE = 1.0

#: 颜色可用于图形化
CLAN_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (255, 128, 0),
    (0, 255, 128),
    (128, 0, 255),
]

CLANS = {}
SAVE_ANIMATION = False


class Clan:
    """
    表示一个族群对象，包含:
    - wealth (财富)
    - capital (资本)
    - tech (技术水平)
    - cells (人口，网格单元坐标集合)
    - color (仅绘图用)
    """

    def __init__(self, clan_id: int, color: tuple, init_wealth: float, init_capital: float, init_tech: float):
        self.id = clan_id
        self.color = color
        self.wealth = init_wealth
        self.capital = init_capital
        self.tech = init_tech
        self.cells = set()

    def population(self) -> int:
        """
        人口等于 cells 大小。
        """
        return len(self.cells)

    def unit_price(self) -> float:
        """
        单位价格 = wealth / population。
        若人口=0，则返回0。
        """
        pop = self.population()
        return self.wealth / pop if pop > 0 else 0.0


class Grid:
    """
    网格类，用于保存大小(rows, cols)与 data[y, x] -> clan_id。
    另含 generations 记录演化步数。
    """

    def __init__(self, size, setup_func):
        self.generations = 0
        self.gridSize = size
        self.data = setup_func(size)


def initialize_clans(size):
    """
    初始化网格与族群。随机为每个单元分配族群ID，并更新CLANS字典。
    默认使用全局NUM_CLANS等参数，也可由simulate_economy覆盖。
    """
    global CLANS
    CLANS.clear()

    rows, cols = size
    for clan_id in range(1, NUM_CLANS + 1):
        color_index = (clan_id - 1) % len(CLAN_COLORS)
        CLANS[clan_id] = Clan(
            clan_id,
            CLAN_COLORS[color_index],
            INITIAL_WEALTH,
            INITIAL_CAPITAL,
            INITIAL_TECH
        )

    data_arr = np.empty(size, dtype=int)
    for y in range(rows):
        for x in range(cols):
            cid = random.choice(list(CLANS.keys()))
            data_arr[y, x] = cid
            CLANS[cid].cells.add((x, y))

    return data_arr


def neighborSquare(x, y):
    """
    返回 (x,y) 的8邻域偏移量。
    """
    return [(-1, 0), (-1, -1), (0, -1), (0, 1),
            (-1, 1), (1, 1), (1, -1), (1, 0)]


# -----------------------------
# 模型核心演化流程
# -----------------------------

def conflict_evolve(g: Grid, neighbor_function):
    """
    冲突演化：每个单元与邻居发生冲突。引入“冲突成本”：
     - 攻击方需先付 cost = CONFLICT_COST_FACTOR * attacker_wealth (若不够则放弃)
     - 若攻击方依然有足够能力，则比较赎金 (defender_price * TRANSACTION_SCALE) vs 攻击方单位价格
     - 若赎金较高，则放弃进攻；否则杀戮并转移单元，获得少量奖励
    """
    rows, cols = g.data.shape
    cell_positions = [(x, y) for y in range(rows) for x in range(cols)]
    random.shuffle(cell_positions)

    for (x, y) in cell_positions:
        attacker_id = g.data[y, x]
        if attacker_id not in CLANS:
            continue
        attacker = CLANS[attacker_id]
        if attacker.population() == 0:
            continue

        # 计算冲突成本
        conflict_cost = CONFLICT_COST_FACTOR * attacker.wealth
        if attacker.wealth < conflict_cost:
            # 军费不足，放弃冲突
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

            # 攻击方先支付冲突成本(军费)
            attacker.wealth -= conflict_cost

            if ransom > attacker_price:
                # 防守方要价高，攻击方放弃
                pass
            else:
                # 成功杀戮
                # 进攻方获得少量奖励(对方单元价格一半)
                reward = 0.5 * defender_price
                attacker.wealth += reward
                # 转移单元归属
                g.data[ny, nx] = attacker_id
                defender.cells.discard((nx, ny))
                attacker.cells.add((nx, ny))

    g.generations += 1


def apply_depreciation():
    """
    每代结束时，对财富进行折旧。
    """
    for clan in CLANS.values():
        clan.wealth *= (1 - WEALTH_DEPRECIATION)


def technology_spillover():
    """
    技术外溢：将所有族群的 tech 与全局平均值靠拢 TECH_SPILLOVER。
    """
    living_clans = [c for c in CLANS.values() if c.population() > 0]
    if not living_clans:
        return

    avg_tech = np.mean([clan.tech for clan in living_clans])
    for clan in living_clans:
        clan.tech += TECH_SPILLOVER * (avg_tech - clan.tech)


def update_investment_and_production() -> dict[int, float]:
    """
    投资与生产：考虑资源限制(RESOURCE_BETA) 以及 capital^alpha * L^(1-alpha-...)。
    Y = A * K^alpha * L^gamma * R^beta, (其中 gamma = 1 - alpha - beta 以保证规模报酬?)
    并加入基础技术增长 (INNOVATION_RATE)。
    返回 {clan_id: production} 用于统计GDP。
    """
    production_dict = {}

    living_clans = [cid for cid, c in CLANS.items() if c.population() > 0]
    if not living_clans:
        return {}

    # 计算所有族群的资源占用(或共享)情况
    # 简化：假定大家共享 GLOBAL_RESOURCE，按人口比或资本比分配可用资源
    total_pop = sum(CLANS[cid].population() for cid in living_clans)
    # exponent for population
    gamma = max(0.0, 1.0 - PRODUCTION_ALPHA - RESOURCE_BETA)

    for cid in living_clans:
        clan = CLANS[cid]
        # 投资
        invest = INVESTMENT_RATE * clan.wealth
        clan.capital = clan.capital * (1 - CAPITAL_DEPRECIATION_RATE) + invest

        # 分配到的资源(简化为： GLOBAL_RESOURCE * (clan_pop / total_pop) )
        pop = clan.population()
        if total_pop > 0:
            resource_share = GLOBAL_RESOURCE * (pop / total_pop)
        else:
            resource_share = 0

        # 生产函数: Y = tech * capital^alpha * pop^gamma * resource_share^beta
        cap_term = clan.capital ** PRODUCTION_ALPHA
        pop_term = (pop ** gamma) if pop > 0 else 0
        resource_term = (resource_share ** RESOURCE_BETA)
        Y = clan.tech * cap_term * pop_term * resource_term

        clan.wealth += Y
        production_dict[cid] = Y

        # 基础技术增长 + 随机扰动
        clan.tech *= (1 + INNOVATION_RATE + random.uniform(-0.005, 0.005))

    return production_dict


# -----------------------------
# 额外统计：Gini系数、灭绝数
# -----------------------------

def compute_gini(values: list[float]) -> float:
    """
    计算给定列表的Gini系数(0~1)。
    若所有值相同，则Gini=0；若极端不平等则接近1。
    """
    if len(values) == 0:
        return 0
    sorted_vals = sorted(values)
    cum = 0
    total = sum(sorted_vals)
    for i, val in enumerate(sorted_vals):
        cum += val
        # G = 1 - sum( (N - i) * val ) / (N * total)  (有多种写法)
    n = len(values)
    # 常用的二重循环形式
    cumulative = 0
    for i, val in enumerate(sorted_vals):
        cumulative += (2*(i+1) - n - 1) * val
    return cumulative / (n * sum(sorted_vals)) if total > 0 else 0


def simulate_economy(
    steps=200,
    grid_size=(50, 50),
    num_clans=NUM_CLANS,
    init_wealth=INITIAL_WEALTH,
    init_capital=INITIAL_CAPITAL,
    init_tech=INITIAL_TECH,
    conflict_cost=CONFLICT_COST_FACTOR,
    tech_spill=TECH_SPILLOVER,
    global_resource=GLOBAL_RESOURCE,
    investment_rate=INVESTMENT_RATE,
    capital_depr=CAPITAL_DEPRECIATION_RATE,
    alpha=PRODUCTION_ALPHA,
    beta=RESOURCE_BETA,
    innovation=INNOVATION_RATE,
    wealth_depr=WEALTH_DEPRECIATION,
    transaction_scale=TRANSACTION_SCALE
) -> dict:
    """
    可直接调用的模拟函数(无图形界面)。返回完整历史数据，供外部分析。

    返回结构:
    {
      "generation": [...],
      "gdp": [...],
      "gini": [...],
      "extinctions": [...],
      "wealth": {cid: [...]},
      "pop": {cid: [...]},
      "capital": {cid: [...]},
      "tech": {cid: [...]}
    }
    """
    # 临时保存并覆盖全局参数
    global NUM_CLANS, INITIAL_WEALTH, INITIAL_CAPITAL, INITIAL_TECH
    global CONFLICT_COST_FACTOR, TECH_SPILLOVER, GLOBAL_RESOURCE
    global INVESTMENT_RATE, CAPITAL_DEPRECIATION_RATE
    global PRODUCTION_ALPHA, RESOURCE_BETA, INNOVATION_RATE, WEALTH_DEPRECIATION
    global TRANSACTION_SCALE

    old_num_clans = NUM_CLANS
    old_init_w = INITIAL_WEALTH
    old_init_c = INITIAL_CAPITAL
    old_init_t = INITIAL_TECH
    old_conf_cost = CONFLICT_COST_FACTOR
    old_spill = TECH_SPILLOVER
    old_res = GLOBAL_RESOURCE
    old_inv = INVESTMENT_RATE
    old_cap_dep = CAPITAL_DEPRECIATION_RATE
    old_alpha = PRODUCTION_ALPHA
    old_beta = RESOURCE_BETA
    old_innov = INNOVATION_RATE
    old_wdepr = WEALTH_DEPRECIATION
    old_tscale = TRANSACTION_SCALE

    NUM_CLANS = num_clans
    INITIAL_WEALTH = init_wealth
    INITIAL_CAPITAL = init_capital
    INITIAL_TECH = init_tech
    CONFLICT_COST_FACTOR = conflict_cost
    TECH_SPILLOVER = tech_spill
    GLOBAL_RESOURCE = global_resource
    INVESTMENT_RATE = investment_rate
    CAPITAL_DEPRECIATION_RATE = capital_depr
    PRODUCTION_ALPHA = alpha
    RESOURCE_BETA = beta
    INNOVATION_RATE = innovation
    WEALTH_DEPRECIATION = wealth_depr
    TRANSACTION_SCALE = transaction_scale

    # ---- 初始化网格与族群
    g = Grid(grid_size, initialize_clans)

    history = {
        "generation": [],
        "gdp": [],
        "gini": [],
        "extinctions": [],
        "wealth": {cid: [] for cid in CLANS},
        "pop": {cid: [] for cid in CLANS},
        "capital": {cid: [] for cid in CLANS},
        "tech": {cid: [] for cid in CLANS},
    }

    for _ in range(steps):
        conflict_evolve(g, neighborSquare)
        apply_depreciation()
        production_dict = update_investment_and_production()
        # 技术外溢
        technology_spillover()

        # 计算指标
        total_gdp = sum(production_dict.values())
        living_clans = [c for c in CLANS.values() if c.population() > 0]
        living_wealths = [c.wealth for c in living_clans]
        gini_val = compute_gini(living_wealths) if living_wealths else 0
        extinction_count = sum(1 for c in CLANS.values() if c.population() == 0)

        # 记录
        gen = g.generations
        history["generation"].append(gen)
        history["gdp"].append(total_gdp)
        history["gini"].append(gini_val)
        history["extinctions"].append(extinction_count)

        for cid, c in CLANS.items():
            history["wealth"][cid].append(c.wealth)
            history["pop"][cid].append(c.population())
            history["capital"][cid].append(c.capital)
            history["tech"][cid].append(c.tech)

    # ---- 恢复全局变量
    NUM_CLANS = old_num_clans
    INITIAL_WEALTH = old_init_w
    INITIAL_CAPITAL = old_init_c
    INITIAL_TECH = old_init_t
    CONFLICT_COST_FACTOR = old_conf_cost
    TECH_SPILLOVER = old_spill
    GLOBAL_RESOURCE = old_res
    INVESTMENT_RATE = old_inv
    CAPITAL_DEPRECIATION_RATE = old_cap_dep
    PRODUCTION_ALPHA = old_alpha
    RESOURCE_BETA = old_beta
    INNOVATION_RATE = old_innov
    WEALTH_DEPRECIATION = old_wdepr
    TRANSACTION_SCALE = old_tscale

    return history
