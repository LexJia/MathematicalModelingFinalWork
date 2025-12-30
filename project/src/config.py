# -*- coding: utf-8 -*-
import os

# --- 路径配置 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")

# --- 随机种子 ---
RANDOM_SEED = 42

# --- Ch2 & Ch3: 英雄与战斗参数 ---
HERO_COUNT = 100
BATTLE_COUNT = 100000
# 真实权重 (用于生成上帝视角的胜率)
REAL_WEIGHTS = {
    "attack": 0.4,
    "defense": 0.3,
    "skill": 0.3
}
# 胜率缩放因子 (控制Sigmoid的陡峭程度)
COMBAT_K_FACTOR = 0.05

# --- Ch4: 成长模型参数 ---
# 体验锚点: {等级: 累计耗时(分钟)}
GROWTH_ANCHORS = {
    1: 0,
    10: 30,          # 新手期：30分钟升到10级
    20: 180,         # 适应期：3小时
    30: 600,         # 成长期：10小时
    40: 1500,        # 进阶期：25小时
    50: 3000         # 长尾期：50小时
}
LEVEL_MAX = 60

# --- Ch5: 概率与抽卡参数 ---
GACHA_PULLS_SIM = 1000000  # 模拟抽卡次数
GACHA_BASE_PROB = 0.006    # 0.6% 基础概率
GACHA_PITY_LIMIT = 90      # 90抽小保底

# --- Ch6: 经济模型参数 ---
ECO_CONVERSION_ALPHA = 0.5 # R1 -> R2 转化率
ECO_CONVERSION_BETA = 0.2  # R2 -> R3 转化率

# --- Ch7: 仿真参数 ---
SIM_AGENT_COUNT = 1000
SIM_DAYS = 30