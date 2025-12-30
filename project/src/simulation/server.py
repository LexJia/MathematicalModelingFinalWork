# src/simulation/server.py
# -*- coding: utf-8 -*-
import numpy as np
from src import config


class GameServer:
    """
    游戏服务器
    处理全局逻辑：战斗裁决、抽卡池管理、英雄数据索引
    """

    # 抽卡单价 (R3 资源)
    GACHA_COST_R3 = 160.0

    def __init__(self, heroes_df):
        # 构建英雄查找表
        self.heroes_map = heroes_df.set_index('hero_id')
        # 计算全服平均属性 (用于战斗力相对值计算)
        self.avg_hero_stats = self.heroes_map[['attack', 'defense', 'skill']].mean()

    def resolve_battle(self, agent):
        """
        裁决战斗 (S2 验证环节)
        计算逻辑：P(Win) = Sigmoid( Real_Weights * (Hero - Average) + Noise )
        """
        hero = agent.current_hero_stats

        # 计算属性差 (Agent vs Environment)
        delta_A = hero['attack'] - self.avg_hero_stats['attack']
        delta_D = hero['defense'] - self.avg_hero_stats['defense']
        delta_S = hero['skill'] - self.avg_hero_stats['skill']

        # 引入随机扰动 (模拟玩家操作水平波动)
        delta_q = np.random.normal(0, 15)

        # 获取真实权重 (Ground Truth)
        w = config.REAL_WEIGHTS

        # 计算 Logit
        # 注意处理 hidden_buff (机制怪的额外优势)
        hidden_power = hero.get('hidden_buff', 0.0)

        logit = (w['attack'] * delta_A +
                 w['defense'] * delta_D +
                 w['skill'] * delta_S +
                 hidden_power +
                 delta_q) * config.COMBAT_K_FACTOR

        # 防止 exp 溢出
        logit = np.clip(logit, -10, 10)

        win_prob = 1.0 / (1.0 + np.exp(-logit))
        is_win = 1 if np.random.random() < win_prob else 0

        return is_win, win_prob

    def process_gacha(self, agent, is_optimized=False):
        """
        处理抽卡请求
        is_optimized: True=实验组(开启保底), False=对照组(纯随机)
        """
        if agent.resources[2] < self.GACHA_COST_R3:
            return False  # 资源不足

        agent.resources[2] -= self.GACHA_COST_R3

        # 确定当前概率
        prob = config.GACHA_BASE_PROB

        if is_optimized:
            # 实验组：启用90抽硬保底
            # 注意：pity_counter 是"连续未出次数"，当它达到89时，说明这是第90抽
            if agent.pity_counter >= config.GACHA_PITY_LIMIT - 1:
                prob = 1.0

        # 执行抽卡
        is_success = False
        if np.random.random() < prob:
            is_success = True
            agent.pity_counter = 0
            self._grant_hero(agent)
        else:
            is_success = False
            agent.pity_counter += 1

        return is_success

    def _grant_hero(self, agent):
        """发放英雄并执行简单的换装策略"""
        new_hero_id = np.random.choice(self.heroes_map.index)

        if new_hero_id not in agent.owned_heroes:
            agent.owned_heroes.append(new_hero_id)

            # 简单策略：如果新英雄攻击力更高，就换上
            new_hero_stats = self.heroes_map.loc[new_hero_id].to_dict()
            if new_hero_stats['attack'] > agent.current_hero_stats['attack']:
                agent.current_hero_id = new_hero_id
                agent.current_hero_stats = new_hero_stats