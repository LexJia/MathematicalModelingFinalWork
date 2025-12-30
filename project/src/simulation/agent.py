# src/simulation/agent.py
# -*- coding: utf-8 -*-
import numpy as np
from src import config


class Agent:
    """
    智能体模型
    模拟单个玩家的行为：战斗、资源获取、抽卡、升级
    """

    def __init__(self, agent_id, initial_heroes, strategies=None):
        self.id = agent_id
        self.level = 1

        # 资源库存 [R1:基础, R2:成长, R3:抽卡]
        # 使用 float64 防止长期运行后的精度误差
        self.resources = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # 英雄池
        self.owned_heroes = [h['hero_id'] for h in initial_heroes]

        # 当前出战英雄 (初始化)
        self.current_hero_id = self.owned_heroes[0]
        self.current_hero_stats = initial_heroes[0].copy()  # 使用副本防止引用污染

        # 抽卡保底计数器
        self.pity_counter = 0

        # 行为策略概率 (Battle, Grow, Gacha)
        # 默认：70%战斗，20%成长，10%抽卡
        self.strategies = strategies if strategies else {'battle': 0.7, 'grow': 0.2, 'gacha': 0.1}

    def update_hero_stats(self, hero_data):
        """当全局英雄数据发生变化时（如数值修正后），同步更新当前英雄属性"""
        if self.current_hero_id in hero_data.index:
            self.current_hero_stats = hero_data.loc[self.current_hero_id].to_dict()

    def decide_action(self):
        """基于策略概率决定今日主要行动"""
        roll = np.random.random()
        # 累积概率判定
        threshold_battle = self.strategies['battle']
        threshold_grow = threshold_battle + self.strategies['grow']

        if roll < threshold_battle:
            return 'battle'
        elif roll < threshold_grow:
            return 'grow'
        else:
            return 'gacha'

    def gain_resources(self, amount_r1):
        """获得战斗产出 (随等级提升，模拟通胀压力)"""
        # 产出系数：每级增加 5%
        multiplier = 1.0 + (self.level - 1) * 0.05
        self.resources[0] += amount_r1 * multiplier

    def convert_resources(self):
        """
        资源流转逻辑 (R1 -> R2, R3)
        策略：保留 20% R1，剩余 80% 均分转化为 R2 和 R3
        """
        r1_stock = self.resources[0]
        if r1_stock <= 0:
            return

        to_convert = r1_stock * 0.8
        self.resources[0] -= to_convert  # 扣除 R1

        # 转化计算
        half_convert = to_convert * 0.5
        # R1 -> R2
        self.resources[1] += half_convert * config.ECO_CONVERSION_ALPHA
        # R1 -> R3 (注意：这里假设先转R2再转R3，或者直接用R1->R3的综合汇率)
        # 修正：根据 config，Alpha是R1->R2, Beta是R2->R3
        # 这里模拟的是直接转化路径: R1 -> R2 -> R3
        self.resources[2] += half_convert * config.ECO_CONVERSION_ALPHA * config.ECO_CONVERSION_BETA

    def try_level_up(self, growth_params):
        """
        尝试升级
        growth_params: S2 拟合得到的曲线参数 (a, k, b)
        """
        if self.level >= config.LEVEL_MAX:
            return False

        # 计算升级所需资源 (映射逻辑：时间成本 -> R2资源成本)
        if growth_params:
            a = float(growth_params['a'])
            k = float(growth_params['k'])
            b = float(growth_params['b'])

            # 累积成本差分
            current_cum_cost = a * np.power(self.level, k) + b
            next_cum_cost = a * np.power(self.level + 1, k) + b
            required_r2 = max(10.0, next_cum_cost - current_cum_cost)
        else:
            # 对照组：线性增长成本 (100 * Level)
            required_r2 = 100.0 * self.level

            # 检查并消耗资源
        if self.resources[1] >= required_r2:
            self.resources[1] -= required_r2
            self.level += 1
            return True

        return False