# src/simulation/engine.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json
import logging
from tqdm import tqdm
from src.simulation.agent import Agent
from src.simulation.server import GameServer

logger = logging.getLogger(__name__)


class SimulationEngine:
    """
    全系统耦合仿真引擎 (对应 Ch7)
    """

    def __init__(self, n_agents=1000, days=30):
        self.n_agents = n_agents
        self.days = days
        self.agents = []
        self.server = None
        self.history = []
        self.final_win_rates = []  # 专门存储最后一天的胜率分布

    def _init_agents(self, heroes_df):
        """初始化智能体"""
        self.agents = []
        hero_ids = heroes_df['hero_id'].values

        # 批量初始化以提升性能
        initial_hero_ids = np.random.choice(hero_ids, size=self.n_agents)

        for i in range(self.n_agents):
            h_id = initial_hero_ids[i]
            # 获取英雄初始属性 (转为dict)
            init_hero_data = heroes_df.loc[heroes_df['hero_id'] == h_id].iloc[0].to_dict()
            self.agents.append(Agent(i, [init_hero_data]))

    def run(self, mode, heroes_file, growth_params=None):
        """
        运行仿真
        mode: 'control' | 'experiment'
        """
        logger.info(f"仿真启动: Mode={mode}, Agents={self.n_agents}, Days={self.days}")

        # 1. 加载数据与初始化
        heroes_df = pd.read_csv(heroes_file)
        self.server = GameServer(heroes_df)
        self._init_agents(heroes_df)

        # 解析成长参数
        g_params = None
        if mode == 'experiment' and growth_params:
            try:
                if isinstance(growth_params, str):
                    with open(growth_params, 'r') as f:
                        g_params = json.load(f)
                else:
                    g_params = growth_params
            except Exception as e:
                logger.error(f"加载成长参数失败: {e}，将使用默认线性成长。")

        # 2. 每日循环
        self.history = []

        for day in tqdm(range(1, self.days + 1), desc=f"Simulating ({mode})"):
            daily_stats = {
                "day": day,
                "mode": mode,
                "total_battles": 0,
                "total_wins": 0,
                "total_level_ups": 0,
                "total_gacha_pulls": 0,
                "total_gacha_wins": 0
            }

            # 临时列表用于计算当日均值
            levels = []
            r3_stocks = []
            daily_win_probs = []

            for agent in self.agents:
                # 1. 资源转化
                agent.convert_resources()

                # 2. 决策与行动
                action = agent.decide_action()

                if action == 'battle':
                    is_win, prob = self.server.resolve_battle(agent)
                    daily_stats['total_battles'] += 1
                    daily_stats['total_wins'] += is_win
                    daily_win_probs.append(prob)

                    # 差异化奖励：胜者100，败者30
                    reward = 100.0 if is_win else 30.0
                    agent.gain_resources(reward)

                elif action == 'grow':
                    # 实验组使用拟合参数，对照组使用None
                    if agent.try_level_up(g_params):
                        daily_stats['total_level_ups'] += 1

                elif action == 'gacha':
                    # 实验组开启保底
                    use_pity = (mode == 'experiment')
                    if self.server.process_gacha(agent, is_optimized=use_pity):
                        daily_stats['total_gacha_wins'] += 1
                    daily_stats['total_gacha_pulls'] += 1

                # 收集状态
                levels.append(agent.level)
                r3_stocks.append(agent.resources[2])

            # 3. 统计聚合
            daily_stats['avg_level'] = float(np.mean(levels))
            daily_stats['avg_R3'] = float(np.mean(r3_stocks))
            daily_stats['avg_win_rate'] = float(np.mean(daily_win_probs)) if daily_win_probs else 0.0

            # 如果是最后一天，保存所有胜率数据用于画分布图 (S4)
            if day == self.days:
                self.final_win_rates = daily_win_probs

            self.history.append(daily_stats)

        # 4. 结果输出
        df_history = pd.DataFrame(self.history)

        # 修复：将 list 存入 CSV 会遇到格式问题
        # 我们将最后一日的详细胜率分布保存为 JSON 字符串，放在最后一行的特定列
        # 在 visualization.py 中读取时需注意解析
        df_history['win_rate_distribution_str'] = ""

        # 安全地将列表转为字符串存入最后一行
        # 注意：CSV 读取后需要 eval() 或 json.loads() 还原
        final_idx = df_history.index[-1]
        df_history.at[final_idx, 'win_rate_distribution_str'] = json.dumps(self.final_win_rates)

        return df_history