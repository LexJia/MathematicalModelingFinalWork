# src/models/combat_balance.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


class CombatModel:
    """
    第三章：数值平衡模型
    对应论文：基于回归分析的权重修正与约束优化
    """

    def __init__(self):
        # fit_intercept=False 是关键，因为我们的数据生成公式 Logit = k*(dP) 没有常数偏差
        # 如果生成数据包含 Blue/Red 方阵营优势，则应设为 True
        self.model = LogisticRegression(fit_intercept=False, solver='lbfgs')
        self.real_weights = {}
        self.df_battles = None

    def load_data(self, battle_logs_path: str):
        if not battle_logs_path:
            raise ValueError("路径为空")
        self.df_battles = pd.read_csv(battle_logs_path)
        logger.info(f"成功加载战斗日志，共 {len(self.df_battles)} 条记录。")

    def estimate_weights(self) -> dict:
        """
        对应公式 (3-2): Pr(Y=1) = Sigmoid(beta * Delta)
        """
        if self.df_battles is None:
            raise ValueError("数据未加载，请先调用 load_data()")

        # 特征必须与 data_generator 中的列名完全一致
        features = ['delta_attack', 'delta_defense', 'delta_skill', 'delta_player_skill']
        X = self.df_battles[features]
        y = self.df_battles['win']

        self.model.fit(X, y)

        coefs = self.model.coef_[0]
        self.real_weights = {
            "attack": coefs[0],
            "defense": coefs[1],
            "skill": coefs[2],
            "player_skill": coefs[3]
        }

        score = self.model.score(X, y)
        logger.info(f"权重回归分析完成。模型准确率 (Accuracy): {score:.4f}")
        logger.info(f"识别到的权重: {self.real_weights}")
        return self.real_weights

    def optimize_balance(self, heroes_path: str, weights: dict, lambda_reg=0.01) -> pd.DataFrame:
        """
        对应公式 (3-5) & (3-6): 约束优化求解
        """
        if not weights:
            logger.warning("权重字典为空，尝试使用模型内部权重...")
            if not self.real_weights:
                raise ValueError("未找到权重参数，请先运行 estimate_weights()。")
            weights = self.real_weights

        heroes_df = pd.read_csv(heroes_path)

        w_A = weights.get('attack', 0)
        w_D = weights.get('defense', 0)
        w_S = weights.get('skill', 0)

        # 计算全服平均属性 (作为"标准对手")
        avg_A = heroes_df['attack'].mean()
        avg_D = heroes_df['defense'].mean()
        avg_S = heroes_df['skill'].mean()

        corrected_heroes = []

        logger.info("开始执行多目标约束优化...")

        for _, hero in heroes_df.iterrows():
            x0 = np.array([hero['attack'], hero['defense'], hero['skill']])

            # 目标函数：让胜率趋近50% (Logit趋近0)，同时惩罚改动幅度
            def objective(x):
                A, D, S = x
                # 面对标准对手的属性差
                delta_A = A - avg_A
                delta_D = D - avg_D
                delta_S = S - avg_S

                # 计算预测胜率的 Logit
                # 注意：这里忽略 player_skill，假设由于匹配机制，双方水平接近
                logit = w_A * delta_A + w_D * delta_D + w_S * delta_S
                p_pred = 1 / (1 + np.exp(-logit))

                # 损失函数 = (胜率偏差)^2 + 正则项 * (改动幅度)^2
                loss = (p_pred - 0.5) ** 2 + lambda_reg * np.sum((x - x0) ** 2)
                return loss

            # 约束：改动幅度不超过 ±20%
            bounds = [
                (x0[0] * 0.8, x0[0] * 1.2),
                (x0[1] * 0.8, x0[1] * 1.2),
                (x0[2] * 0.8, x0[2] * 1.2)
            ]

            # 必须使用支持 bounds 的方法，如 L-BFGS-B 或 SLSQP
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

            new_attrs = res.x
            hero_copy = hero.copy()
            hero_copy['attack'] = round(new_attrs[0], 1)
            hero_copy['defense'] = round(new_attrs[1], 1)
            hero_copy['skill'] = round(new_attrs[2], 1)
            hero_copy['is_op'] = 0  # 标记为已修正

            corrected_heroes.append(hero_copy)

        logger.info(f"已完成 {len(corrected_heroes)} 个英雄的数值修正。")
        return pd.DataFrame(corrected_heroes)