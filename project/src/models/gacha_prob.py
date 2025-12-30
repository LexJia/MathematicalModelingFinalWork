# src/models/gacha_prob.py
# -*- coding: utf-8 -*-
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MarkovChainAnalyzer:
    """
    第五章：概率机制模型
    对应论文：保底策略的马尔可夫链分析 (公式 5-1, 5-2)
    """

    def analyze_pity_system(self, base_prob=0.006, pity_limit=90):
        """
        计算带保底机制的期望抽数与稳态分布
        """
        n_states = pity_limit  # 状态空间: 0 ~ 89 (共90个状态)
        P = np.zeros((n_states, n_states))

        # 1. 构建转移矩阵 P
        for k in range(n_states):
            # 计算当前状态下的出货率
            if k == n_states - 1:
                p_k = 1.0  # 硬保底 (第90抽必出)
            else:
                p_k = base_prob

            # 转移逻辑
            # 出货 -> 回到状态 0
            P[k, 0] = p_k

            # 未出货 -> 状态 +1
            if k < n_states - 1:
                P[k, k + 1] = 1.0 - p_k

        # 2. 计算稳态分布 pi (pi * P = pi) => (P.T - I) * pi.T = 0
        A = P.T - np.eye(n_states)
        # 替换最后一个方程为归一化条件: sum(pi) = 1
        A[-1, :] = 1.0
        b = np.zeros(n_states)
        b[-1] = 1.0

        try:
            steady_state = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            logger.warning("矩阵奇异，无法求解稳态分布，使用均匀分布代替。")
            steady_state = np.ones(n_states) / n_states

        # 3. 计算期望抽数 E[T]
        # E[T] = 1 + sum(P(T > k)) = sum(prod(1-pi)) ...
        # 这里使用简单的迭代法计算期望值: E = sum( (k+1) * P(正好在第k+1抽出) )
        expected_pulls = 0.0
        prob_survive = 1.0  # 活过前 k 抽的概率

        for k in range(n_states):
            # 当前这一抽的出货率
            if k == n_states - 1:
                p_current = 1.0
            else:
                p_current = base_prob

            # 在第 k+1 抽正好出货的概率 = (活过前k抽) * (当前出货)
            prob_hit_now = prob_survive * p_current

            # 贡献期望： (k+1) * 概率
            expected_pulls += (k + 1) * prob_hit_now

            # 更新存活率
            prob_survive *= (1.0 - p_current)

        return expected_pulls, steady_state