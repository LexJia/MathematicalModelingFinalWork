# src/models/economy_dynamics.py
# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
import logging

logger = logging.getLogger(__name__)


# --- 默认的输入输出函数 (作为模块级函数，避免序列化问题) ---
def default_input(t):
    # 示例: 每日产出 R1=100, R2=0, R3=0
    return np.array([100.0, 0.0, 0.0])


def default_output(t):
    # 示例: 每日固定消耗 R3=20
    return np.array([0.0, 0.0, 20.0])


class StabilityAnalyzer:
    """
    第六章：经济系统模型
    对应论文：动态平衡的微分方程与稳定性分析 (公式 6-1, 6-2)
    """

    def __init__(self, alpha=0.5, beta=0.2):
        self.alpha = alpha  # R1 -> R2 转化率
        self.beta = beta  # R2 -> R3 转化率

    def system_dynamics(self, R, t, input_func, output_func):
        """
        定义微分方程组 dR/dt
        """
        R1, R2, R3 = R

        # 允许传入 lambda 或 函数
        I = np.array(input_func(t))
        O = np.array(output_func(t))

        # 确保维度匹配
        if I.shape != (3,) or O.shape != (3,):
            raise ValueError("Input/Output function must return shape (3,)")

        # 动力学方程 (公式 6-1)
        # dR1 = 产出 - 消耗 - 转化流出
        dR1_dt = I[0] - O[0] - self.alpha * R1
        # dR2 = 产出 - 消耗 + R1转化流入 - 转化流出
        dR2_dt = I[1] - O[1] + self.alpha * R1 - self.beta * R2
        # dR3 = 产出 - 消耗 + R2转化流入
        dR3_dt = I[2] - O[2] + self.beta * R2

        return [dR1_dt, dR2_dt, dR3_dt]

    def check_stability(self, input_func=default_input, output_func=default_output):
        """
        求解稳态平衡点并检查稳定性
        """

        # 定义代数方程组：f(R) = 0 (即 dR/dt = 0)
        # fsolve 会自动传入 R 作为第一个参数
        def equations(R):
            # 我们假设在稳态分析时，输入输出取 t=0 时刻的值（或长期均值）
            return self.system_dynamics(R, 0, input_func, output_func)

        # 初始猜测：假设有一定的库存
        R_guess = [100.0, 50.0, 10.0]

        try:
            R_steady, infodict, ier, msg = fsolve(equations, R_guess, full_output=True)

            if ier != 1:
                logger.warning(f"稳态求解未完全收敛: {msg}")
                return False, R_steady

            # 检查解的经济意义 (非负性)
            if np.any(R_steady < -1e-5):  # 容忍微小数值误差
                logger.warning(f"发现负库存稳态 (数学解非物理): {R_steady}")
                return False, R_steady

            logger.info(f"经济系统存在稳态点: R1={R_steady[0]:.1f}, R2={R_steady[1]:.1f}, R3={R_steady[2]:.1f}")
            return True, R_steady

        except Exception as e:
            logger.error(f"稳态分析计算错误: {e}")
            return False, None