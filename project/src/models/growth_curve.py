# src/models/growth_curve.py
# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import curve_fit
import json
import logging

logger = logging.getLogger(__name__)


class PiecewiseGrowthModel:
    """
    第四章：成长体验模型
    对应论文：混合曲线拟合与节奏控制
    """

    def __init__(self):
        self.params = None

    def power_func(self, l, a, k, b):
        """拟合目标函数: T = a * L^k + b"""
        return a * np.power(l, k) + b

    def fit_and_smooth(self, anchors: dict):
        """
        对应公式 (4-3): 最小二乘拟合
        """
        # 确保转换为 float 类型数组，防止整数除法问题
        levels = np.array(list(anchors.keys()), dtype=float)
        times = np.array(list(anchors.values()), dtype=float)

        # 初始猜测 [a, k, b]
        # 经验值：a=1, k=2 (二次增长), b=0
        p0 = [1.0, 2.0, 0.0]

        try:
            # maxfev=5000 增加迭代次数上限
            popt, pcov = curve_fit(self.power_func, levels, times, p0=p0, maxfev=5000)

            self.params = {
                "a": float(popt[0]),
                "k": float(popt[1]),
                "b": float(popt[2]),
                "formula": "a * L^k + b"
            }
            logger.info(f"成长曲线拟合参数: a={popt[0]:.4f}, k={popt[1]:.4f}, b={popt[2]:.4f}")

        except RuntimeError as e:
            logger.error(f"曲线拟合未收敛，使用默认参数: {e}")
            self.params = {"a": 2.0, "k": 1.8, "b": 0.0, "formula": "fallback"}
        except Exception as e:
            logger.error(f"曲线拟合发生未知错误: {e}")
            self.params = {"a": 2.0, "k": 1.8, "b": 0.0, "formula": "fallback"}

        return self.params

    def calculate_level_up_time(self, level):
        """计算 L -> L+1 所需时间"""
        if not self.params:
            return 10.0  # 默认值

        a, k, b = self.params['a'], self.params['k'], self.params['b']

        # 计算累积时间差
        t_current = a * np.power(level, k) + b
        t_next = a * np.power(level + 1, k) + b

        dt = t_next - t_current
        return max(1.0, dt)  # 保证时间非负


class CliffDetector:
    """断崖检测器"""

    def detect(self, levels, times, threshold_std=2.0):
        """
        对应公式 (4-4): 一阶差分异常检测
        """
        times = np.array(times, dtype=float)
        dt = np.diff(times)  # 一阶差分

        if len(dt) == 0:
            return []

        mean_dt = np.mean(dt)
        std_dt = np.std(dt)

        # 防止 std 为 0 (完全线性) 导致误报
        if std_dt < 1e-6:
            return []

        cliffs = []
        for i, val in enumerate(dt):
            # i 对应的是 levels[i] 到 levels[i+1] 的过程
            if val > mean_dt + threshold_std * std_dt:
                cliffs.append(levels[i])

        return cliffs