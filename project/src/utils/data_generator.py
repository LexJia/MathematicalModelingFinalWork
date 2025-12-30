# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import random
from typing import Tuple, List

# 引用论文配置：真实属性权重 (用于生成Ground Truth，后续需通过回归反推)
REAL_WEIGHTS = {
    "attack": 0.4,
    "defense": 0.3,
    "skill": 0.3
}


def ensure_dir(file_path: str):
    """确保输出目录存在"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


class DataGenerator:
    """
    数据生成工厂类
    对应论文第二章：数据生成与特征工程
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)

    def generate_heroes(self, n_heroes: int = 100, output_path: str = None) -> pd.DataFrame:
        """生成英雄基础属性数据"""
        roles = ['Tank', 'Assassin', 'Mage', 'Marksman', 'Support']
        heroes = []

        for i in range(n_heroes):
            role = np.random.choice(roles)
            if role == 'Tank':
                mu_A, mu_D, mu_S = 50, 90, 40
            elif role == 'Assassin':
                mu_A, mu_D, mu_S = 95, 40, 70
            elif role == 'Mage':
                mu_A, mu_D, mu_S = 60, 40, 95
            elif role == 'Marksman':
                mu_A, mu_D, mu_S = 90, 35, 50
            else:  # Support
                mu_A, mu_D, mu_S = 40, 60, 80

            attack = max(10, np.random.normal(mu_A, 10))
            defense = max(10, np.random.normal(mu_D, 10))
            skill = max(10, np.random.normal(mu_S, 10))

            heroes.append({
                "hero_id": i,
                "role": role,
                "attack": round(attack, 1),
                "defense": round(defense, 1),
                "skill": round(skill, 1),
                "is_op": 0,
                "hidden_buff": 0.0
            })

        df = pd.DataFrame(heroes)

        # 注入 5% 超模数据
        n_op = int(n_heroes * 0.05)
        op_indices = np.random.choice(df.index, n_op, replace=False)
        for idx in op_indices:
            df.at[idx, 'is_op'] = 1
            if np.random.random() < 0.5:
                df.at[idx, 'attack'] *= 1.3
                df.at[idx, 'skill'] *= 1.3
            else:
                df.at[idx, 'hidden_buff'] = 20.0

        # 计算理论战斗力
        df['power_score'] = (
                df['attack'] * REAL_WEIGHTS['attack'] +
                df['defense'] * REAL_WEIGHTS['defense'] +
                df['skill'] * REAL_WEIGHTS['skill'] +
                df['hidden_buff']
        )

        if output_path:
            ensure_dir(output_path)
            df.to_csv(output_path, index=False)
            print(f"[DataGen] 英雄数据已保存: {output_path}")

        return df

    def simulate_battles(self, n_battles: int, heroes_path: str, output_path: str = None) -> pd.DataFrame:
        """生成历史对局日志"""
        if not os.path.exists(heroes_path):
            raise FileNotFoundError(f"未找到英雄数据文件: {heroes_path}，请先运行 generate_heroes。")

        heroes_df = pd.read_csv(heroes_path).set_index("hero_id")
        battles = []

        hero_ids = heroes_df.index.values
        h_a_list = np.random.choice(hero_ids, n_battles)
        h_b_list = np.random.choice(hero_ids, n_battles)

        for i in range(n_battles):
            h_a = h_a_list[i]
            h_b = h_b_list[i]
            if h_a == h_b:
                h_b = np.random.choice(hero_ids)

            stats_a = heroes_df.loc[h_a]
            stats_b = heroes_df.loc[h_b]

            delta_A = stats_a['attack'] - stats_b['attack']
            delta_D = stats_a['defense'] - stats_b['defense']
            delta_S = stats_a['skill'] - stats_b['skill']

            delta_P = stats_a['power_score'] - stats_b['power_score']
            delta_q = np.random.normal(0, 15)  # 玩家水平扰动

            # Sigmoid 胜率生成
            k_factor = 0.05
            logit = k_factor * (delta_P + delta_q)
            win_prob = 1 / (1 + np.exp(-logit))
            result = 1 if np.random.random() < win_prob else 0

            battles.append({
                "battle_id": i,
                "hero_a": h_a,
                "hero_b": h_b,
                "delta_attack": round(delta_A, 2),
                "delta_defense": round(delta_D, 2),
                "delta_skill": round(delta_S, 2),
                "delta_player_skill": round(delta_q, 2),
                "win_prob_real": round(win_prob, 4),
                "win": result
            })

        df = pd.DataFrame(battles)

        if output_path:
            ensure_dir(output_path)
            df.to_csv(output_path, index=False)
            print(f"[DataGen] 战斗日志已保存: {output_path}")

        return df

    def simulate_gacha_logs(self, n_pulls: int, output_path: str = None) -> pd.DataFrame:
        """生成抽卡序列样本"""
        base_prob = 0.006
        pity_limit = 90
        records = []

        random_values = np.random.random(n_pulls)
        current_pity = 0

        for i in range(n_pulls):
            is_success = 0
            current_prob = base_prob

            if current_pity >= (pity_limit - 1):
                current_prob = 1.0

            if random_values[i] < current_prob:
                is_success = 1
                current_pity = 0
            else:
                is_success = 0
                current_pity += 1

            records.append({
                "pull_id": i,
                "pity_counter_before": current_pity - 1 if is_success else current_pity - 1,
                "is_pity_triggered": 1 if current_prob == 1.0 else 0,
                "is_success": is_success,
                "prob_at_pull": current_prob
            })

        df = pd.DataFrame(records)

        if output_path:
            ensure_dir(output_path)
            df.to_csv(output_path, index=False)
            print(f"[DataGen] 抽卡日志已保存: {output_path}")

        return df


# --- 模块级入口函数 ---

def generate_heroes(n=100, output_path="data/raw/heroes.csv"):
    gen = DataGenerator()
    return gen.generate_heroes(n, output_path)


def simulate_battles(n_battles=100000, heroes_path="data/raw/heroes.csv", output_path="data/raw/battle_logs.csv"):
    gen = DataGenerator()
    return gen.simulate_battles(n_battles, heroes_path, output_path)


def simulate_gacha_logs(n_pulls=1000000, output_path="data/raw/gacha_sequences.csv"):
    gen = DataGenerator()
    return gen.simulate_gacha_logs(n_pulls, output_path)


if __name__ == "__main__":
    print(">>> 正在运行 data_generator.py (独立模式)")

    # 1. 生成项目所需的正式数据 (修正点：原来只生成了测试数据)
    print("\n[1/2] 生成项目正式数据 (Production Data)...")
    generate_heroes(n=100, output_path="../../data/raw/heroes.csv")
    simulate_battles(n_battles=100000, heroes_path="../../data/raw/heroes.csv", output_path="../../data/raw/battle_logs.csv")
    simulate_gacha_logs(n_pulls=1000000, output_path="../../data/raw/gacha_sequences.csv")

    # 2. 生成少量测试数据 (可选，用于快速检查)
    # print("\n[2/2] 生成单元测试数据 (Test Data)...")
    # generate_heroes(n=20, output_path="../data/test/test_heroes.csv")
    # simulate_battles(n_battles=100, heroes_path="../data/test/test_heroes.csv", output_path="../data/test/test_battles.csv")

    print("\n>>> 所有数据生成完毕！请检查 data/raw/ 目录。")