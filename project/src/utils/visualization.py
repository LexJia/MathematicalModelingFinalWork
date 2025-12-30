# src/utils/visualization.py
# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.4)
sns.set_palette("deep")


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def _parse_distribution_str(df, col_name):
    """解析存储在DataFrame最后一行特定列中的JSON字符串列表"""
    try:
        # 增加健壮性检查：确保列存在且最后一行有数据
        if col_name not in df.columns:
            return []
        json_str = df.iloc[-1][col_name]
        if pd.isna(json_str) or json_str == "":
            return []
        return json.loads(json_str)
    except Exception as e:
        print(f"[Viz Warning] 解析分布数据失败 ({col_name}): {e}")
        return []


# ==========================================
#  第七章：结果对比可视化 (Ch7 Output)
# ==========================================

def plot_win_rate_comparison(control_csv, experiment_csv, save_path):
    """【图 7-4 vs 7-8】胜率分布对比"""
    print(f"正在生成胜率分布图: {os.path.basename(save_path)}")
    if not os.path.exists(control_csv) or not os.path.exists(experiment_csv):
        print("警告：输入文件缺失，跳过胜率对比图。")
        return

    df_c = pd.read_csv(control_csv)
    df_e = pd.read_csv(experiment_csv)
    wr_c = _parse_distribution_str(df_c, 'win_rate_distribution_str')
    wr_e = _parse_distribution_str(df_e, 'win_rate_distribution_str')

    if not wr_c or not wr_e:
        print("警告：解析到的胜率列表为空，跳过绘图。")
        return

    plt.figure(figsize=(10, 6))
    sns.kdeplot(wr_c, label='Control (Original)', fill=True, color='#e74c3c', alpha=0.3)
    sns.kdeplot(wr_e, label='Experiment (Optimized)', fill=True, color='#2ecc71', alpha=0.3)
    plt.axvline(0.5, color='gray', linestyle='--', alpha=0.8)

    # 标注方差
    plt.text(0.02, 0.95, f'Var(Control): {np.var(wr_c):.4f}', transform=plt.gca().transAxes, color='#e74c3c',
             fontweight='bold', fontsize=10)
    plt.text(0.02, 0.90, f'Var(Experiment): {np.var(wr_e):.4f}', transform=plt.gca().transAxes, color='#2ecc71',
             fontweight='bold', fontsize=10)

    plt.title('Hero Win Rate Distribution Comparison')
    plt.xlabel('Win Rate')
    plt.ylabel('Density')
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    ensure_dir(save_path)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_economy_trends(control_csv, experiment_csv, save_path):
    """【图 7-7 vs 7-11】经济库存趋势"""
    print(f"正在生成经济趋势图: {os.path.basename(save_path)}")
    if not os.path.exists(control_csv) or not os.path.exists(experiment_csv): return

    df_c = pd.read_csv(control_csv)
    df_e = pd.read_csv(experiment_csv)

    plt.figure(figsize=(10, 6))
    plt.plot(df_c['day'], df_c['avg_R3'], label='Control (Linear Cost)', color='#e74c3c', marker='o', markersize=4)
    plt.plot(df_e['day'], df_e['avg_R3'], label='Experiment (Stable)', color='#2ecc71', marker='s', markersize=4)

    plt.title('Economic System Stability: Rare Resource (R3)')
    plt.xlabel('Simulation Day')
    plt.ylabel('Avg R3 Stock per Agent')
    plt.legend()
    plt.grid(True, alpha=0.3)
    ensure_dir(save_path)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_growth_analysis(growth_params_path, save_path):
    """【图 7-9】成长曲线断崖分析"""
    print(f"正在生成成长分析图: {os.path.basename(save_path)}")
    try:
        with open(growth_params_path, 'r') as f:
            params = json.load(f)
            a, k, b = params['a'], params['k'], params['b']
    except:
        print("警告：成长参数读取失败，使用默认参数演示。")
        a, k, b = 2.0, 1.8, 0

    levels = np.arange(1, 61)
    time_exp = a * np.power(levels, k) + b
    # 计算一阶差分
    cost_exp = np.diff(time_exp, prepend=time_exp[0])  # 修正 prepend 以保持长度对齐

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：累计曲线
    ax1.plot(levels, time_exp, '-', label='Fitted Curve', color='#3498db', linewidth=2)
    ax1.set_title('Cumulative Cost vs Level')
    ax1.set_xlabel('Level')
    ax1.set_ylabel('Total Time/Resource')
    ax1.grid(True, alpha=0.3)

    # 右图：边际成本（检测断崖）
    ax2.plot(levels, cost_exp, '-', color='#e67e22', label='Level-up Cost')
    threshold = np.mean(cost_exp) + 2 * np.std(cost_exp)
    cliffs = np.where(cost_exp > threshold)[0]
    if len(cliffs) > 0:
        # levels[cliffs] 可能会越界，需注意索引对齐
        valid_cliffs = [i for i in cliffs if i < len(levels)]
        ax2.scatter(levels[valid_cliffs], cost_exp[valid_cliffs], color='red', s=50, label='Cliff Detected')

    ax2.set_title('Level-up Cost Structure')
    ax2.set_xlabel('Level')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    ensure_dir(save_path)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_gacha_pity_validation(raw_gacha_csv, save_path):
    """【图 7-6】抽卡保底验证"""
    print(f"正在生成抽卡验证图: {os.path.basename(save_path)}")
    if not os.path.exists(raw_gacha_csv): return

    df = pd.read_csv(raw_gacha_csv)
    # 筛选出货数据
    success_df = df[df['is_success'] == 1]
    if len(success_df) == 0:
        print("警告：没有成功出货记录，跳过抽卡验证图。")
        return

    pulls_needed = success_df['pity_counter_before'] + 1

    plt.figure(figsize=(10, 6))
    sns.histplot(pulls_needed, bins=90, stat="density", element="step", fill=True, label='Observed (With Pity)',
                 color='#9b59b6')

    # 理论曲线
    x = np.arange(1, 92)
    p = 0.006
    geom_dist = p * (1 - p) ** (x - 1)
    plt.plot(x, geom_dist, 'r--', label='Theoretical (No Pity)')
    plt.axvline(90, color='red', linestyle=':', label='Hard Pity (90)')

    plt.title('Gacha Pity Mechanism Validation')
    plt.xlabel('Pulls Needed')
    plt.legend()
    ensure_dir(save_path)
    plt.savefig(save_path, dpi=300)
    plt.close()


# ==========================================
#  第二章：数据生成 EDA (Chapter 2 Input)
# ==========================================

def plot_hero_attributes_eda(heroes_path, save_path):
    """【图 2-1】英雄属性分布与定位分层"""
    print(f"正在生成 EDA 图: {os.path.basename(save_path)}")
    if not os.path.exists(heroes_path): return
    df = pd.read_csv(heroes_path)

    plt.figure(figsize=(10, 6))
    # 标记超模单位
    df['Type'] = df['is_op'].map({0: 'Normal', 1: 'Overpowered'})

    # 使用 style 区分是否超模，hue 区分职业
    sns.scatterplot(
        data=df, x='defense', y='attack',
        hue='role', style='Type',
        s=100, alpha=0.8, palette='viridis'
    )
    plt.title('Hero Attribute Distribution by Role')
    plt.xlabel('Defense')
    plt.ylabel('Attack')
    # 调整图例位置
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    ensure_dir(save_path)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_combat_sigmoid_eda(battle_path, save_path):
    """【图 2-2】战斗力差与胜率 Sigmoid 验证"""
    print(f"正在生成 EDA 图: {os.path.basename(save_path)}")
    if not os.path.exists(battle_path): return
    df = pd.read_csv(battle_path)

    # 1. 计算总属性差 (近似)
    df['total_delta'] = df['delta_attack'] + df['delta_defense'] + df['delta_skill']

    # 2. 分桶计算观测胜率 (用于画散点)
    bins = np.linspace(df['total_delta'].min(), df['total_delta'].max(), 30)
    df['bin'] = pd.cut(df['total_delta'], bins=bins)
    binned = df.groupby('bin', observed=True)['win'].mean().reset_index()
    binned['delta_center'] = binned['bin'].apply(lambda x: x.mid)

    plt.figure(figsize=(10, 6))

    # [修正点]：散点图使用分桶后的均值
    sns.scatterplot(data=binned, x='delta_center', y='win', s=80, color='blue', label='Observed (Binned)')

    # [修正点]：回归曲线使用原始0/1数据进行 Logistic 拟合
    # 为了绘图速度，如果数据量过大 (>10000)，可以采样
    sample_df = df.sample(min(10000, len(df)), random_state=42)
    sns.regplot(
        data=sample_df, x='total_delta', y='win',
        scatter=False, logistic=True, color='red',
        line_kws={'linestyle': '--'}, label='Sigmoid Fit (Logistic)'
    )

    plt.title('Win Rate vs. Attribute Advantage')
    plt.xlabel('Power Delta')
    plt.ylabel('Win Probability')
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)

    ensure_dir(save_path)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_growth_anchors_eda(anchors_dict, save_path):
    """【图 2-3】成长节奏锚点"""
    print(f"正在生成 EDA 图: {os.path.basename(save_path)}")
    levels = list(anchors_dict.keys())
    times = list(anchors_dict.values())

    plt.figure(figsize=(10, 6))
    plt.plot(levels, times, 'o--', color='#2c3e50', markersize=10, label='Design Anchors')

    # 标注数值
    for l, t in zip(levels, times):
        plt.text(l, t + (max(times) * 0.05), f'{t}m', ha='center', fontsize=10)

    plt.title('Growth Rhythm Anchors')
    plt.xlabel('Target Level')
    plt.ylabel('Expected Cumulative Time (min)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    ensure_dir(save_path)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_chapter_2_eda(heroes_path, battle_path, anchors_dict, save_dir):
    """一键生成第二章 EDA 图表"""
    plot_hero_attributes_eda(heroes_path, os.path.join(save_dir, "fig_2_1_hero_attributes.png"))
    plot_combat_sigmoid_eda(battle_path, os.path.join(save_dir, "fig_2_2_combat_sigmoid.png"))
    plot_growth_anchors_eda(anchors_dict, os.path.join(save_dir, "fig_2_3_growth_anchors.png"))