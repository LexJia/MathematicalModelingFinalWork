# main.py
# -*- coding: utf-8 -*-

"""
基于多目标优化理论的游戏数值系统建模与仿真研究
主程序入口 (Main Entry Point)

流程：
1. Data Generation (Ch2): 生成合成数据 (英雄、战斗日志、抽卡序列)
2. Model Optimization (Ch3-6): 训练回归模型、拟合成长曲线、计算稳态分布
3. System Simulation (Ch7): 运行多智能体仿真 (对照组 vs 实验组)
4. Visualization (Ch7): 生成对比图表与分析结果
"""

import os
import sys
import logging
import json
import pandas as pd
import numpy as np

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入各模块
from src import config
from src.utils import data_generator, visualization
from src.models import combat_balance, growth_curve, gacha_prob, economy_dynamics
from src.simulation import SimulationEngine

# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("simulation.log", mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def setup_directories():
    """初始化目录结构"""
    dirs = [
        config.DATA_RAW_DIR,
        config.DATA_PROCESSED_DIR,
        config.OUTPUT_FIG_DIR
    ]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
    logger.info("目录结构检查完成。")


def step_1_data_generation():
    """
    【第二章：数据生成】
    生成支撑后续建模的合成数据，并输出EDA可视化图表
    """
    logger.info("-" * 30)
    logger.info(">>> STEP 1: 数据生成 (Data Generation)")

    # --- 1. 定义所有路径变量 (确保路径都在此处定义) ---
    path_heroes = os.path.join(config.DATA_RAW_DIR, "heroes.csv")
    path_battles = os.path.join(config.DATA_RAW_DIR, "battle_logs.csv")
    path_gacha = os.path.join(config.DATA_RAW_DIR, "gacha_sequences.csv")
    path_fig_dir = config.OUTPUT_FIG_DIR  # <--- 关键：显式定义这个变量

    # --- 2. 执行数据生成逻辑 ---

    # 生成英雄 (包含超模数据)
    if not os.path.exists(path_heroes):
        logger.info("正在生成英雄基础数据...")
        data_generator.generate_heroes(n=config.HERO_COUNT, output_path=path_heroes)
    else:
        logger.info("英雄数据已存在，跳过生成。")

    # 生成战斗日志 (用于回归分析)
    if not os.path.exists(path_battles):
        logger.info(f"正在模拟 {config.BATTLE_COUNT} 场战斗...")
        data_generator.simulate_battles(
            n_battles=config.BATTLE_COUNT,
            heroes_path=path_heroes,
            output_path=path_battles
        )
    else:
        logger.info("战斗日志已存在，跳过生成。")

    # 生成抽卡序列 (用于验证保底机制)
    if not os.path.exists(path_gacha):
        logger.info(f"正在模拟 {config.GACHA_PULLS_SIM} 次抽卡...")
        data_generator.simulate_gacha_logs(
            n_pulls=config.GACHA_PULLS_SIM,
            output_path=path_gacha
        )
    else:
        logger.info("抽卡日志已存在，跳过生成。")

    # --- 3. 生成第二章可视化图表 (EDA) ---
    # 确保此处调用的变量在上面都已定义
    logger.info("正在生成第二章数据探索性分析(EDA)图表...")
    try:
        visualization.plot_chapter_2_eda(
            heroes_path=path_heroes,  # 对应上面的 path_heroes
            battle_path=path_battles,  # 对应上面的 path_battles
            anchors_dict=config.GROWTH_ANCHORS,  # 对应 config 中的字典
            save_dir=path_fig_dir  # 对应上面定义的 path_fig_dir
        )
        logger.info(f"第二章图表已生成至 {path_fig_dir}")
    except Exception as e:
        logger.error(f"生成EDA图表时发生错误: {e}")


def step_2_model_optimization():
    """
    【第三章至第六章：模型求解】
    计算最优参数、拟合曲线、分析稳定性
    """
    logger.info("-" * 30)
    logger.info(">>> STEP 2: 模型求解与优化 (Model Optimization)")

    # --- 3. 数值平衡 (Combat Balance) ---
    logger.info("[Ch3] 正在执行数值平衡回归与优化...")
    combat_model = combat_balance.CombatModel()

    # 加载日志并反推权重
    combat_model.load_data(os.path.join(config.DATA_RAW_DIR, "battle_logs.csv"))
    real_weights = combat_model.estimate_weights()

    # 保存识别出的权重
    with open(os.path.join(config.DATA_PROCESSED_DIR, "optimized_weights.json"), "w") as f:
        json.dump(real_weights, f)

    # 执行约束优化，生成平衡版英雄表
    corrected_heroes = combat_model.optimize_balance(
        heroes_path=os.path.join(config.DATA_RAW_DIR, "heroes.csv"),
        weights=real_weights
    )
    corrected_heroes.to_csv(os.path.join(config.DATA_PROCESSED_DIR, "corrected_heroes.csv"), index=False)
    logger.info("英雄属性修正完成，已保存至 processed/corrected_heroes.csv")

    # --- 4. 成长体验 (Growth Curve) ---
    logger.info("[Ch4] 正在拟合混合成长曲线...")
    growth_model = growth_curve.PiecewiseGrowthModel()

    # 使用配置中的锚点进行拟合
    growth_params = growth_model.fit_and_smooth(config.GROWTH_ANCHORS)

    with open(os.path.join(config.DATA_PROCESSED_DIR, "growth_curve_params.json"), "w") as f:
        json.dump(growth_params, f)
    logger.info(f"成长参数拟合完成: {growth_params}")

    # --- 5. 概率机制 (Gacha Probability) ---
    logger.info("[Ch5] 正在分析马尔可夫链稳态分布...")
    prob_model = gacha_prob.MarkovChainAnalyzer()
    exp_pulls, steady_state = prob_model.analyze_pity_system(
        base_prob=config.GACHA_BASE_PROB,
        pity_limit=config.GACHA_PITY_LIMIT
    )
    logger.info(f"理论期望抽数: {exp_pulls:.2f} (保底={config.GACHA_PITY_LIMIT})")

    # --- 6. 经济系统 (Economy Dynamics) ---
    logger.info("[Ch6] 正在分析经济系统稳定性...")
    eco_model = economy_dynamics.StabilityAnalyzer(
        alpha=config.ECO_CONVERSION_ALPHA,
        beta=config.ECO_CONVERSION_BETA
    )
    is_stable, steady_R = eco_model.check_stability()
    status = "稳定 (Stable)" if is_stable else "发散 (Unstable)"
    logger.info(f"经济系统状态判定: {status}, 稳态库存估算: {steady_R}")


def step_3_system_simulation():
    """
    【第七章：系统仿真】
    运行对照组与实验组
    """
    logger.info("-" * 30)
    logger.info(">>> STEP 3: 全系统耦合仿真 (System Simulation)")

    sim_engine = SimulationEngine(n_agents=config.SIM_AGENT_COUNT, days=config.SIM_DAYS)

    # 1. 运行对照组 (Control Group)
    # 特征：原始英雄数据(含超模)、无成长曲线(线性)、无保底(或纯随机)
    logger.info("正在运行【对照组】仿真 (Control Group)...")
    results_control = sim_engine.run(
        mode="control",
        heroes_file=os.path.join(config.DATA_RAW_DIR, "heroes.csv"),
        growth_params=None
    )
    results_control.to_csv(os.path.join(config.DATA_PROCESSED_DIR, "sim_results_control.csv"), index=False)

    # 2. 运行实验组 (Experimental Group)
    # 特征：修正后英雄数据、拟合的成长曲线、开启保底机制
    logger.info("正在运行【实验组】仿真 (Experimental Group)...")
    results_experiment = sim_engine.run(
        mode="experiment",
        heroes_file=os.path.join(config.DATA_PROCESSED_DIR, "corrected_heroes.csv"),
        growth_params=os.path.join(config.DATA_PROCESSED_DIR, "growth_curve_params.json")
    )
    results_experiment.to_csv(os.path.join(config.DATA_PROCESSED_DIR, "sim_results_experiment.csv"), index=False)

    logger.info("仿真结束，结果已保存。")


def step_4_visualization():
    """
    【结果可视化】
    生成论文所需的四张核心对比图
    """
    logger.info("-" * 30)
    logger.info(">>> STEP 4: 结果可视化 (Visualization)")

    # 路径准备
    path_control = os.path.join(config.DATA_PROCESSED_DIR, "sim_results_control.csv")
    path_experiment = os.path.join(config.DATA_PROCESSED_DIR, "sim_results_experiment.csv")
    path_growth_params = os.path.join(config.DATA_PROCESSED_DIR, "growth_curve_params.json")
    path_gacha_raw = os.path.join(config.DATA_RAW_DIR, "gacha_sequences.csv")

    # 1. 胜率分布 (公平性)
    visualization.plot_win_rate_comparison(
        path_control,
        path_experiment,
        save_path=os.path.join(config.OUTPUT_FIG_DIR, "fig_7_win_rate_comparison.png")
    )

    # 2. 经济趋势 (稳定性)
    visualization.plot_economy_trends(
        path_control,
        path_experiment,
        save_path=os.path.join(config.OUTPUT_FIG_DIR, "fig_7_economy_trends.png")
    )

    # 3. 成长曲线分析 (节奏)
    visualization.plot_growth_analysis(
        path_growth_params,
        save_path=os.path.join(config.OUTPUT_FIG_DIR, "fig_7_growth_analysis.png")
    )

    # 4. 抽卡保底验证 (风险)
    visualization.plot_gacha_pity_validation(
        path_gacha_raw,
        save_path=os.path.join(config.OUTPUT_FIG_DIR, "fig_7_gacha_validation.png")
    )

    logger.info(f"可视化完成，请查看 {config.OUTPUT_FIG_DIR} 目录。")


def main():
    """主程序"""
    logger.info("=== 启动基于多目标优化理论的游戏数值系统建模与仿真项目 ===")

    try:
        # 0. 环境初始化
        setup_directories()

        # 1. 数据生成
        step_1_data_generation()

        # 2. 模型优化
        step_2_model_optimization()

        # 3. 系统仿真
        step_3_system_simulation()

        # 4. 结果绘图
        step_4_visualization()

        logger.info("=== 项目全流程执行成功！ ===")

    except Exception as e:
        logger.error(f"运行时发生严重错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()