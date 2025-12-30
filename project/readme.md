# Project: Game Numerical System Modeling & Simulation

**基于多目标优化理论的游戏数值系统建模与仿真研究**

## 1. 项目概述 (Project Overview)

本项目旨在为游戏数值系统提供一套可解释、可验证的数学建模与仿真框架。项目包含四个核心子系统（对抗、成长、概率、经济），通过合成数据（Synthetic Data）进行统计推断与参数优化，并最终在多智能体仿真环境（Agent-based Simulation）中验证系统的长期稳定性与平衡性。

## 2. 目录结构 (Directory Structure)

```text
Game_Numerical_Modeling/
├── data/                        # 数据存储目录
│   ├── raw/                     # 原始生成数据 (CSV)
│   │   ├── heroes.csv           # 英雄基础属性 (Ch 2.1)
│   │   ├── battle_logs.csv      # 历史对局日志 (Ch 2.1)
│   │   └── gacha_sequences.csv  # 抽卡模拟序列 (Ch 2.2)
│   └── processed/               # 优化后的参数表 & 仿真结果
├── src/                         # 源代码目录
│   ├── __init__.py
│   ├── config.py                # 全局配置 (随机种子、常量、超参数)
│   │
│   ├── models/                  # 数学模型与求解器 (对应论文Ch3-6)
│   │   ├── __init__.py
│   │   ├── combat_balance.py    # 对抗平衡模型 (逻辑回归, QP优化)
│   │   ├── growth_curve.py      # 成长体验模型 (混合函数拟合, 断崖检测)
│   │   ├── gacha_prob.py        # 概率机制模型 (马尔可夫链, 蒙特卡洛)
│   │   └── economy_dynamics.py  # 经济系统模型 (微分方程求解, 稳定性分析)
│   │
│   ├── simulation/              # 全系统耦合仿真 (对应论文Ch7)
│   │   ├── __init__.py
│   │   ├── agent.py             # 智能体类 (玩家行为策略)
│   │   ├── server.py            # 虚拟服务器类 (处理资源流转与模块耦合)
│   │   └── engine.py            # 仿真主循环 (初始化->更新->记录)
│   │
│   └── utils/                   # 工具库
│       ├── data_generator.py    # 数据生成工厂 (对应论文Ch2)
│       └── visualization.py     # 绘图模块 (统一论文图表风格)
│
├── main.py                      # 项目主入口 (一键运行全流程)
├── requirements.txt             # 依赖库列表
└── readme.md                    # 项目说明文档

```

---

## 3. 核心模块说明 (Module Descriptions)

### 3.1 数据生成与预处理 (`src/utils/data_generator.py`)

* **对应章节**：第二章：数据生成与特征工程
* **功能**：
* `generate_heroes(n=100)`: 生成英雄属性，注入“超模”数据。
* `simulate_battles(n_battles=100000)`: 基于Sigmoid生成胜负日志，构造  和  特征。
* `simulate_gacha_logs(n_pulls=1e6)`: 生成百万级抽卡序列用于统计验证。



### 3.2 对抗平衡模型 (`src/models/combat_balance.py`)

* **对应章节**：第三章：数值平衡模型
* **核心类/函数**：
* `LogisticWeightEstimator`: 使用 `sklearn.linear_model.LogisticRegression` 从战斗日志中反推真实属性权重  (公式3-2)。
* `SensitivityAnalyzer`: 计算属性边际敏感度 (公式3-4)。
* `BalanceOptimizer`: 使用 `scipy.optimize.minimize` 求解带约束的二次规划问题 (公式3-5)，输出属性修正方案 。



### 3.3 成长体验模型 (`src/models/growth_curve.py`)

* **对应章节**：第四章：成长体验模型
* **核心类/函数**：
* `PiecewiseGrowthModel`: 定义“线性-指数-对数”混合函数结构 (公式4-1)。
* `CurveFitter`: 基于体验锚点 ，使用最小二乘法拟合曲线参数 (公式4-3)。
* `CliffDetector`: 计算一阶差分 ，检测并标记“断崖”等级 (公式4-4)。



### 3.4 概率机制模型 (`src/models/gacha_prob.py`)

* **对应章节**：第五章：概率机制模型
* **核心类/函数**：
* `MarkovChainAnalyzer`: 构建转移矩阵 ，解析计算期望抽数与稳态分布 (公式5-1, 5-2)。
* `MonteCarloSimulator`: 模拟对比“纯随机”与“保底机制”的方差与P95尾部风险。



### 3.5 经济系统模型 (`src/models/economy_dynamics.py`)

* **对应章节**：第六章：经济系统模型
* **核心类/函数**：
* `ResourceDynamics`: 定义三资源微分方程组 `dydt` (公式6-1)。
* `StabilityAnalyzer`: 求解稳态点 (Eq. 6-2) 并计算雅可比矩阵特征值以判断稳定性。
* `InflationController`: 实现动态税率与循环消耗的控制逻辑。



### 3.6 全系统耦合仿真 (`src/simulation/`)

* **对应章节**：第七章：全系统耦合仿真
* **核心类**：
* `Agent`: 拥有属性（活跃度、策略偏好、当前资源），执行 `decide_action()`（对抗/抽卡/强化）。
* `GameServer`: 管理全局状态，处理跨模块交互（如：抽卡结果更新Agent英雄池 -> 影响Agent对抗胜率 -> 影响Agent资源产出）。
* `SimulationEngine`: 运行 `run_experiment(config, optimized=True/False)`，对比优化前后系统指标。



---

## 4. 运行流程 (Workflow)

本项目设计为**流水线式**运行，通过 `main.py` 依次执行以下步骤：

1. **Step 1: Data Synthesis (数据合成)**
* 运行 `data_generator.py`，在 `data/raw/` 下生成初始数据。
* *产出*：`heroes.csv`, `battle_logs.csv`


2. **Step 2: Model Solving (模型求解)**
* 加载原始数据，运行 `combat_balance.py` 识别权重并计算修正量。
* 运行 `growth_curve.py` 拟合经验曲线。
* *产出*：`data/processed/` 下的参数文件。


3. **Step 3: System Simulation (系统仿真)**
* 初始化 1000 个 `Agent`。
* **Control Group (对照组)**: 使用原始参数运行 30 天仿真。
* **Experimental Group (实验组)**: 使用优化后参数运行 30 天仿真。
* *产出*：`simulation_results.csv`


4. **Step 4: Visualization (可视化)**
* 读取仿真结果，绘制论文所需的对比图（胜率分布对比、资源库存对比、断崖检测图等）。
* *产出*：保存图片至 `outputs/figures/`。



---

## 5. 依赖库 (Requirements)

* `numpy`, `pandas`: 数据处理
* `scipy`: 曲线拟合与优化求解
* `scikit-learn`: 逻辑回归分析
* `matplotlib`, `seaborn`: 绘图与可视化
* `tqdm`: 进度条显示

---

**Next Step Suggestion**:
您可以先创建上述文件夹结构，然后我可以为您提供 `src/config.py` 和 `src/utils/data_generator.py` 的具体代码，帮您完成第一步数据生成。