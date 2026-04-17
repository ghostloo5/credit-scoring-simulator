# Credit Scoring Simulator

A credit scoring simulator with end-to-end machine learning pipeline: data generation, feature engineering, model training, and risk evaluation.

## 📌 项目概述
这是一个模拟银行信用评分流程的端到端机器学习项目。项目从模拟客户数据开始，构建完整的特征工程与机器学习管道，最终训练一个信用风险评估模型，并提供了简易的交互界面。

## ✨ 主要功能
- **客户管理与模拟交易**：扩展的银行核心系统，支持创建账户、模拟存款、取款等交易，并记录行为数据。
- **结构化数据生成**：程序化生成包含多维度特征（年龄、收入、职业稳定性、交易习惯等）的模拟客户数据集。
- **端到端机器学习管道**：
  - 特征工程：从原始交易数据中构建风险相关特征。
  - 模型训练：使用随机森林算法训练信用评分模型。
  - 模型评估：提供AUC、精确率、召回率等详细评估指标与特征重要性分析。
- **信用风险评估**：输入客户特征，输出违约概率、风险等级（高/中/低）和信用评分。

## 🏗️ 系统架构
模拟数据生成 -> 特征抽取与工程 -> 模型训练与评估 -> 风险评估接口

↑               ↑               ↑               ↑

客户行为模拟      交易日志        随机森林模型    命令行交互界面
复制
## 🚀 如何运行
1.  **克隆仓库**
bash

git clone https://github.com/ghostloo5/credit-scoring-simulator.git

cd credit-scoring-simulator
复制
2.  **安装依赖**
bash

pip install -r requirements.txt
复制
3.  **运行主程序**
bash

python main.py
复制
*请根据您的主文件名运行，如果入口文件是其他名称（如 `credit_score.py`），请替换 `main.py`。*

## 📁 项目结构
credit-scoring-simulator/

├── main.py                 # 主程序入口（或您定义的其他入口文件名）

├── requirements.txt        # 项目依赖清单 (pandas, numpy, scikit-learn)

└── README.md              # 项目说明文件
复制
*如果您的代码包含多个模块文件，请在此处列出。*

## 🔧 技术栈
- **语言**: Python
- **核心库**: Pandas, NumPy, Scikit-learn
- **机器学习算法**: 随机森林 (Random Forest)
- **评估指标**: AUC-ROC, 分类报告, 特征重要性
- **版本控制**: Git & GitHub

## 📄 许可证
本项目基于 MIT 许可证开源。
