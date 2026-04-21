智能信用评分系统

基于机器学习的信用风险评估与决策支持平台。

🚀 在线体验

应用已通过 Streamlit Cloud 部署，点击此处体验：智能信用评分系统(https://credit-scoring-simulator-5dtywbn8wejjjqquhnp6ut.streamlit.app)

✨ 核心功能

• 📊 客户管理：手动创建客户档案或批量生成模拟数据，支持数据筛选、导出与备份。

• 🤖 模型训练：使用随机森林算法训练信用评分模型，可视化特征重要性，支持模型导出（.pkl格式）。

• 🔍 风险评估：输入客户特征，实时获取违约概率、信用评分、风险等级（低/中/高）及审批建议。

• 📈 数据分析：查看数据分布、特征相关性及模型性能（AUC）可视化报告。

🛠️ 技术栈

• 前端/应用框架: Streamlit

• 数据处理: Pandas, NumPy

• 机器学习: Scikit-learn (RandomForestClassifier)

• 数据可视化: Plotly

📦 本地运行

1. 克隆仓库

git clone https://github.com/GhostLoo/credit-scoring-simulator.git
cd credit-scoring-simulator


2. 安装依赖

pip install -r requirements.txt


3. 启动应用

streamlit run app.py


应用将在本地启动，默认在 http://localhost:8501(http://localhost:8501) 可访问。

📁 项目结构

credit-scoring-simulator/
├── app.py              # 主应用文件
├── requirements.txt    # Python 依赖包列表
└── README.md          # 项目说明文档


📄 许可证

本项目基于 MIT License(LICENSE) 开源。
