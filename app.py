import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import random
import pickle
import textwrap
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# 版本号
APP_VERSION = "0.0.1"

# -----------------------------
# 全局可视化主题
# -----------------------------
PLOTLY_DISCRETE_COLORS = [
    "#4F8EF7",  # 蓝色
    "#8A6FE8",  # 紫色
    "#34BFA3",  # 绿色
    "#F2A65A",  # 橙色
    "#E57373",  # 红色
    "#4FC3D9",  # 青色
    "#7A8CA5",  # 灰蓝
    "#9AA5B5",  # 灰色
]

# 连续色阶
PLOTLY_CONTINUOUS_SCALE = [
    [0.0, "#D6E6FF"],
    [0.25, "#B7D2FF"],
    [0.5, "#95BCFF"],
    [0.75, "#709FEB"],
    [1.0, "#4F84D6"],
]

# 统一模板
PLOTLY_TEMPLATE = "plotly_white"
px.defaults.template = PLOTLY_TEMPLATE
px.defaults.color_discrete_sequence = PLOTLY_DISCRETE_COLORS

# Plotly配置
PLOTLY_CONFIG = {
    "responsive": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "displayModeBar": False,
    "scrollZoom": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "chart",
        "height": 500,
        "width": 700,
        "scale": 2
    }
}

# 同级图表统一布局参数（保证齐平/不裁切）
CHART_PAIR_HEIGHT = 470
CHART_MARGIN_COMPACT = dict(t=56, b=36, l=18, r=48)
CHART_TITLE_FONT = dict(size=20, color="#333333", family="Arial, sans-serif")

# 设置页面
st.set_page_config(
    page_title="智能信用评分系统",
    page_icon="🏦",
    layout="wide"
)

def _scroll_to_top_js() -> str:
    return r"""
<script>
(function () {
  function scrollAllToTop(doc, win) {
    try { win.scrollTo(0, 0); } catch (e) {}
    try { doc.documentElement.scrollTop = 0; } catch (e) {}
    try { doc.body.scrollTop = 0; } catch (e) {}
    
    const candidates = new Set();
    [
      'html',
      'body',
      'div[data-testid="stAppViewContainer"]',
      'section[data-testid="stMain"]',
      'div[data-testid="stMain"]',
      '.main',
      'div.block-container'
    ].forEach((sel) => {
      doc.querySelectorAll(sel).forEach((el) => candidates.add(el));
    });

    const all = Array.from(doc.querySelectorAll('div, section, main')).slice(0, 250);
    all.forEach((el) => {
      try {
        const style = win.getComputedStyle(el);
        const overflowY = style.overflowY;
        const canScroll = (overflowY === 'auto' || overflowY === 'scroll') && el.scrollHeight > el.clientHeight;
        if (canScroll) candidates.add(el);
      } catch (e) {}
    });

    candidates.forEach((el) => {
      try { el.scrollTop = 0; } catch (e) {}
    });
  }

  let tries = 0;
  function scrollWithRetries() {
    const doc = (window.parent && window.parent.document) ? window.parent.document : document;
    const win = (window.parent && window.parent.window) ? window.parent.window : window;
    scrollAllToTop(doc, win);
    tries += 1;
    if (tries < 3) {
      requestAnimationFrame(scrollWithRetries);
    } else {
      setTimeout(() => scrollAllToTop(doc, win), 150);
      setTimeout(() => scrollAllToTop(doc, win), 450);
    }
  }

  scrollWithRetries();
})();
</script>
"""

def scroll_to_top():
    nonce = st.session_state.get("_scroll_nonce", 0)
    components.html(f"<!--scroll:{nonce}-->{_scroll_to_top_js()}", height=0, width=0)

# 专业风格CSS
st.markdown("""
<style>
    :root {
        --bg: #F5F5F7;
        --card: #FFFFFF;
        --text: #1D1D1F;
        --muted: #6E6E73;
        --line: rgba(0,0,0,0.08);
        --blue: #0A84FF;
        --blueHover: #007AFF;
        --btnBg: rgba(10,132,255,0.12);
        --btnBgHover: rgba(10,132,255,0.18);
        --shadow: 0 8px 30px rgba(0,0,0,0.06);
        --card-height-unified: 160px;
        --btn-height-unified: 40px;
        --hint-success-fg: #166534;
        --hint-success-bg: rgba(34, 197, 94, 0.16);
        --hint-success-border: rgba(34, 197, 94, 0.38);
        --hint-warn-fg: #B45309;
        --hint-warn-bg: rgba(245, 158, 11, 0.20);
        --hint-warn-border: rgba(245, 158, 11, 0.45);
    }

    [data-testid="stAppViewContainer"] {
        background: var(--bg);
    }
    
    /* 隐藏 Streamlit 顶栏（含 Deploy 等英文入口） */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    [data-testid="stToolbar"] {
        display: none !important;
    }
    #MainMenu {
        visibility: hidden;
        height: 0;
    }

    * {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: var(--text);
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 5.5rem;
        max-width: 1200px;
    }

    /* 主标题上方空行 */
    h1 {
        margin-top: 0.8em !important;
        margin-bottom: 0.3em !important;
    }

    h2, h3 {
        letter-spacing: -0.02em;
    }

    .stButton > button {
        background: linear-gradient(135deg, #43E97B 0%, #38F9D7 100%);
        color: #ffffff !important;
        border: 1px solid rgba(56, 185, 143, 0.45);
        border-radius: 10px;
        padding: 0.58rem 1rem;
        min-height: var(--btn-height-unified);
        font-weight: 600;
        transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
        box-shadow: 0 8px 20px rgba(52, 191, 163, 0.25);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #3EDB72 0%, #31E3C3 100%);
        transform: translateY(-1px);
        box-shadow: 0 10px 24px rgba(52, 191, 163, 0.32);
    }

    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #43E97B 0%, #38F9D7 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(56, 185, 143, 0.45) !important;
        border-radius: 10px;
        min-height: var(--btn-height-unified);
        box-shadow: 0 8px 20px rgba(52, 191, 163, 0.25);
    }

    .stFormSubmitButton > button:hover {
        background: linear-gradient(135deg, #3EDB72 0%, #31E3C3 100%) !important;
        box-shadow: 0 10px 24px rgba(52, 191, 163, 0.32);
    }

    button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #43E97B 0%, #38F9D7 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(56, 185, 143, 0.45) !important;
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #43E97B 0%, #38F9D7 100%);
        color: #ffffff !important;
        border: 1px solid rgba(56, 185, 143, 0.45);
        border-radius: 10px;
        min-height: var(--btn-height-unified);
        padding: 0.58rem 1rem;
        box-shadow: 0 8px 20px rgba(52, 191, 163, 0.25);
    }

    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #3EDB72 0%, #31E3C3 100%);
        box-shadow: 0 10px 24px rgba(52, 191, 163, 0.32);
    }

    [data-testid="stMetric"],
    [data-testid="stPlotlyChart"],
    .stRadio > div {
        transition: transform 0.18s ease, box-shadow 0.2s ease, opacity 0.2s ease;
    }
    [data-testid="stMetric"]:hover,
    [data-testid="stPlotlyChart"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 26px rgba(0,0,0,0.10);
    }

    .stTextInput input, .stNumberInput input, .stSelectbox select {
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 10px 12px;
        background: var(--card);
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }

    .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }

    [data-testid="stMetric"] {
        background: var(--card);
        border-radius: 14px;
        padding: 0.95rem 1rem;
        border: 1px solid var(--line);
        box-shadow: var(--shadow);
        min-height: var(--card-height-unified);
        height: var(--card-height-unified);
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    [data-testid="stMetricLabel"] p {
        color: #6B7280 !important;
        font-size: 0.88rem !important;
        margin-bottom: 0.42rem !important;
        font-weight: 500 !important;
    }
    [data-testid="stMetricValue"] {
        color: #111827 !important;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
    }
    [data-testid="stMetricDelta"] {
        margin-top: 0.48rem !important;
        font-size: 0.92rem !important;
        font-weight: 700 !important;
        color: var(--hint-success-fg) !important;
        background: var(--hint-success-bg) !important;
        border: 1px solid var(--hint-success-border) !important;
        border-radius: 999px !important;
        padding: 0.3rem 0.62rem !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 0 !important;
        width: fit-content !important;
    }
    [data-testid="stMetricDelta"] * {
        color: inherit !important;
    }
    [data-testid="stMetricDelta"] svg {
        display: none !important;
    }
    .status-card {
        background: var(--card);
        border-radius: 14px;
        padding: 0.95rem 1rem;
        border: 1px solid var(--line);
        box-shadow: var(--shadow);
        min-height: var(--card-height-unified);
        height: var(--card-height-unified);
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .status-card__label {
        color: #6B7280;
        font-size: 0.88rem;
        margin-bottom: 0.42rem;
    }
    .status-card__value {
        color: #111827;
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin-bottom: 0.5rem;
    }
    .status-card__hint {
        display: inline-flex;
        align-items: center;
        gap: 0.38rem;
        font-size: 0.92rem;
        font-weight: 700;
        width: fit-content;
        padding: 0.3rem 0.62rem;
        border-radius: 999px;
    }
    .status-card__hint--success {
        color: var(--hint-success-fg);
        background: var(--hint-success-bg);
        border: 1px solid var(--hint-success-border);
    }
    .status-card__hint--warn {
        color: var(--hint-warn-fg);
        background: var(--hint-warn-bg);
        border: 1px solid var(--hint-warn-border);
    }

    /* 系统概览流程卡统一尺寸 */
    .overview-flow-card {
        min-height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: rgba(255,255,255,0.65);
        border: none;
        border-radius: 12px;
        padding: 4px;
        width: fit-content;
        box-shadow: none;
    }

    /* 彻底移除 Tabs 区域下方横线（含默认高亮条/分割线） */
    .stTabs [data-baseweb="tab-border"],
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tabs"],
    .stTabs [data-baseweb="tabs"] > div,
    .stTabs [role="tablist"],
    .stTabs [role="tablist"]::before,
    .stTabs [role="tablist"]::after,
    .stTabs [data-baseweb="tab-list"]::before,
    .stTabs [data-baseweb="tab-list"]::after {
        border: none !important;
        border-bottom: none !important;
        box-shadow: none !important;
        background-image: none !important;
    }

    .stTabs [role="tab"]::before,
    .stTabs [role="tab"]::after,
    .stTabs [aria-selected="true"]::before,
    .stTabs [aria-selected="true"]::after {
        border: none !important;
        border-bottom: none !important;
        box-shadow: none !important;
        background: transparent !important;
        background-image: none !important;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.46rem 0.9rem;
        font-weight: 600;
        color: var(--muted);
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--card);
        color: var(--text);
        border: none;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    }

    /* 侧边栏导航专业化样式 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FCFCFD 0%, #F8FAFC 100%);
        border-right: 1px solid var(--line);
        min-width: 220px !important;
        max-width: 220px !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    [data-testid="stSidebar"] .block-container {
        width: 100%;
        max-width: 190px;
        padding-left: 0.25rem;
        padding-right: 0.25rem;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #111827 !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
        text-align: center;
    }
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 0.35rem;
        width: 100%;
    }
    [data-testid="stSidebar"] [role="radiogroup"] > label {
        border: 1px solid transparent;
        border-radius: 10px;
        padding: 0.5rem 0.6rem;
        transition: all 0.18s ease;
        background: transparent;
    }
    [data-testid="stSidebar"] [role="radiogroup"] > label:hover {
        background: rgba(10,132,255,0.08);
        border-color: rgba(10,132,255,0.18);
    }
    [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
        background: rgba(10,132,255,0.14);
        border-color: rgba(10,132,255,0.28);
        box-shadow: 0 6px 16px rgba(10,132,255,0.12);
    }
    [data-testid="stSidebar"] [role="radiogroup"] > label > div {
        font-weight: 600;
        color: #1F2937;
    }

    .stDataFrame {
        border: 1px solid var(--line);
        border-radius: 14px;
        overflow: hidden;
        background: var(--card);
    }

    .app-footer {
        position: fixed;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: 1000;
        background: rgba(245,245,247,0.92);
        backdrop-filter: blur(12px);
        border-top: 1px solid var(--line);
        padding: 0.85rem 0;
    }

    .stAlert {
        border-radius: 14px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# 侧边栏 - 导航
st.sidebar.title("导航菜单")

# 存储当前页面状态
if 'current_page' not in st.session_state:
    st.session_state.current_page = "系统概览"

# 创建单选按钮并获取选择
page = st.sidebar.radio(
    "选择工作模块：",
    ["系统概览", "客户管理", "模型训练", "风险评估", "数据分析"],
    key="page_selector",
    label_visibility="collapsed"
)

# 页面切换时：强制回到顶部
if st.session_state.current_page != page:
    st.session_state.current_page = page
    st.session_state["_scroll_nonce"] = st.session_state.get("_scroll_nonce", 0) + 1
    scroll_to_top()
else:
    # 保留当前滚动位置，减少每次交互触发的脚本开销与跳动
    pass

# 初始化数据存储
if 'clients' not in st.session_state:
    st.session_state.clients = []
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'assessment_history' not in st.session_state:
    st.session_state.assessment_history = []

# 1. 系统概览页面
if page == "系统概览":
    st.header("系统概览")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        client_count = len(st.session_state.clients)
        st.metric("客户总数", client_count, 
                 delta=f"+{client_count}" if client_count > 0 else None,
                 help="系统中已建立的客户档案数量")
    
    with col2:
        if st.session_state.trained_model:
            st.markdown("""
            <div class="status-card">
                <div class="status-card__label">模型状态</div>
                <div class="status-card__value">已部署</div>
                <div class="status-card__hint status-card__hint--success">
                    运行中
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-card">
                <div class="status-card__label">模型状态</div>
                <div class="status-card__value">未训练</div>
                <div class="status-card__hint status-card__hint--warn">
                    需初始化
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if st.session_state.model_metrics:
            auc = st.session_state.model_metrics.get('auc', 0)
            st.metric("模型性能", f"AUC: {auc:.3f}",
                     help="模型区分客户风险的能力指标")
        else:
            st.metric("模型性能", "待评估")
    
    with col4:
        if st.session_state.clients:
            avg_income = np.mean([c.get('income', 0) for c in st.session_state.clients])
            st.metric("平均收入", f"{avg_income:,.0f}元")
        else:
            st.metric("平均收入", "待录入")
    
    st.markdown("---")
    
    # 工作流程图示
    st.subheader("工作流程概览")
    
    workflow_cols = st.columns(3)
    with workflow_cols[0]:
        st.markdown("""
        <div class='overview-flow-card' style='background: linear-gradient(135deg, #8FA7FF 0%, #A78BFA 100%); color: white; padding: 1.5rem; border-radius: 10px;'>
            <h4 style='margin: 0 0 1rem 0; color: white;'>1. 数据准备</h4>
            <ul style='margin: 0; padding-left: 1rem; color: white;'>
                <li style='color: white;'>客户信息录入</li>
                <li style='color: white;'>数据质量控制</li>
                <li style='color: white;'>特征工程处理</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with workflow_cols[1]:
        st.markdown("""
        <div class='overview-flow-card' style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 1.5rem; border-radius: 10px;'>
            <h4 style='margin: 0 0 1rem 0; color: white;'>2. 模型构建</h4>
            <ul style='margin: 0; padding-left: 1rem; color: white;'>
                <li style='color: white;'>算法选择优化</li>
                <li style='color: white;'>模型训练验证</li>
                <li style='color: white;'>性能评估调优</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with workflow_cols[2]:
        st.markdown("""
        <div class='overview-flow-card' style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white; padding: 1.5rem; border-radius: 10px;'>
            <h4 style='margin: 0 0 1rem 0; color: white;'>3. 决策支持</h4>
            <ul style='margin: 0; padding-left: 1rem; color: white;'>
                <li style='color: white;'>实时风险评估</li>
                <li style='color: white;'>额度策略制定</li>
                <li style='color: white;'>报告生成输出</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # 系统状态指示
    st.markdown("---")
    st.subheader("系统状态指示")
    
    if not st.session_state.clients:
        st.warning("""
        ⚠️ **当前无客户数据**
        
        请先前往 **客户管理** 模块创建客户档案或生成训练数据。
        """)
    elif not st.session_state.trained_model:
        st.info("""
        ℹ️ **模型未训练**
        
        已有客户数据，但尚未训练信用评分模型。请前往 **模型训练** 模块初始化模型。
        """)
    else:
        st.success("""
        ✅ **系统运行正常**
        
        所有模块准备就绪，可进行风险评估和数据分析。
        """)

# 2. 客户管理页面
elif page == "客户管理":
    st.header("客户管理")
    st.markdown("<p style='color: #666;'>建立和维护客户信息档案，支持信用模型训练与评估</p>", unsafe_allow_html=True)
    
    customer_tab = st.radio(
        "客户管理功能",
        ["新增客户", "客户列表", "数据操作", "批量生成"],
        horizontal=True,
        label_visibility="collapsed",
        key="customer_tab_selector"
    )
    
    if customer_tab == "新增客户":
        st.subheader("新增客户档案")
        
        with st.form("create_customer", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("客户姓名*", placeholder="请输入客户全名")
                age = st.slider("年龄*", 18, 70, 35)
                income = st.number_input("年收入（元）*", 20000, 500000, 60000, 5000)
                education = st.selectbox("教育程度", ["高中及以下", "专科", "本科", "硕士", "博士"])
            
            with col2:
                balance = st.number_input("账户余额（元）", 0, 1000000, 50000, 1000)
                job_years = st.slider("工作年限", 0, 40, 5)
                credit_cards = st.slider("信用卡数量", 0, 10, 2)
                credit_score = st.slider("信用评分", 300, 850, 650)
            
            # 贷款信息
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            st.markdown("**贷款信息**")
            col_a, col_b = st.columns(2)
            with col_a:
                has_mortgage = st.checkbox("有房贷记录")
            with col_b:
                has_car_loan = st.checkbox("有车贷记录")
            
            # 信用历史
            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
            default_history = st.radio("历史信用记录", ["无违约", "曾有逾期", "多次违约"], horizontal=True)
            
            submitted = st.form_submit_button("保存客户档案", use_container_width=True)
            
            if submitted:
                if not name:
                    st.error("请填写客户姓名")
                else:
                    # 将违约记录转化为数值标签
                    default_map = {"无违约": 0, "曾有逾期": 1, "多次违约": 1}
                    
                    new_client = {
                        'id': len(st.session_state.clients) + 10001,
                        'name': name,
                        'age': age,
                        'income': income,
                        'balance': balance,
                        'job_years': job_years,
                        'education': education,
                        'has_mortgage': 1 if has_mortgage else 0,
                        'has_car_loan': 1 if has_car_loan else 0,
                        'credit_cards': credit_cards,
                        'credit_score': credit_score,
                        'default_history': default_history,
                        'default_label': default_map[default_history],
                        'created_date': datetime.now().strftime('%Y-%m-%d %H:%M')
                    }
                    
                    st.session_state.clients.append(new_client)
                    st.success(f"✅ 客户档案已保存: {name}")
                    st.info(f"档案编号: {new_client['id']}")
    
    elif customer_tab == "客户列表":
        st.subheader("客户档案列表")
        
        if st.session_state.clients:
            df = pd.DataFrame(st.session_state.clients)
            st.markdown(f"**已建立 {len(df)} 份客户档案**")
            
            # 高级筛选
            with st.expander("高级筛选选项"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    min_income = st.number_input("最低收入", 0, 500000, 0, 10000)
                with col2:
                    max_age = st.number_input("最大年龄", 18, 100, 100)
                with col3:
                    show_defaults = st.selectbox("违约记录", ["全部", "有违约", "无违约"])
            
            filtered_df = df.copy()
            if min_income > 0:
                filtered_df = filtered_df[filtered_df['income'] >= min_income]
            if max_age < 100:
                filtered_df = filtered_df[filtered_df['age'] <= max_age]
            if show_defaults == "有违约":
                filtered_df = filtered_df[filtered_df['default_label'] == 1]
            elif show_defaults == "无违约":
                filtered_df = filtered_df[filtered_df['default_label'] == 0]
            
            # 显示数据
            display_cols = ['id', 'name', 'age', 'income', 'balance', 'default_history', 'created_date']
            st.dataframe(filtered_df[display_cols], use_container_width=True, height=400)
            
            # 统计摘要
            st.markdown("---")
            st.subheader("档案统计摘要")
            
            if not filtered_df.empty:
                stats_cols = st.columns(4)
                with stats_cols[0]:
                    st.metric("平均年龄", f"{filtered_df['age'].mean():.1f}岁")
                with stats_cols[1]:
                    st.metric("平均收入", f"{filtered_df['income'].mean():,.0f}元")
                with stats_cols[2]:
                    default_rate = filtered_df['default_label'].mean() * 100
                    st.metric("违约率", f"{default_rate:.1f}%")
                with stats_cols[3]:
                    st.metric("档案数量", len(filtered_df))
        else:
            st.info("暂无客户档案，请先创建或生成数据")
    
    elif customer_tab == "数据操作":
        st.subheader("数据操作")
        
        if st.session_state.clients:
            df = pd.DataFrame(st.session_state.clients)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**导出数据**")
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="导出CSV",
                    data=csv,
                    file_name=f"客户档案_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                st.markdown("**备份数据**")
                if st.button("创建数据备份", use_container_width=True):
                    st.session_state.backup_clients = st.session_state.clients.copy()
                    st.success("数据备份已创建")
            
            with col3:
                st.markdown("**数据管理**")
                if st.button("重置系统数据", type="secondary", use_container_width=True):
                    st.session_state.clients = []
                    st.session_state.trained_model = None
                    st.session_state.model_metrics = {}
                    st.success("系统数据已重置")
                    st.rerun()
        else:
            st.info("暂无数据可操作")
    
    elif customer_tab == "批量生成":
        st.subheader("批量数据生成")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_samples = st.slider("生成数量", 50, 2000, 500, 50)
        
        with col2:
            default_rate = st.slider("违约率设定", 0.05, 0.5, 0.15, 0.05)
        
        if st.button("执行批量生成", type="primary", use_container_width=True):
            with st.spinner(f"正在生成 {num_samples} 条模拟数据..."):
                new_clients = []
                
                for i in range(num_samples):
                    # 生成合理数据
                    age = random.randint(22, 65)
                    income = int(np.random.lognormal(10.5, 0.6))  # 对数正态分布
                    income = max(20000, min(500000, income))
                    balance = int(income * random.uniform(0.1, 2))
                    
                    # 违约概率计算
                    risk_score = (
                        (age < 25) * 0.2 +
                        (income < 40000) * 0.3 +
                        (balance < 10000) * 0.2
                    )
                    
                    will_default = 1 if random.random() < (default_rate + risk_score * 0.2) else 0
                    
                    # 违约客户特征调整
                    if will_default:
                        income = int(income * random.uniform(0.7, 0.9))
                        balance = int(balance * random.uniform(0.3, 0.7))
                    
                    client = {
                        'id': 20000 + len(st.session_state.clients) + i,
                        'name': f"模拟客户_{i+1:04d}",
                        'age': age,
                        'income': income,
                        'balance': balance,
                        'job_years': random.randint(0, 30),
                        'education': random.choices(["高中及以下", "专科", "本科", "硕士"], weights=[0.3, 0.3, 0.3, 0.1])[0],
                        'has_mortgage': 1 if random.random() < 0.3 else 0,
                        'has_car_loan': 1 if random.random() < 0.2 else 0,
                        'credit_cards': random.randint(0, 8),
                        'credit_score': random.randint(400, 800),
                        'default_history': "曾有逾期" if will_default else "无违约",
                        'default_label': will_default,
                        'created_date': datetime.now().strftime('%Y-%m-%d %H:%M')
                    }
                    new_clients.append(client)
                
                st.session_state.clients.extend(new_clients)
                st.success(f"✅ 成功生成 {num_samples} 条数据")
                
                # 统计展示
                df_new = pd.DataFrame(new_clients)
                st.markdown("---")
                st.subheader("生成数据统计")
                
                metrics_cols = st.columns(4)
                with metrics_cols[0]:
                    default_rate_actual = df_new['default_label'].mean() * 100
                    st.metric("违约率", f"{default_rate_actual:.1f}%")
                with metrics_cols[1]:
                    st.metric("平均收入", f"{df_new['income'].mean():,.0f}元")
                with metrics_cols[2]:
                    st.metric("平均年龄", f"{df_new['age'].mean():.1f}岁")
                with metrics_cols[3]:
                    st.metric("总数据量", len(st.session_state.clients))

# 3. 模型训练页面
elif page == "模型训练":
    st.header("模型训练")
    st.markdown("<p style='color: #666;'>构建和优化信用评分预测模型</p>", unsafe_allow_html=True)
    
    if not st.session_state.clients:
        st.warning("""
        ⚠️ **训练数据不足**
        
        请先创建客户数据或生成模拟数据，至少需要20条记录。
        """)
        st.stop()
    
    df = pd.DataFrame(st.session_state.clients)
    st.markdown(f"**可用训练数据: {len(df)} 条记录**")
    
    # 数据预览
    with st.expander("查看数据样本"):
        st.dataframe(df[['name', 'age', 'income', 'default_history']].head(), use_container_width=True)
    
    # 特征选择
    st.subheader("特征工程")
    
    feature_options = {
        'age': '年龄',
        'income': '年收入',
        'balance': '账户余额',
        'job_years': '工作年限',
        'credit_cards': '信用卡数量',
        'credit_score': '信用评分',
        'has_mortgage': '房贷记录',
        'has_car_loan': '车贷记录'
    }
    
    selected_features = []
    cols = st.columns(3)
    for idx, (feature, desc) in enumerate(feature_options.items()):
        with cols[idx % 3]:
            if st.checkbox(f"{desc}", value=True, key=f"feat_{feature}"):
                selected_features.append(feature)
    
    if len(selected_features) < 2:
        st.error("请至少选择2个特征进行模型训练")
        st.stop()
    
    # 训练参数
    st.subheader("训练参数配置")
    
    param_cols = st.columns(3)
    with param_cols[0]:
        test_size = st.slider(
            "测试集比例",
            0.1,
            0.5,
            0.2,
            0.05,
            help="用于验证模型效果的数据占比。数据较少时建议 0.2，样本较多可提高到 0.3。"
        )
    with param_cols[1]:
        n_estimators = st.slider(
            "决策树数量",
            50,
            300,
            100,
            10,
            help="随机森林中树的数量。数量越大通常越稳定，但训练更慢；建议 100-200。"
        )
    with param_cols[2]:
        random_state = st.number_input(
            "随机种子",
            0,
            100,
            42,
            help="控制训练过程可复现。固定同一随机种子，可重复得到一致结果。"
        )
    
    # 开始训练
    if st.button("开始模型训练", type="primary", use_container_width=True):
        with st.spinner("正在进行模型训练..."):
            # 准备数据
            X = df[selected_features]
            y = df['default_label']
            
            # 检查数据平衡性
            if len(y[y==1]) < 3:
                st.error("违约样本过少，无法有效训练模型")
                st.stop()
            
            # 划分数据集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # 训练模型
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                class_weight='balanced',
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # 评估模型
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # 保存结果
            st.session_state.trained_model = model
            st.session_state.model_metrics = {
                'auc': auc,
                'test_size': len(X_test),
                'train_size': len(X_train),
                'features': selected_features
            }
            
            # 特征重要性
            feature_importance_df = pd.DataFrame({
                'feature': selected_features,
                'importance': model.feature_importances_,
                'description': [feature_options[f] for f in selected_features]
            }).sort_values('importance', ascending=False)
            
            st.session_state.feature_importance = feature_importance_df
            
            st.success("✅ 模型训练完成")
    
    # 显示训练结果
    if st.session_state.trained_model:
        st.markdown("---")
        st.subheader("模型评估结果")
        
        metrics = st.session_state.model_metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("AUC分数", f"{metrics['auc']:.4f}")
        with col2:
            accuracy = np.mean(y_pred == y_test) if 'y_pred' in locals() else 0
            st.metric("准确率", f"{accuracy:.2%}")
        with col3:
            st.metric("训练样本", f"{metrics['train_size']:,}")
        with col4:
            st.metric("测试样本", f"{metrics['test_size']:,}")
        
        # 特征重要性
        st.subheader("特征重要性分析")
        
        if st.session_state.feature_importance is not None:
            fig = px.bar(
                st.session_state.feature_importance,
                x='importance',
                y='description',
                orientation='h',
                title='特征重要性排名',
                color='importance',
                color_continuous_scale=PLOTLY_CONTINUOUS_SCALE,
                text='importance',
                labels={"importance": "重要性", "description": "特征"}
            )
            
            fig.update_traces(
                texttemplate='%{text:.3f}',
                textposition='outside',
                marker_line_color='white',
                marker_line_width=1.5,
                cliponaxis=False
            )
            
            fig.update_layout(
                xaxis_range=[0, 0.5],
                height=450,
                plot_bgcolor='rgba(248, 249, 250, 0.5)',
                paper_bgcolor='rgba(248, 249, 250, 0.5)',
                font=dict(family="Arial, sans-serif", size=12, color="#333333"),
                margin=dict(t=50, b=30, l=10, r=10)
            )
            
            fig.update_xaxes(
                title_text="重要性",
                title_font=dict(size=14, color="#666666"),
                gridcolor='rgba(0,0,0,0.1)',
                zerolinecolor='rgba(0,0,0,0.1)',
                automargin=True
            )
            
            fig.update_yaxes(
                title_text="特征",
                title_font=dict(size=14, color="#666666"),
                categoryorder='total ascending',
                automargin=True
            )
            
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        
        # 模型部署选项
        st.markdown("---")
        st.subheader("模型部署")
        
        if st.button("导出模型文件"):
            model_bytes = pickle.dumps(st.session_state.trained_model)
            st.download_button(
                label="下载模型 (.pkl)",
                data=model_bytes,
                file_name=f"credit_model_v{datetime.now().strftime('%Y%m%d')}.pkl",
                mime="application/octet-stream"
            )

# 4. 风险评估页面
elif page == "风险评估":
    st.header("风险评估")
    st.markdown("<p style='color: #666;'>基于模型对客户信用风险进行评估和决策</p>", unsafe_allow_html=True)
    
    if not st.session_state.trained_model:
        st.warning("""
        ⚠️ **模型未就绪**
        
        请先完成模型训练，以进行风险评估。
        """)
        st.stop()
    
    # 两列布局
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("客户信息输入")
        
        with st.form("risk_assessment_form"):
            used_features = st.session_state.model_metrics.get('features', [])
            
            input_data = {}
            
            # 基本信息
            st.markdown("**基本信息**")
            col1a, col1b = st.columns(2)
            with col1a:
                input_data['age'] = st.slider("年龄", 20, 70, 35, 1, key="input_age")
            with col1b:
                input_data['income'] = st.number_input("年收入（元）", 20000, 500000, 60000, 5000, key="input_income")
            
            # 职业与资产
            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
            st.markdown("**职业与资产**")
            col2a, col2b = st.columns(2)
            with col2a:
                input_data['job_years'] = st.slider("工作年限", 0, 40, 5, 1, key="input_job_years")
            with col2b:
                input_data['balance'] = st.number_input("账户余额（元）", 0, 1000000, 50000, 1000, key="input_balance")
            
            # 信用信息
            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
            st.markdown("**信用信息**")
            col3a, col3b = st.columns(2)
            with col3a:
                input_data['credit_cards'] = st.slider("信用卡数量", 0, 10, 2, 1, key="input_credit_cards")
            with col3b:
                input_data['credit_score'] = st.slider("信用评分", 300, 850, 650, 10, key="input_credit_score")
            
            # 贷款记录
            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
            st.markdown("**贷款记录**")
            col4a, col4b = st.columns(2)
            with col4a:
                input_data['has_mortgage'] = 1 if st.checkbox("有房贷记录", key="input_mortgage") else 0
            with col4b:
                input_data['has_car_loan'] = 1 if st.checkbox("有车贷记录", key="input_car_loan") else 0
            
            submitted = st.form_submit_button("执行风险评估", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("评估结果")
        
        if submitted:
            with st.spinner("正在计算风险等级..."):
                # 模型预测
                X_input = pd.DataFrame([input_data])[used_features]
                model = st.session_state.trained_model
                prediction_proba = model.predict_proba(X_input)[0]
                
                # 违约概率
                default_prob = prediction_proba[1]
                
                # 信用评分
                credit_score = 850 - (default_prob * 550)
                credit_score = max(300, min(850, credit_score))
                
                # 风险等级判定
                if default_prob < 0.1:
                    risk_level = "低风险"
                    risk_color = "#10B981"  # 绿色
                    recommendation = "✅ 建议批准"
                elif default_prob < 0.25:
                    risk_level = "中低风险"
                    risk_color = "#3B82F6"  # 蓝色
                    recommendation = "👍 正常审批"
                elif default_prob < 0.4:
                    risk_level = "中等风险"
                    risk_color = "#F59E0B"  # 橙色
                    recommendation = "⚠️ 审慎审批"
                elif default_prob < 0.6:
                    risk_level = "中高风险"
                    risk_color = "#EF4444"  # 红色
                    recommendation = "🔍 严格审批"
                else:
                    risk_level = "高风险"
                    risk_color = "#7F1D1D"  # 深红色
                    recommendation = "❌ 建议拒绝"
                
                # 修复的HTML结果展示
                recommendation_label = recommendation.split(" ", 1)[-1] if " " in recommendation else recommendation
                risk_index = default_prob * 100
                result_html = textwrap.dedent(f"""
                <style>
                    .risk-num {{
                        font-family: "SF Pro Display", "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                        font-variant-numeric: tabular-nums;
                        font-feature-settings: "tnum" 1, "ss01" 1, "cv11" 1;
                        letter-spacing: 0.014em;
                    }}
                </style>
                <div style='background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%); border-radius: 16px; padding: 1.4rem; margin-bottom: 1rem; border: 1px solid #E5E7EB; box-shadow: 0 14px 28px rgba(15, 23, 42, 0.08);'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;'>
                        <div style='font-size: 1.05rem; color: #111827; letter-spacing: 0.02em; font-weight: 700;'>信用风控评估</div>
                        <div style='background: {risk_color}; color: white; padding: 0.42rem 0.9rem; border-radius: 999px; font-weight: 700; font-size: 0.88rem;'>{risk_level}</div>
                    </div>

                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; margin-bottom: 1rem;'>
                        <div style='background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 0.9rem;'>
                            <div style='color: #64748B; font-size: 0.82rem; margin-bottom: 0.35rem;'>违约概率</div>
                            <div class='risk-num' style='font-size: 1.68rem; line-height: 1.08; font-weight: 650; background: linear-gradient(140deg, #0B1324 0%, #26324A 52%, #42506C 100%); -webkit-background-clip: text; background-clip: text; color: transparent; text-rendering: geometricPrecision;'>{default_prob:.2%}</div>
                        </div>
                        <div style='background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 0.9rem;'>
                            <div style='color: #64748B; font-size: 0.82rem; margin-bottom: 0.35rem;'>信用评分</div>
                            <div class='risk-num' style='font-size: 1.68rem; line-height: 1.08; font-weight: 650; background: linear-gradient(140deg, #0B1324 0%, #26324A 52%, #42506C 100%); -webkit-background-clip: text; background-clip: text; color: transparent; text-rendering: geometricPrecision;'>{credit_score:.0f}</div>
                        </div>
                    </div>

                    <div style='margin: 0.9rem 0 0.85rem 0;'>
                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.45rem;'>
                            <div style='color: #64748B; font-size: 0.82rem;'>风险指数</div>
                            <div class='risk-num' style='color: #1F2937; font-size: 0.82rem; font-weight: 700;'>{risk_index:.1f}/100</div>
                        </div>
                        <div style='height: 8px; background: #E5E7EB; border-radius: 999px; overflow: hidden;'>
                            <div style='height: 100%; width: {risk_index:.1f}%; background: linear-gradient(90deg, #10B981 0%, #F59E0B 55%, #EF4444 100%); border-radius: 999px;'></div>
                        </div>
                    </div>

                    <div style='display: grid; grid-template-columns: 1fr; gap: 0.55rem; margin-top: 0.9rem;'>
                        <div style='display: flex; justify-content: space-between; align-items: center; padding: 0.75rem 0.85rem; border-radius: 10px; background: #FFFFFF; border: 1px solid #E5E7EB;'>
                            <span style='color: #64748B; font-size: 0.86rem;'>审批建议</span>
                            <span class='risk-num' style='color: #111827; font-size: 1.08rem; font-weight: 800;'>{recommendation_label}</span>
                        </div>
                        <div style='display: flex; justify-content: space-between; align-items: center; padding: 0.75rem 0.85rem; border-radius: 10px; background: #FFFFFF; border: 1px solid #E5E7EB;'>
                            <span style='color: #64748B; font-size: 0.86rem;'>推荐额度</span>
                            <span class='risk-num' style='color: #111827; font-size: 1.08rem; font-weight: 800;'>{input_data.get('income', 60000) * 0.3:,.0f}元</span>
                        </div>
                    </div>
                </div>
                """).strip()
                
                components.html(result_html, height=500, scrolling=False)
                
                # 保存记录
                record = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'default_probability': default_prob,
                    'credit_score': credit_score,
                    'risk_level': risk_level,
                    'recommendation': recommendation
                }
                st.session_state.assessment_history.append(record)
        else:
            st.info("""
            **等待评估...**
            
            请在左侧输入客户信息，
            点击"执行风险评估"开始分析。
            """)

# 5. 数据分析页面
elif page == "数据分析":
    st.header("数据分析")
    st.markdown("<p style='color: #666;'>数据洞察与模型性能分析</p>", unsafe_allow_html=True)
    
    if not st.session_state.clients:
        st.warning("暂无数据可供分析。")
        st.stop()
    
    df = pd.DataFrame(st.session_state.clients)
    
    analysis_tab = st.radio(
        "数据分析视图",
        ["数据分布", "模型分析"],
        horizontal=True,
        label_visibility="collapsed",
        key="analysis_tab_selector"
    )
    
    if analysis_tab == "数据分布":
        st.subheader("数据分布分析")
        feature_labels = {
            "age": "年龄",
            "income": "年收入",
            "balance": "账户余额",
            "job_years": "工作年限",
            "credit_score": "信用评分",
        }
        unit_map = {
            "age": "岁",
            "income": "元",
            "balance": "元",
            "job_years": "年",
            "credit_score": "分",
        }

        feature = st.selectbox(
            "分析特征",
            list(feature_labels.keys()),
            format_func=lambda x: feature_labels.get(x, x),
            key="dist_feature",
        )

        summary_cols = st.columns(4)
        with summary_cols[0]:
            st.metric("样本量", f"{len(df):,}")
        with summary_cols[1]:
            st.metric("均值", f"{df[feature].mean():,.1f}{unit_map[feature]}")
        with summary_cols[2]:
            st.metric("中位数", f"{df[feature].median():,.1f}{unit_map[feature]}")
        with summary_cols[3]:
            st.metric("标准差", f"{df[feature].std():,.1f}{unit_map[feature]}")

        col1, col2 = st.columns(2)

        with col1:
            hist_fig = px.histogram(
                df,
                x=feature,
                nbins=24,
                title=f"{feature_labels[feature]}分布直方图",
                opacity=0.9,
                color_discrete_sequence=[PLOTLY_DISCRETE_COLORS[0]],
            )
            hist_fig.update_traces(
                marker_line_color="white",
                marker_line_width=1,
                hovertemplate=f"{feature_labels[feature]}: " + "%{x}<br>样本数: %{y}<extra></extra>",
            )
            hist_fig.update_layout(
                height=CHART_PAIR_HEIGHT,
                margin=CHART_MARGIN_COMPACT,
                plot_bgcolor="rgba(248, 249, 250, 0.5)",
                paper_bgcolor="rgba(248, 249, 250, 0.5)",
                font=dict(family="Arial, sans-serif", size=12, color="#333333"),
                bargap=0.08,
            )
            hist_fig.update_xaxes(
                title_text=f"{feature_labels[feature]}（{unit_map[feature]}）",
                gridcolor="rgba(0,0,0,0.08)",
            )
            hist_fig.update_yaxes(
                title_text="样本数",
                gridcolor="rgba(0,0,0,0.08)",
            )
            st.plotly_chart(hist_fig, use_container_width=True, config=PLOTLY_CONFIG)

        with col2:
            if "default_label" in df.columns and not df.empty:
                income_bins = pd.qcut(df["income"], q=5, duplicates="drop")
                default_by_income = df.groupby(income_bins)["default_label"].mean().reset_index()
                default_by_income["income_bin_str"] = default_by_income["income"].astype(str)

                risk_fig = px.bar(
                    default_by_income,
                    x="income_bin_str",
                    y="default_label",
                    title="收入分段与违约率关系",
                    labels={"income_bin_str": "收入区间", "default_label": "违约率"},
                    color="default_label",
                    color_continuous_scale=PLOTLY_CONTINUOUS_SCALE,
                    opacity=0.95,
                    text="default_label",
                )
                risk_fig.update_traces(
                    texttemplate="%{text:.1%}",
                    textposition="outside",
                    hovertemplate="收入区间: %{x}<br>违约率: %{y:.1%}<extra></extra>",
                    cliponaxis=False,
                )
                risk_fig.update_layout(
                    height=CHART_PAIR_HEIGHT,
                    margin=CHART_MARGIN_COMPACT,
                    yaxis_tickformat=".1%",
                    plot_bgcolor="rgba(248, 249, 250, 0.5)",
                    paper_bgcolor="rgba(248, 249, 250, 0.5)",
                    font=dict(family="Arial, sans-serif", size=12, color="#333333"),
                    coloraxis_showscale=False,
                )
                risk_fig.update_xaxes(gridcolor="rgba(0,0,0,0.08)", automargin=True)
                risk_fig.update_yaxes(title_text="违约率", gridcolor="rgba(0,0,0,0.08)", automargin=True)
                st.plotly_chart(risk_fig, use_container_width=True, config=PLOTLY_CONFIG)
            else:
                st.info("暂无违约标签数据，无法展示收入分段与违约率关系。")
    
    elif analysis_tab == "模型分析":
        st.subheader("模型性能分析")
        
        if st.session_state.trained_model and st.session_state.model_metrics:
            metrics = st.session_state.model_metrics

            metric_row = st.columns(3)
            with metric_row[0]:
                st.metric("AUC", f"{metrics['auc']:.3f}")
            with metric_row[1]:
                st.metric("训练样本", f"{metrics.get('train_size', 0):,}")
            with metric_row[2]:
                st.metric("测试样本", f"{metrics.get('test_size', 0):,}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                auc_value = metrics['auc'] * 100

                # 更专业的 AUC 表现：子弹图（Bullet Chart）+目标阈值
                benchmark_auc = 80.0
                fig = go.Figure(go.Indicator(
                    mode="number+gauge+delta",
                    value=auc_value,
                    number={
                        'font': {'size': 34, 'color': '#1F2937', 'family': 'Arial, sans-serif'},
                        'suffix': '%',
                        'valueformat': '.1f'
                    },
                    delta={
                        'reference': benchmark_auc,
                        'relative': False,
                        'increasing': {'color': '#5FBFA6'},
                        'decreasing': {'color': '#F3A3A3'},
                        'valueformat': '.1f'
                    },
                    title={'text': "", 'font': CHART_TITLE_FONT},
                    gauge={
                        'shape': "bullet",
                        'axis': {'range': [50, 100], 'tickwidth': 1, 'tickcolor': '#9CA3AF'},
                        'bar': {'color': '#9FB6FF'},
                        'bgcolor': 'white',
                        'borderwidth': 1,
                        'bordercolor': '#E5E7EB',
                        'steps': [
                            {'range': [50, 70], 'color': '#F0B9B9'},
                            {'range': [70, 85], 'color': '#EFC98E'},
                            {'range': [85, 100], 'color': '#A9E2C9'},
                        ],
                        'threshold': {
                            'line': {'color': '#86A8FF', 'width': 3},
                            'thickness': 0.9,
                            'value': benchmark_auc
                        }
                    },
                    domain={'x': [0.1, 0.95], 'y': [0.28, 0.95]}
                ))

                fig.update_layout(
                    title={
                        "text": "模型AUC评分",
                        "x": 0.0,
                        "xanchor": "left",
                    },
                    height=CHART_PAIR_HEIGHT,
                    margin=CHART_MARGIN_COMPACT,
                    font={'color': "#333", 'family': "Arial, sans-serif", 'size': 12},
                    paper_bgcolor='rgba(248, 249, 250, 0.5)'
                )
                
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            
            with col2:
                if st.session_state.feature_importance is not None:
                    top_features = st.session_state.feature_importance.head(5).copy()
                    
                    if not top_features.empty:
                        top_features['percentage'] = (top_features['importance'] / top_features[
                            'importance'].sum()) * 100
                        
                        top_features = top_features.sort_values("percentage", ascending=True)
                        fig = px.bar(
                            top_features,
                            x="percentage",
                            y="description",
                            orientation="h",
                            title="Top 5 特征重要性（占比）",
                            text="percentage",
                            color="percentage",
                            color_continuous_scale=PLOTLY_CONTINUOUS_SCALE,
                        )
                        fig.update_traces(
                            texttemplate="%{text:.1f}%",
                            textposition="outside",
                            marker_line_color="white",
                            marker_line_width=1.5,
                            hovertemplate="<b>%{y}</b><br>占比: %{x:.2f}%<extra></extra>",
                            cliponaxis=False,
                        )
                        fig.update_layout(
                            height=CHART_PAIR_HEIGHT,
                            margin=dict(t=56, b=30, l=14, r=30),
                            plot_bgcolor='rgba(248, 249, 250, 0.5)',
                            paper_bgcolor='rgba(248, 249, 250, 0.5)',
                            font=dict(family="Arial, sans-serif", size=12, color="#333333"),
                            coloraxis_showscale=False,
                        )
                        fig.update_xaxes(
                            title_text="重要性占比（%）",
                            gridcolor="rgba(0,0,0,0.08)",
                            automargin=True,
                        )
                        fig.update_yaxes(
                            title_text="特征",
                            gridcolor="rgba(0,0,0,0.0)",
                            automargin=True,
                        )
                        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("请先完成模型训练以查看分析")

# 页脚
st.markdown("---")
st.markdown(f"""
<div class="app-footer">
  <div style='text-align: center; color: #666; font-size: 0.9rem;'>
      <p style='margin: 0;'><strong>智能信用评分系统</strong> | 专业信用风险评估平台</p>
      <p style='margin: 0.3rem 0 0 0; color: #4F46E5; font-weight: 600;'>开发作者: GhostLoo | 版本 {APP_VERSION}</p>
  </div>
</div>
""", unsafe_allow_html=True)