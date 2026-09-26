def inject_safe_css():
    st.markdown("""
    <style>
        /* 根容器設定 */
        .stApp {
            background-color: #F8FAFC !important;
            color: #0F172A !important;
        }

        .main .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 3rem !important;
            color: #0F172A !important;
        }

        /* -----------------------------------------------------------
           核心修復 1: 所有元件標題 Label 強制深黑顯色 (解決 Scan Member ID 隱形)
        ----------------------------------------------------------- */
        div[data-testid="stWidgetLabel"],
        div[data-testid="stWidgetLabel"] *,
        div[data-testid="stWidgetLabel"] p,
        div[data-testid="stWidgetLabel"] span,
        div[data-testid="stWidgetLabel"] label {
            color: #0F172A !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            opacity: 1 !important;
            visibility: visible !important;
        }

        /* -----------------------------------------------------------
           核心修復 2: 單選 Radio 與 Checkbox 文字強制顯色 (解決選項文字消失)
        ----------------------------------------------------------- */
        div[data-testid="stRadio"] *,
        div[data-testid="stRadio"] label,
        div[data-testid="stRadio"] label p,
        div[data-testid="stRadio"] label span,
        div[data-testid="stRadio"] label div,
        div[data-testid="stCheckbox"] *,
        div[data-testid="stCheckbox"] label,
        div[data-testid="stCheckbox"] label p {
            color: #0F172A !important;
            font-size: 0.92rem !important;
            font-weight: 600 !important;
            opacity: 1 !important;
            visibility: visible !important;
        }

        /* -----------------------------------------------------------
           核心修復 3: 標籤頁 (Tabs) 未選中與選中狀態文字均清晰可見
        ----------------------------------------------------------- */
        button[data-baseweb="tab"] {
            background: transparent !important;
            padding: 10px 20px !important;
        }
        /* 未選中 Tab：深岩灰，字體加粗，清晰分明 */
        button[data-baseweb="tab"] * {
            color: #334155 !important;
            font-size: 1.0rem !important;
            font-weight: 700 !important;
            opacity: 1 !important;
            visibility: visible !important;
        }
        /* 選中 Tab：高亮大家樂科技藍 */
        button[data-baseweb="tab"][aria-selected="true"] {
            border-bottom: 3px solid #2563EB !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] * {
            color: #2563EB !important;
            font-weight: 800 !important;
        }

        /* -----------------------------------------------------------
           核心修復 4: 輸入框文字與 Placeholder 高對比度
        ----------------------------------------------------------- */
        input[type="text"], 
        input[type="number"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"] {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1.5px solid #94A3B8 !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        input::placeholder {
            color: #64748B !important;
            opacity: 1 !important;
            font-weight: 500 !important;
        }
        div[data-baseweb="select"] * {
            color: #0F172A !important;
        }

        /* 頂部標題 */
        .trayzero-header {
            background: #FFFFFF !important;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 18px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.05);
        }
        .trayzero-title {
            color: #0F172A !important;
            font-size: 1.35rem !important;
            font-weight: 800 !important;
            margin: 0 !important;
        }

        /* 側邊欄樣式 */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
        }
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] label, 
        [data-testid="stSidebar"] span {
            color: #1E293B !important;
            font-weight: 600 !important;
        }
        [data-testid="stSidebar"] [data-testid="stImage"] {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            margin: 0 auto !important;
            text-align: center !important;
        }
        [data-testid="stSidebar"] [data-testid="stImage"] img {
            margin: 0 auto !important;
            display: block !important;
        }

        /* 按鈕樣式 */
        button[kind="primary"] {
            background-color: #2563EB !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
        }
        button[kind="secondary"] {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1.5px solid #CBD5E1 !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
        }

        /* 卡片元件 */
        .clean-card {
            background: #FFFFFF !important;
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 12px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        }
        .clean-label {
            font-size: 0.75rem;
            color: #64748B !important;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .clean-val {
            font-size: 1.6rem;
            font-weight: 800;
            color: #0F172A !important;
            line-height: 1.1;
        }

        .crm-card {
            background: #F0FDF4 !important;
            border: 1.5px solid #86EFAC !important;
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 12px;
            color: #14532D !important;
        }

        .member-live-badge {
            background: #EFF6FF !important;
            border: 1.5px solid #BFDBFE !important;
            border-radius: 10px;
            padding: 10px 14px;
            margin-bottom: 12px;
            font-size: 0.88rem;
            color: #1E3A8A !important;
        }

        .directive-card {
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 10px;
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #CBD5E1;
        }
        .directive-chef { border-left-color: #EF4444 !important; }
        .directive-pos { border-left-color: #10B981 !important; }
        .directive-crm { border-left-color: #3B82F6 !important; }
    </style>
    """, unsafe_allow_html=True)
