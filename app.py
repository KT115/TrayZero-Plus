# ==============================================================================
# 1. UI Styling & Typography (已修復對比度、輸入框黑底與文字不可讀問題)
# ==============================================================================
def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Noto+Sans+TC:wght@400;500;700;900&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', 'Noto Sans TC', sans-serif;
            color: #0F172A !important;
        }

        .stApp {
            background-color: #F8FAFC !important;
        }

        /* 確保頂部標題不被 Streamlit 頂部黑條遮擋 */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
        }

        /* 側邊欄樣式 */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
        }
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
            color: #1E293B !important;
            font-weight: 600 !important;
        }

        /* 頂部白底大標題卡片 */
        .trayzero-header {
            background: #FFFFFF !important;
            border-radius: 14px;
            padding: 16px 22px;
            margin-bottom: 20px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
        }
        .trayzero-title {
            color: #0F172A !important;
            font-size: 1.3rem !important;
            font-weight: 800 !important;
            margin: 0 !important;
            line-height: 1.3 !important;
        }

        /* -----------------------------------------------------------
           核心修復 1: 所有輸入框、下拉選單背景為白色、文字為深色清晰黑
        ----------------------------------------------------------- */
        /* 下拉選單 Selectbox / 文本輸入 Text Input */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        input[type="text"],
        input[type="number"] {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1.5px solid #CBD5E1 !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }

        /* 選單展開後的文字顏色 */
        div[data-baseweb="popover"], div[data-baseweb="menu"] {
            background-color: #FFFFFF !important;
        }
        div[data-baseweb="menu"] li, div[data-baseweb="menu"] div {
            color: #0F172A !important;
            font-weight: 500 !important;
        }

        /* 輸入框內的 Placeholder 顏色 */
        input::placeholder {
            color: #94A3B8 !important;
            font-weight: normal !important;
        }

        /* -----------------------------------------------------------
           核心修復 2: 徹底修復單選按鈕 (Radio Buttons) 文字消失隱形問題
        ----------------------------------------------------------- */
        div[data-testid="stRadio"] label,
        div[data-testid="stRadio"] label p,
        div[data-testid="stRadio"] label span,
        div[data-testid="stCheckbox"] label,
        div[data-testid="stCheckbox"] label p {
            color: #0F172A !important;
            font-size: 0.92rem !important;
            font-weight: 600 !important;
            opacity: 1 !important;
            visibility: visible !important;
        }

        /* -----------------------------------------------------------
           卡片與資訊塊修飾
        ----------------------------------------------------------- */
        .clean-card {
            background: #FFFFFF !important;
            border-radius: 14px;
            padding: 16px 18px;
            margin-bottom: 12px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
        }
        .clean-label {
            font-size: 0.75rem;
            color: #64748B !important;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }
        .clean-val {
            font-size: 1.7rem;
            font-weight: 800;
            color: #0F172A !important;
            line-height: 1.1;
        }

        .crm-card {
            background: #F0FDF4 !important;
            border: 1.5px solid #86EFAC !important;
            border-radius: 12px;
            padding: 16px 18px;
            margin-bottom: 14px;
            color: #14532D !important;
        }
        .crm-title {
            font-size: 0.95rem;
            font-weight: 800;
            color: #166534 !important;
            margin-bottom: 6px;
        }

        .member-live-badge {
            background: #EFF6FF !important;
            border: 1.5px solid #BFDBFE !important;
            border-radius: 10px;
            padding: 12px 14px;
            margin-bottom: 14px;
            font-size: 0.88rem;
            color: #1E3A8A !important;
        }

        /* 指令卡片樣式 */
        .directive-card {
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 10px;
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #CBD5E1;
        }
        .directive-chef { border-left-color: #EF4444 !important; }
        .directive-pos { border-left-color: #10B981 !important; }
        .directive-crm { border-left-color: #3B82F6 !important; }
        .directive-title {
            font-weight: 700;
            font-size: 0.88rem;
            color: #0F172A !important;
            margin-bottom: 4px;
        }
        .directive-body {
            font-size: 0.85rem;
            color: #334155 !important;
            line-height: 1.6;
        }
    </style>
    """, unsafe_allow_html=True)
