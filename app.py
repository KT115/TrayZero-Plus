import os
import datetime
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification,
    pipeline
)
import altair as alt

# ==============================================================================
# 0. Primary Streamlit Execution Configuration
# ==============================================================================
st.set_page_config(
    page_title="TrayZero+ | 智能餐盤審計與會員獎勵系統", 
    page_icon="🍽️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 1. Global Paths & Fast-Casual POS Enterprise CSS Theme
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BRANCH_FILE = os.path.join(BASE_DIR, "master_branches.csv")
DISH_FILE = os.path.join(BASE_DIR, "master_dishes.csv")
REWARD_FILE = os.path.join(BASE_DIR, "master_rewards.csv")
SEED_AUDIT_FILE = os.path.join(BASE_DIR, "seed_audit_logs.csv")
DB_FILE = os.path.join(BASE_DIR, "trayzero_audit.db")
DISH_IMG_DIR = os.path.join(BASE_DIR, "dish_references")
LOGO_FILE_PNG = os.path.join(BASE_DIR, "CDC_810.png")
LOGO_FILE_JPG = os.path.join(BASE_DIR, "CDC_810.jpg")

os.makedirs(DISH_IMG_DIR, exist_ok=True)

DEFAULT_BASE_GDRIVE_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSloK2WPNFd8HPY4RfL2rNhwhk_kD12H0q09nDcrlMrx5O_zqslCOi1TPAXvlHtnP1FWxyJxGgG99QX/pub?output=csv"

def inject_safe_css():
    st.markdown("""
    <style>
        :root {
            --cdc-red: #DC2626 !important;
            --cdc-amber: #D97706 !important;
            --text-color: #0F172A !important;
            --background-color: #F8FAFC !important;
            --secondary-background-color: #FFFFFF !important;
        }

        .stApp {
            background-color: #F8FAFC !important;
            color: #0F172A !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        }

        .main .block-container {
            padding-top: 1.2rem !important;
            padding-bottom: 3rem !important;
            color: #0F172A !important;
        }

        .pos-header-banner {
            background: linear-gradient(135deg, #C2301A 0%, #D95D1A 48%, #D87B18 100%) !important;
            border-radius: 14px !important;
            padding: 16px 24px !important;
            margin-bottom: 22px !important;
            box-shadow: 0 4px 14px rgba(194, 48, 26, 0.22) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            display: flex !important;
            align-items: center !important;
        }
        .pos-header-title {
            color: #FFFFFF !important;
            font-size: 1.45rem !important;
            font-weight: 900 !important;
            margin: 0 !important;
            letter-spacing: 0.02em !important;
            text-shadow: 0 1px 3px rgba(0, 0, 0, 0.25) !important;
        }

        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1.5px solid #E2E8F0 !important;
            padding-top: 1rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stImage"] {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            margin-left: auto !important;
            margin-right: auto !important;
            margin-bottom: 16px !important;
            width: 100% !important;
            text-align: center !important;
        }
        [data-testid="stSidebar"] [data-testid="stImage"] > img {
            margin-left: auto !important;
            margin-right: auto !important;
            display: block !important;
            max-width: 175px !important;
        }

        label[data-testid="stWidgetLabel"],
        div[data-testid="stWidgetLabel"] label,
        div[data-testid="stWidgetLabel"] p,
        div[data-testid="stWidgetLabel"] span {
            color: #0F172A !important;
            font-size: 0.95rem !important;
            font-weight: 800 !important;
            opacity: 1 !important;
            visibility: visible !important;
        }

        div[data-testid="stRadio"] [role="radiogroup"] label,
        div[data-testid="stRadio"] [role="radiogroup"] label p,
        div[data-testid="stRadio"] [role="radiogroup"] label span,
        div[data-testid="stCheckbox"] label p,
        div[data-testid="stCheckbox"] label span {
            color: #0F172A !important;
            font-size: 0.92rem !important;
            font-weight: 700 !important;
            opacity: 1 !important;
            visibility: visible !important;
        }

        div[data-baseweb="tab-list"] button[data-baseweb="tab"] {
            background: transparent !important;
            padding: 10px 18px !important;
        }
        div[data-baseweb="tab-list"] button[data-baseweb="tab"] p,
        div[data-baseweb="tab-list"] button[data-baseweb="tab"] span {
            color: #475569 !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            opacity: 1 !important;
            visibility: visible !important;
        }
        div[data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] {
            border-bottom: 3px solid #D95D1A !important;
        }
        div[data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] p,
        div[data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] span {
            color: #C2301A !important;
            font-weight: 900 !important;
        }

        input[type="text"], 
        input[type="number"],
        div[data-baseweb="input"] input,
        div[data-baseweb="select"] div {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border-color: #CBD5E1 !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
        }
        input::placeholder {
            color: #94A3B8 !important;
            font-weight: 600 !important;
        }

        .pos-coupon-card {
            background: #FFFBEB !important;
            border: 2px dashed #D97706 !important;
            border-radius: 12px;
            padding: 16px 18px;
            margin-bottom: 16px;
            box-shadow: 0 2px 6px rgba(217, 119, 6, 0.05);
        }
        .pos-coupon-header {
            font-size: 0.85rem;
            font-weight: 900;
            color: #D97706;
            margin-bottom: 4px;
        }
        .pos-coupon-value {
            font-size: 1.22rem;
            font-weight: 900;
            color: #0F172A;
            margin-bottom: 4px;
        }
        .pos-coupon-desc {
            font-size: 0.8rem;
            font-weight: 700;
            color: #78350F;
        }
        .pos-coupon-badge {
            display: inline-block;
            background: #D97706;
            color: #FFFFFF;
            font-size: 0.75rem;
            font-weight: 900;
            padding: 4px 10px;
            border-radius: 6px;
            margin-top: 8px;
        }

        .pos-metric-card {
            background: #FFFFFF !important;
            border-radius: 12px;
            padding: 14px 16px;
            border: 1.5px solid #E2E8F0;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
            text-align: center;
        }
        .pos-metric-card.amber-glow {
            background: #FFFBEB !important;
            border-color: #FDE68A !important;
        }
        .pos-metric-card.green-glow {
            background: #F0FDF4 !important;
            border-color: #86EFAC !important;
        }
        .pos-metric-label {
            font-size: 0.72rem;
            font-weight: 800;
            color: #64748B !important;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .pos-metric-val {
            font-size: 1.7rem;
            font-weight: 900;
            line-height: 1.1;
        }

        .chart-box {
            background: #FFFFFF;
            border-radius: 12px;
            padding: 18px 20px;
            border: 1.5px solid #E2E8F0;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
            margin-bottom: 16px;
        }
        .chart-title {
            font-size: 0.95rem;
            font-weight: 800;
            color: #0F172A;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .pos-directive-card {
            border-radius: 10px;
            padding: 14px 18px;
            margin-top: 14px;
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0;
            border-left: 5px solid #DC2626 !important;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
        }

        .export-control-box {
            background: #FFFFFF;
            border: 1.5px solid #CBD5E1;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 16px;
        }

        .empty-state-box {
            background: #FFFFFF;
            border: 2px dashed #CBD5E1;
            border-radius: 14px;
            padding: 40px;
            text-align: center;
            color: #64748B;
            margin-top: 20px;
            margin-bottom: 20px;
        }

        button[kind="primary"] {
            background-color: #D97706 !important;
            color: #FFFFFF !important;
            font-weight: 800 !important;
            border-radius: 8px !important;
            border: none !important;
        }
        button[kind="secondary"] {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1.5px solid #CBD5E1 !important;
            font-weight: 800 !important;
            border-radius: 8px !important;
        }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. Database Connection & Schema Setup
# ==============================================================================
def db_conn(): 
    return sqlite3.connect(DB_FILE)

def init_db():
    with db_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                audit_date TEXT,
                audit_month TEXT,
                branch_name TEXT,
                member_id TEXT,
                dish_name TEXT,
                primary_waste TEXT,
                waste_ratio REAL,
                cost_waste_hkd REAL,
                co2_emission_kg REAL,
                reward_issued TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS master_dishes_db (
                dish_id TEXT PRIMARY KEY,
                name TEXT,
                main_carb TEXT,
                protein TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS master_branches_db (
                name TEXT PRIMARY KEY,
                level TEXT,
                district TEXT,
                traffic TEXT,
                avg_covers INTEGER,
                base_rice_g INTEGER
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS master_rewards_db (
                reward_id TEXT PRIMARY KEY,
                tier_name TEXT,
                max_waste_ratio REAL,
                reward_type TEXT,
                reward_description TEXT,
                is_active INTEGER
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS cloud_config_db (
                key TEXT PRIMARY KEY,
                url TEXT
            )
        """)

        conn.execute("INSERT OR REPLACE INTO cloud_config_db VALUES ('base_gdrive_url', ?)", (DEFAULT_BASE_GDRIVE_URL,))
        conn.execute("INSERT OR REPLACE INTO cloud_config_db VALUES ('dishes_url', ?)", (f"{DEFAULT_BASE_GDRIVE_URL}&sheet=dishes",))
        conn.execute("INSERT OR REPLACE INTO cloud_config_db VALUES ('branches_url', ?)", (f"{DEFAULT_BASE_GDRIVE_URL}&sheet=branches",))
        conn.execute("INSERT OR REPLACE INTO cloud_config_db VALUES ('rewards_url', ?)", (f"{DEFAULT_BASE_GDRIVE_URL}&sheet=rewards",))

        init_dishes = [
            ("D01", "一哥焗豬扒飯 (Baked Pork Chop Rice)", "白米飯", "焗厚切豬扒"),
            ("D02", "咖喱牛腩飯 (Curry Beef Brisket Rice)", "白米飯", "慢燉牛腩"),
            ("D03", "滑蛋蝦仁飯 (Scrambled Egg Shrimp Rice)", "白米飯", "滑蛋蝦仁"),
            ("D04", "香辣肉燥肉餅飯 (Minced Pork Patty Rice)", "白米飯", "煎肉餅"),
            ("D05", "焗肉醬意粉 (Baked Spaghetti Bolognese)", "意大利麵", "慢燉牛肉醬"),
            ("D06", "車仔麵 (Kart Noodle)", "中式麵條", "牛腩/魚蛋/蘿蔔")
        ]
        conn.executemany("""
            INSERT OR REPLACE INTO master_dishes_db (dish_id, name, main_carb, protein) 
            VALUES (?, ?, ?, ?)
        """, init_dishes)

        init_branches = [
            ("中環威靈頓街店", "Level A (商業核心區 / CBD)", "中西區", "白領上班族為主，午市尖峰翻檯率極高", 1200, 240),
            ("沙田新城市廣場店", "Level B (住宅商場 / Residential)", "沙田區", "家庭客、長者與週末休閒客群", 1500, 260),
            ("香港科技大學店 (HKUST)", "Level C (校園與青年區 / Campus)", "西貢區", "學生、教職員，運動量及食量顯著較大", 1800, 280),
            ("將軍澳 Popcorn 店", "Level B (住宅商場 / Residential)", "西貢區", "家庭客及換乘鐵路客流", 1400, 260)
        ]
        conn.executemany("""
            INSERT OR REPLACE INTO master_branches_db (name, level, district, traffic, avg_covers, base_rice_g) 
            VALUES (?, ?, ?, ?, ?, ?)
        """, init_branches)

        init_rewards = [
            ("R01", "極致光盤獎 (Ultra Clean)", 10.0, "Coupon + Points", "【$3 堂食現金券】+【50 綠色積分】+【凍檸茶半價券】", 1),
            ("R02", "達標惜食獎 (Standard Clean)", 20.0, "Coupon", "【$2 堂食電子券】+【20 綠色積分】", 1),
            ("R03", "支持環保獎 (Green Return)", 100.0, "Points", "【10 綠色環保積分】", 1)
        ]
        conn.executemany("""
            INSERT OR REPLACE INTO master_rewards_db (reward_id, tier_name, max_waste_ratio, reward_type, reward_description, is_active) 
            VALUES (?, ?, ?, ?, ?, ?)
        """, init_rewards)

# ==============================================================================
# 3. Google Drive / Sheets 雲端連線支援模組
# ==============================================================================
def get_cloud_urls():
    urls = {
        "base_url": DEFAULT_BASE_GDRIVE_URL,
        "dishes_url": f"{DEFAULT_BASE_GDRIVE_URL}&sheet=dishes",
        "branches_url": f"{DEFAULT_BASE_GDRIVE_URL}&sheet=branches",
        "rewards_url": f"{DEFAULT_BASE_GDRIVE_URL}&sheet=rewards"
    }
    with db_conn() as conn:
        rows = conn.execute("SELECT key, url FROM cloud_config_db").fetchall()
        for k, u in rows:
            if u and u.strip():
                urls[k] = u.strip()
    return urls

def save_cloud_urls(urls):
    with db_conn() as conn:
        for k, u in urls.items():
            conn.execute("INSERT OR REPLACE INTO cloud_config_db VALUES (?, ?)", (k, u.strip()))

# ==============================================================================
# 4. 雙向永續資料讀取與儲存
# ==============================================================================
def get_live_dishes():
    cloud_urls = get_cloud_urls()
    for u in [cloud_urls.get("dishes_url"), cloud_urls.get("base_url")]:
        if u and u.startswith("http"):
            try:
                cloud_df = pd.read_csv(u, encoding="utf-8-sig")
                if not cloud_df.empty and "name" in cloud_df.columns:
                    cloud_df = cloud_df.dropna(subset=["name"])
                    cloud_df = cloud_df[cloud_df["name"].astype(str).str.strip() != ""]
                    cloud_df.to_csv(DISH_FILE, index=False, encoding="utf-8-sig")
                    with db_conn() as conn:
                        cloud_df.to_sql("master_dishes_db", conn, if_exists="replace", index=False)
                    return cloud_df
            except Exception:
                pass

    with db_conn() as conn:
        db_df = pd.read_sql("SELECT dish_id, name, main_carb, protein FROM master_dishes_db", conn)

    csv_df = pd.DataFrame(columns=["dish_id", "name", "main_carb", "protein"])
    if os.path.exists(DISH_FILE):
        try:
            csv_df = pd.read_csv(DISH_FILE, encoding="utf-8-sig")
        except Exception:
            pass

    combined = pd.concat([db_df, csv_df], ignore_index=True)
    if not combined.empty:
        combined = combined.dropna(subset=["name"])
        combined = combined[combined["name"].astype(str).str.strip() != ""]
        combined = combined.drop_duplicates(subset=["name"], keep="last")
        combined.to_csv(DISH_FILE, index=False, encoding="utf-8-sig")
        with db_conn() as conn:
            combined.to_sql("master_dishes_db", conn, if_exists="replace", index=False)
        return combined
    return pd.DataFrame(columns=["dish_id", "name", "main_carb", "protein"])

def save_live_dishes(df):
    clean_df = df.dropna(subset=["name"]).copy()
    clean_df = clean_df[clean_df["name"].astype(str).str.strip() != ""]
    clean_df = clean_df.drop_duplicates(subset=["name"], keep="last")
    clean_df.to_csv(DISH_FILE, index=False, encoding="utf-8-sig")
    with db_conn() as conn:
        clean_df.to_sql("master_dishes_db", conn, if_exists="replace", index=False)

def get_live_branches():
    cloud_urls = get_cloud_urls()
    u = cloud_urls.get("branches_url")
    if u and u.startswith("http"):
        try:
            cloud_df = pd.read_csv(u, encoding="utf-8-sig")
            if not cloud_df.empty and "name" in cloud_df.columns:
                cloud_df = cloud_df.dropna(subset=["name"])
                cloud_df = cloud_df[cloud_df["name"].astype(str).str.strip() != ""]
                cloud_df.to_csv(BRANCH_FILE, index=False, encoding="utf-8-sig")
                with db_conn() as conn:
                    cloud_df.to_sql("master_branches_db", conn, if_exists="replace", index=False)
                return cloud_df
        except Exception:
            pass

    with db_conn() as conn:
        db_df = pd.read_sql("SELECT name, level, district, traffic, avg_covers, base_rice_g FROM master_branches_db", conn)

    csv_df = pd.DataFrame(columns=["name", "level", "district", "traffic", "avg_covers", "base_rice_g"])
    if os.path.exists(BRANCH_FILE):
        try:
            csv_df = pd.read_csv(BRANCH_FILE, encoding="utf-8-sig")
        except Exception:
            pass

    combined = pd.concat([db_df, csv_df], ignore_index=True)
    if not combined.empty:
        combined = combined.dropna(subset=["name"])
        combined = combined[combined["name"].astype(str).str.strip() != ""]
        combined = combined.drop_duplicates(subset=["name"], keep="last")
        combined.to_csv(BRANCH_FILE, index=False, encoding="utf-8-sig")
        with db_conn() as conn:
            combined.to_sql("master_branches_db", conn, if_exists="replace", index=False)
        return combined
    return pd.DataFrame(columns=["name", "level", "district", "traffic", "avg_covers", "base_rice_g"])

def save_live_branches(df):
    clean_df = df.dropna(subset=["name"]).copy()
    clean_df = clean_df[clean_df["name"].astype(str).str.strip() != ""]
    clean_df = clean_df.drop_duplicates(subset=["name"], keep="last")
    clean_df.to_csv(BRANCH_FILE, index=False, encoding="utf-8-sig")
    with db_conn() as conn:
        clean_df.to_sql("master_branches_db", conn, if_exists="replace", index=False)

def get_live_rewards():
    cloud_urls = get_cloud_urls()
    u = cloud_urls.get("rewards_url")
    if u and u.startswith("http"):
        try:
            cloud_df = pd.read_csv(u, encoding="utf-8-sig")
            if not cloud_df.empty and "tier_name" in cloud_df.columns:
                cloud_df = cloud_df.dropna(subset=["tier_name"])
                cloud_df = cloud_df[cloud_df["tier_name"].astype(str).str.strip() != ""]
                cloud_df.to_csv(REWARD_FILE, index=False, encoding="utf-8-sig")
                with db_conn() as conn:
                    cloud_df.to_sql("master_rewards_db", conn, if_exists="replace", index=False)
                return cloud_df
        except Exception:
            pass

    with db_conn() as conn:
        db_df = pd.read_sql("SELECT reward_id, tier_name, max_waste_ratio, reward_type, reward_description, is_active FROM master_rewards_db", conn)

    csv_df = pd.DataFrame(columns=["reward_id", "tier_name", "max_waste_ratio", "reward_type", "reward_description", "is_active"])
    if os.path.exists(REWARD_FILE):
        try:
            csv_df = pd.read_csv(REWARD_FILE, encoding="utf-8-sig")
        except Exception:
            pass

    combined = pd.concat([db_df, csv_df], ignore_index=True)
    if not combined.empty:
        combined = combined.dropna(subset=["tier_name"])
        combined = combined[combined["tier_name"].astype(str).str.strip() != ""]
        combined = combined.drop_duplicates(subset=["tier_name"], keep="last")
        combined.to_csv(REWARD_FILE, index=False, encoding="utf-8-sig")
        with db_conn() as conn:
            combined.to_sql("master_rewards_db", conn, if_exists="replace", index=False)
        return combined

    default_rewards = [
        {"reward_id": "R01", "tier_name": "極致光盤獎 (Ultra Clean)", "max_waste_ratio": 10.0, "reward_type": "Coupon + Points", "reward_description": "【$3 堂食現金券】+【50 綠色積分】+【凍檸茶半價券】", "is_active": True},
        {"reward_id": "R02", "tier_name": "達標惜食獎 (Standard Clean)", "max_waste_ratio": 20.0, "reward_type": "Coupon", "reward_description": "【$2 堂食電子券】+【20 綠色積分】", "is_active": True},
        {"reward_id": "R03", "tier_name": "支持環保獎 (Green Return)", "max_waste_ratio": 100.0, "reward_type": "Points", "reward_description": "【10 綠色環保積分】", "is_active": True}
    ]
    df_r = pd.DataFrame(default_rewards)
    save_live_rewards(df_r)
    return df_r

def save_live_rewards(df):
    clean_df = df.dropna(subset=["tier_name"]).copy()
    clean_df = clean_df[clean_df["tier_name"].astype(str).str.strip() != ""]
    clean_df = clean_df.drop_duplicates(subset=["tier_name"], keep="last")
    clean_df.to_csv(REWARD_FILE, index=False, encoding="utf-8-sig")
    with db_conn() as conn:
        clean_df.to_sql("master_rewards_db", conn, if_exists="replace", index=False)

def evaluate_customer_rewards(waste_ratio_pct):
    df_r = get_live_rewards()
    if df_r.empty:
        return ["已累積 10 綠色環保積分"]

    active_rules = df_r[df_r["is_active"] == True].copy()
    if active_rules.empty:
        return ["已累積 10 綠色環保積分"]

    active_rules["max_waste_ratio"] = pd.to_numeric(active_rules["max_waste_ratio"], errors="coerce").fillna(100.0)
    matched = active_rules[active_rules["max_waste_ratio"] >= waste_ratio_pct].sort_values(by="max_waste_ratio", ascending=True)

    if not matched.empty:
        best_tier = matched.iloc[0]
        return [f"🎉 【{best_tier['tier_name']}】已派發至大家樂 App：{best_tier['reward_description']}"]
    else:
        return ["已完成還盤 (未符合獎勵門檻)"]

def save_record(r):
    with db_conn() as conn:
        conn.execute("""
            INSERT INTO audit_logs (
                timestamp, audit_date, audit_month, branch_name, member_id,
                dish_name, primary_waste, waste_ratio, cost_waste_hkd, co2_emission_kg, reward_issued
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r["timestamp"], r["audit_date"], r["audit_month"], r["branch_name"], r["member_id"],
            r["dish_name"], r["primary_waste"], r["waste_ratio"], r["cost_waste_hkd"], r["co2_emission_kg"], r["reward_issued"]
        ))
    try:
        current_df = get_records()
        current_df.to_csv(SEED_AUDIT_FILE, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"CSV Save Error: {e}")

def get_records():
    with db_conn() as conn: 
        df = pd.read_sql("SELECT * FROM audit_logs ORDER BY id DESC", conn)
        if not df.empty:
            if "member_id" not in df.columns: df["member_id"] = "STAFF"
            df["member_id"] = df["member_id"].fillna("STAFF")
        return df

# ==============================================================================
# 5. AI Engine (載入已完成凍結骨幹優化、推送到雲端的專屬模型)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_ai_engine():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 🌟 直接指向您剛上傳至 Hugging Face 的優化模型專案
    model_path = "kktlau115/trayzero-frozen-swin-model"
    
    print(f"📥 正在從 Hugging Face 載入 TrayZero+ 專屬優化 AI 模型: {model_path}...")
    
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForImageClassification.from_pretrained(model_path).to(dev)
    model.eval()
    
    clip_classifier = pipeline(
        "zero-shot-image-classification", 
        model="openai/clip-vit-base-patch32", 
        device=0 if torch.cuda.is_available() else -1
    )
    
    return {
        "processor": processor,
        "model": model,
        "clip": clip_classifier,
        "device": dev
    }

def detect_tray(image, engine, selected_dish="", carb_type_from_csv=""):
    image_rgb = image.convert("RGB")
    width, height = image_rgb.size
    
    # 透過微調後的 Swin-Base 迴歸模型進行精準殘食率預測
    inputs = engine["processor"](images=image_rgb, return_tensors="pt").to(engine["device"])
    with torch.no_grad():
        outputs = engine["model"](**inputs)
        raw_pred = outputs.logits.item() if outputs.logits.numel() == 1 else outputs.logits[0][0].item()
        
        # 迴歸模型直接輸出的數值即為 0.0 ~ 1.0 的殘食比例 (經過 Sigmoid 或直接截斷保護)
        ratio = float(1.0 / (1.0 + np.exp(-raw_pred)))
        ratio = max(0.0, min(1.0, ratio))

    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    items = []
    
    food_type_labels = [
        "leftover noodles or pasta in the bowl",
        "leftover rice on the plate",
        "leftover meat, fishballs or sausages",
        "leftover vegetables, green leaves or soup",
        "clean empty dish"
    ]
    type_res = engine["clip"](image_rgb, candidate_labels=food_type_labels)
    top_type = type_res[0]["label"]

    is_noodle_menu = any(kw in str(carb_type_from_csv) for kw in ["麵", "意粉", "粉", "Spaghetti", "Noodle"])
    
    if "noodles" in top_type or is_noodle_menu:
        primary = "主食殘留 (麵食) Carb Residual (Noodles)"
        accent_color = "#DC2626"
    elif "rice" in top_type or "飯" in str(carb_type_from_csv):
        primary = "主食殘留 (米飯) Carb Residual (Rice)"
        accent_color = "#DC2626"
    elif "meat" in top_type:
        primary = "肉類殘留 Meat Residual"
        accent_color = "#D97706"
    else:
        primary = "蔬菜/湯汁 Sides & Broth"
        accent_color = "#10B981"

    if ratio < 0.05:
        primary_cat = "光盤 Clean Plate"
        ratio = 0.0
    elif ratio > 0.85:
        primary_cat = "完整未動餐點 (未食用浪費)"
    else:
        primary_cat = primary

    box = [int(width * 0.1), int(height * 0.1), int(width * 0.9), int(height * 0.9)]
    draw.rectangle(box, outline=accent_color, width=4)
    draw.text((box[0] + 10, box[1] + 10), f"Swin-Base 優化模型預測殘食率: {ratio*100:.1f}%", fill=accent_color)
    
    items.append({
        "分類項目 Category": primary.split(" ")[0], 
        "置信度 Confidence": f"{type_res[0]['score']:.1%}", 
        "佔比 Coverage": f"{ratio*100:.1f}%"
    })

    return img_draw, items, ratio, primary_cat, True

def auto_detect_dish_clip(image, candidate_dishes, engine):
    if not candidate_dishes:
        return "未定義餐點", 0.0
    clean_labels = [d.strip() for d in candidate_dishes]
    try:
        results = engine["clip"](image.convert("RGB"), candidate_labels=clean_labels)
        return results[0]["label"], results[0]["score"]
    except Exception:
        return candidate_dishes[0], 0.75

# ==============================================================================
# 6. Member Profile Synthesis (Loyalty Engine)
# ==============================================================================
def analyze_member_loyalty_profile(member_id, df_all):
    if not member_id or member_id == "STAFF" or df_all.empty:
        return None
    
    m_df = df_all[df_all["member_id"] == member_id]
    if m_df.empty:
        return {
            "member_id": member_id,
            "total_visits": 0,
            "avg_waste": 0.0,
            "favorite_dish": "尚無資料",
            "pos_default_rice": "正常份量",
            "pos_default_sauce": "正常汁",
            "retarget_strategy": "發送迎新 $5 折扣券吸引二訪",
            "crm_segment": "新註冊會員 (New Sign-up)"
        }
    
    total_visits = len(m_df)
    avg_waste = m_df["waste_ratio"].mean()
    
    dish_stats = m_df.groupby("dish_name").agg(
        count=("waste_ratio", "count"),
        clean_waste=("waste_ratio", "mean")
    ).reset_index()
    fav_dish = dish_stats.sort_values(by=["count", "clean_waste"], ascending=[False, True]).iloc[0]["dish_name"]
    
    carb_wastes = m_df[m_df["primary_waste"].str.contains("主食", na=False)]
    if not carb_wastes.empty and carb_wastes["waste_ratio"].mean() > 20.0:
        pos_rice = "預設【少飯/少麵 (-30g / 立減 $2)】"
        crm_seg = "控醣輕食族 (Low-Carb Diners)"
    elif avg_waste < 10.0:
        pos_rice = "預設【正常份量 (光盤常客)】"
        crm_seg = "高飽足飽腹族 (Standard/High-Calorie)"
    else:
        pos_rice = "預設【標準份量】"
        crm_seg = "均衡飲食族 (Balanced Diners)"

    sauce_wastes = m_df[m_df["primary_waste"].str.contains("配菜|醬汁|湯汁", na=False)]
    if not sauce_wastes.empty and sauce_wastes["waste_ratio"].mean() > 25.0:
        pos_sauce = "預設【少汁 / 醬汁另上】"
    else:
        pos_sauce = "預設【正常汁】"

    if "控醣" in crm_seg:
        retarget_strategy = f"針對最愛餐點【{fav_dish}】推送「少飯少麵換特飲券」；推廣高蛋白輕食。"
    elif "高飽足" in crm_seg:
        retarget_strategy = f"向其大家樂 App 推送【{fav_dish}】加配小食（雞翼/紅豆冰）$8 組合券。"
    else:
        retarget_strategy = f"推送【{fav_dish}】午市立減 $3 現金回訪券，鎖定工作日高頻復購。"

    return {
        "member_id": member_id,
        "total_visits": total_visits,
        "avg_waste": avg_waste,
        "favorite_dish": fav_dish,
        "pos_default_rice": pos_rice,
        "pos_default_sauce": pos_sauce,
        "retarget_strategy": retarget_strategy,
        "crm_segment": crm_seg
    }

# ==============================================================================
# 7. Mode Renderers
# ==============================================================================
def render_header():
    st.markdown("""
    <div class="pos-header-banner">
        <div class="pos-header-title">🍽️ TrayZero+ 智能餐盤審計與會員獎勵系統 (Frozen Backbone Swin-Base Powered)</div>
    </div>
    """, unsafe_allow_html=True)

def render_mode1(engine, modules):
    df_b = get_live_branches()
    df_d = get_live_dishes()

    if df_b.empty or df_d.empty:
        st.warning("⚠️ 門市或餐點清單為空！請先至 Mode 3 上傳或新增菜單與分店。")
        return

    df_history = get_records()
    c1, c2 = st.columns([1.15, 0.85])
    
    with c1:
        st.markdown("#### 🏢 會員識別與還盤掃描 (Member Scan & Ingest)")
        b_name = st.selectbox("執勤門市 (Active Store Location)", df_b["name"].tolist())
        
        active_member_id = "STAFF"
        if modules.get("mod4", True):
            st.markdown("##### 📲 大家樂 Club 100 會員識別 (Member Scanner)")
            raw_member_id = st.text_input(
                "掃描或輸入會員卡號 / 手機號碼 (未輸入則系統預設為員工還盤)", 
                value=st.session_state.get("last_input_member", ""),
                placeholder="例: 001 / C100-8801 / 手機號碼 (若為員工還盤請留空)"
            )
            
            active_member_id = raw_member_id.strip() if raw_member_id.strip() else "STAFF"
            st.session_state["last_input_member"] = raw_member_id

            if active_member_id != "STAFF":
                prof = analyze_member_loyalty_profile(active_member_id, df_history)
                if prof:
                    st.markdown(f"""
                    <div style="background:#FFFBEB; border:1.5px solid #FDE68A; border-radius:10px; padding:12px 16px; margin-bottom:14px;">
                        <b style="color:#92400E; font-size:0.92rem;">💳 會員檔案: #{active_member_id} ({prof['crm_segment']} • 熟客回頭 {prof['total_visits']} 次)</b><br>
                        <span style="font-size:0.84rem; color:#78350F; font-weight:700;">歷史平均殘食: <b>{prof['avg_waste']:.1f}%</b> | 最喜愛餐點: <b>{prof['favorite_dish']}</b></span><br>
                        <span style="font-size:0.84rem; color:#D97706; font-weight:800;">🎯 點餐機 (Kiosk) 自動預載: {prof['pos_default_rice']} • {prof['pos_default_sauce']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("ℹ️ 目前狀態：【員工還盤 (STAFF)】（未輸入會員卡號）")

        auto_dish = st.checkbox("🤖 啟用 AI 自動辨識餐點類型 (Auto Dish Recognition via CLIP)", value=True)
        scan_mode = st.radio(
            "感應鏡頭作業模式", 
            ["🟢 Live 自動感應 (Auto-Scan)", "📸 手動快照 (Snapshot)", "📁 上傳照片 (Upload)"], 
            horizontal=True
        )
        
        img_cap = None
        should_run = False

        if scan_mode.startswith("🟢"):
            cam = st.camera_input("大家樂回收輸送台即時視頻長開中 (Live Monitor)", key="live_cam")
            if cam:
                img_cap = Image.open(cam).convert("RGB")
                h = hash(img_cap.tobytes()[:3000])
                if h != st.session_state.get("last_h"):
                    st.session_state["last_h"] = h
                    should_run = True
        elif scan_mode.startswith("📸"):
            m_cam = st.camera_input("快照拍攝 (Take Snapshot)", key="manual_cam")
            if m_cam: 
                img_cap = Image.open(m_cam).convert("RGB")
                should_run = True
        else:
            up = st.file_uploader("上傳餐盤相片 (Upload Image)", type=["jpg", "png", "jpeg"], key="tray_file_uploader")
            if up is not None:
                img_bytes = up.getvalue()
                current_file_hash = hash(img_bytes)
                img_cap = Image.open(up).convert("RGB")
                if current_file_hash != st.session_state.get("active_upload_hash"):
                    st.session_state["active_upload_hash"] = current_file_hash
                    should_run = True

        if img_cap is not None and should_run:
            with st.spinner("🚀 TrayZero+ 專屬優化 AI 模型正在進行精準殘食分析..."):
                candidate_names = df_d["name"].tolist()
                if auto_dish:
                    sel_dish, dish_conf = auto_detect_dish_clip(img_cap, candidate_names, engine)
                else:
                    sel_dish = candidate_names[0]
                    dish_conf = 1.0

                matched_rows = df_d[df_d["name"] == sel_dish]
                carb_type = matched_rows.iloc[0]["main_carb"] if not matched_rows.empty else "未知主食"

                anno_img, items, ratio, primary_cat, is_food = detect_tray(
                    img_cap, engine, selected_dish=sel_dish, carb_type_from_csv=carb_type
                )

                loss_hkd = round(ratio * 25 * 0.45, 1) if ratio > 0 else 0.0
                now = datetime.datetime.now()
                waste_pct = round(ratio * 100, 1)
                
                if modules.get("mod4", True) and active_member_id != "STAFF":
                    rewards_list = evaluate_customer_rewards(waste_pct)
                    reward_msg = " • ".join(rewards_list)
                else:
                    reward_msg = "員工還盤完成 (Staff Return - 僅記錄數據)"

                save_record({
                    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"), 
                    "audit_date": now.strftime("%Y-%m-%d"),
                    "audit_month": now.strftime("%Y-%m"), 
                    "branch_name": b_name, 
                    "member_id": active_member_id,
                    "dish_name": sel_dish, 
                    "primary_waste": primary_cat, 
                    "waste_ratio": waste_pct,
                    "cost_waste_hkd": loss_hkd, 
                    "co2_emission_kg": round(loss_hkd * 0.12, 2),
                    "reward_issued": reward_msg
                })

                st.session_state["latest"] = {
                    "img": anno_img, 
                    "dish": sel_dish, 
                    "conf": dish_conf,
                    "time": now.strftime("%H:%M:%S"),
                    "ratio": ratio, 
                    "cat": primary_cat, 
                    "cost": loss_hkd, 
                    "branch": b_name, 
                    "member": active_member_id,
                    "reward": reward_msg,
                    "items": items
                }
                st.toast(f"✅ 還盤數據已即時同步！")
                st.rerun()

    with c2:
        st.markdown("#### 🎯 即時判斷結果與獎勵 (Live Ticket)")
        latest = st.session_state.get("latest")
        if not latest:
            st.info("💡 尚未執行偵測。請對準餐盤拍照或上傳。")
        else:
            conf_str = f"({latest.get('conf', 1.0):.1%})" if 'conf' in latest else ""
            st.image(latest["img"], caption=f"🍽️️ {latest['dish']} {conf_str} • {latest['time']}")
            
            if modules.get("mod4", True) and latest.get("member") != "STAFF":
                st.markdown(f"""
                <div class="pos-coupon-card">
                    <div class="pos-coupon-header">🎟️ 大家樂 CLUB 100 電子現金券 (即時入帳)</div>
                    <div class="pos-coupon-value">{latest.get('reward').split('：')[-1] if '：' in latest.get('reward') else latest.get('reward')}</div>
                    <div class="pos-coupon-desc">• 關聯會員卡號: <code>{latest.get('member')}</code>  |  • 達成狀態: 惜食獎勵門檻達標</div>
                    <div class="pos-coupon-badge">已派送至手機大家樂錢包 (App Push Sent)</div>
                </div>
                """, unsafe_allow_html=True)
            elif latest.get("member") == "STAFF":
                st.markdown("""
                <div style="background:#F1F5F9; border:1px solid #CBD5E1; border-radius:10px; padding:10px 14px; margin-bottom:14px; font-size:0.85rem; color:#475569; font-weight:700;">
                    👤 員工還盤記錄完成 (不派發個人會員券)
                </div>
                """, unsafe_allow_html=True)

            k1, k2, k3 = st.columns(3)
            with k1:
                ratio_val = latest["ratio"]
                color_css = "#D97706" if ratio_val == 0.0 else ("#DC2626" if ratio_val > 0.3 else "#D97706")
                st.markdown(f"""
                <div class="pos-metric-card amber-glow">
                    <div class="pos-metric-label">殘食佔比 WASTE</div>
                    <div class="pos-metric-val" style="color:{color_css};">{ratio_val:.1%}</div>
                    <div style="font-size:0.75rem; color:#059669; font-weight:800; margin-top:4px;">{'🎉 達成光盤' if ratio_val == 0.0 else '需份量校準'}</div>
                </div>
                """, unsafe_allow_html=True)

            with k2:
                st.markdown(f"""
                <div class="pos-metric-card">
                    <div class="pos-metric-label">主要殘留 PRIMARY</div>
                    <div class="pos-metric-val" style="font-size:1.15rem; margin-top:4px; color:#0F172A;">{latest["cat"].split(' ')[0]}</div>
                    <div style="font-size:0.75rem; color:#64748B; font-weight:700; margin-top:6px;">{'完全吃淨' if ratio_val == 0.0 else '主食/配料'}</div>
                </div>
                """, unsafe_allow_html=True)

            with k3:
                st.markdown(f"""
                <div class="pos-metric-card">
                    <div class="pos-metric-label">推算損耗 LOSS</div>
                    <div class="pos-metric-val" style="color:#0F172A;">HK${latest["cost"]}</div>
                    <div style="font-size:0.75rem; color:#059669; font-weight:800; margin-top:4px;">{'零浪費標準' if ratio_val == 0.0 else '單盤損耗'}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="pos-directive-card">
                <b style="color:#0F172A; font-size:0.9rem;">👨‍🍳 大家樂後廚計量校準 (Kitchen Advisory)</b><br>
                <span style="font-size:0.8rem; color:#475569; font-weight:700;">• 餐點【{latest['dish'].split(' ')[0]}】殘食率為 {latest['ratio']:.1%}，由雲端優化 Swin-Base 迴歸模型運算。</span><br>
                <span style="font-size:0.8rem; color:#475569; font-weight:700;">• 系統已自動將交易流水與積分寫入 seed_audit_logs.csv 與資料庫。</span>
            </div>
            """, unsafe_allow_html=True)

# ==============================================================================
# Mode 2: 總部即時營運大盤 (支援防禦性空狀態與指定 Period 匯出)
# ==============================================================================
def render_mode2(engine, modules):
    df_b = get_live_branches()
    df_d = get_live_dishes()
    df_raw = get_records()

    st.markdown("### 📊 總部即時營運大盤與會員客群 Retargeting 數據中心")

    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        period_filter = st.radio(
            "統計時間維度 (Period)",
            ["⚡ 本日 (Today)", "📅 本周 (This Week)", "🗓 本月 (This Month)", "📈 本年度 (This Year)", "🌐 全部歷史 (All Time)"],
            horizontal=True,
            index=4
        )
    with col_ctrl2:
        st.write("")
        if st.button("🔄 刷新即時數據 (Reload Live Data)"):
            st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        b_filter = st.selectbox(
            "1. 門市維度過濾", 
            ["🌐 全部分店"] + df_b["name"].tolist() if not df_b.empty else ["🌐 全部分店"]
        )
        sel_b = "ALL" if "全部" in b_filter else b_filter
    with c2:
        d_filter = st.selectbox(
            "2. 食物種類維度過濾", 
            ["🍱 全部餐點品項"] + df_d["name"].tolist() if not df_d.empty else ["🍱 全部餐點品項"]
        )
        sel_d = "ALL" if "全部" in d_filter else d_filter

    df_filtered = df_raw.copy()

    now_dt = datetime.datetime.now()
    if not df_filtered.empty and "timestamp" in df_filtered.columns:
        df_filtered["parsed_dt"] = pd.to_datetime(df_filtered["timestamp"], errors="coerce")
        
        if "本日" in period_filter:
            today_str = now_dt.strftime("%Y-%m-%d")
            df_filtered = df_filtered[df_filtered["parsed_dt"].dt.strftime("%Y-%m-%d") == today_str]
        elif "本周" in period_filter:
            start_of_week = now_dt - datetime.timedelta(days=now_dt.weekday())
            start_of_week_date = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
            df_filtered = df_filtered[df_filtered["parsed_dt"] >= start_of_week_date]
        elif "本月" in period_filter:
            cur_month_str = now_dt.strftime("%Y-%m")
            df_filtered = df_filtered[df_filtered["parsed_dt"].dt.strftime("%Y-%m") == cur_month_str]
        elif "本年度" in period_filter:
            cur_year_str = str(now_dt.year)
            df_filtered = df_filtered[df_filtered["parsed_dt"].dt.strftime("%Y") == cur_year_str]

    if not df_filtered.empty:
        if sel_b != "ALL": 
            df_filtered = df_filtered[df_filtered["branch_name"] == sel_b]
        if sel_d != "ALL": 
            df_filtered = df_filtered[df_filtered["dish_name"] == sel_d]

    n = len(df_filtered)

    if n == 0:
        st.markdown(f"""
        <div class="empty-state-box">
            <div style="font-size: 2.5rem; margin-bottom: 10px;">📭</div>
            <div style="font-size: 1.2rem; font-weight: 800; color: #0F172A; margin-bottom: 6px;">
                目前選定的門市或時間區間尚無審計數據
            </div>
            <div style="font-size: 0.9rem; color: #64748B; font-weight: 600;">
                執勤分店：<b>{b_filter}</b> | 時間區間：<b>{period_filter}</b><br>
                請前往 <b>Mode 1: 前線回收感應台</b> 執行實體餐盤掃描。
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    avg_w = df_filtered["waste_ratio"].mean()
    tot_hkd = df_filtered["cost_waste_hkd"].sum()
    tot_co2 = df_filtered["co2_emission_kg"].sum()

    if "reward_issued" in df_filtered.columns:
        coupon_records = df_filtered[df_filtered["reward_issued"].str.contains("券|獎|積分", na=False)]
        total_coupons = len(coupon_records[coupon_records["member_id"] != "STAFF"]) if not coupon_records.empty else 0
    else:
        total_coupons = 0

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f'<div class="pos-metric-card"><div class="pos-metric-label">審計樣本盤數</div><div class="pos-metric-val">{n} <span style="font-size:0.85rem;color:#94A3B8">TRAYS</span></div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="pos-metric-card amber-glow"><div class="pos-metric-label">平均殘食率</div><div class="pos-metric-val" style="color:{"#DC2626" if avg_w > 25 else "#D97706"}">{avg_w:.1f}%</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="pos-metric-card"><div class="pos-metric-label">食材損耗總額</div><div class="pos-metric-val" style="color:#D97706">HK${tot_hkd:,.1f}</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="pos-metric-card"><div class="pos-metric-label">累計碳排放</div><div class="pos-metric-val" style="color:#2563EB">{tot_co2:.2f} <span style="font-size:0.85rem;color:#94A3B8">kg</span></div></div>', unsafe_allow_html=True)
    with k5:
        st.markdown(f'<div class="pos-metric-card green-glow"><div class="pos-metric-label">已派發會員券數</div><div class="pos-metric-val" style="color:#059669">{total_coupons} <span style="font-size:0.85rem;color:#94A3B8">張</span></div></div>', unsafe_allow_html=True)

    st.markdown("---")

    chart_c1, chart_c2 = st.columns(2)

    with chart_c1:
        st.markdown('<div class="chart-box"><div class="chart-title">🍱 剩菜最多餐品排行 (Top Waste by Dish)</div>', unsafe_allow_html=True)
        if "dish_name" in df_filtered.columns:
            dish_waste = df_filtered.groupby("dish_name").agg(
                avg_waste=("waste_ratio", "mean"),
                total_loss=("cost_waste_hkd", "sum"),
                tray_count=("id", "count")
            ).reset_index().sort_values(by="avg_waste", ascending=False)
            
            dish_waste["avg_waste_pct"] = dish_waste["avg_waste"].round(1)
            
            chart_dish = alt.Chart(dish_waste).mark_bar(cornerRadius=6, color="#D97706").encode(
                x=alt.X('avg_waste_pct:Q', title='平均殘食率 (%)'),
                y=alt.Y('dish_name:N', sort='-x', title='餐點品項'),
                tooltip=['dish_name', 'avg_waste_pct', 'total_loss', 'tray_count']
            ).properties(height=260)
            
            st.altair_chart(chart_dish, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_c2:
        st.markdown('<div class="chart-box"><div class="chart-title">📈 平均殘食率時間趨勢 (Waste Ratio Trending)</div>', unsafe_allow_html=True)
        if "parsed_dt" in df_filtered.columns and df_filtered["parsed_dt"].notna().any():
            trend_df = df_filtered.dropna(subset=["parsed_dt"]).copy()
            
            if "本日" in period_filter:
                trend_df["time_group"] = trend_df["parsed_dt"].dt.strftime("%H:00")
            elif "本周" in period_filter or "本月" in period_filter:
                trend_df["time_group"] = trend_df["parsed_dt"].dt.strftime("%m-%d")
            else:
                trend_df["time_group"] = trend_df["parsed_dt"].dt.strftime("%Y-%m")

            trend_grouped = trend_df.groupby("time_group")["waste_ratio"].mean().reset_index()
            trend_grouped["waste_ratio"] = trend_grouped["waste_ratio"].round(1)

            chart_trend = alt.Chart(trend_grouped).mark_line(
                point=alt.OverlayMarkDef(color="#DC2626", size=60),
                color="#DC2626",
                strokeWidth=3
            ).encode(
                x=alt.X('time_group:N', title='時間時間軸'),
                y=alt.Y('waste_ratio:Q', title='殘食率 (%)', scale=alt.Scale(domain=[0, 100])),
                tooltip=['time_group', 'waste_ratio']
            ).properties(height=260)

            st.altair_chart(chart_trend, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📥 匯出指定 Period 審計與客群資料 (Data Export Engine)")

    with st.container():
        st.markdown('<div class="export-control-box">', unsafe_allow_html=True)
        col_exp_mode, col_exp_dates = st.columns([1.2, 2.8])
        
        with col_exp_mode:
            export_mode = st.radio(
                "選擇匯出區間模式 (Export Scope)",
                ["連動上方週期 (" + period_filter.split(" ")[1] + ")", "自訂自選日期範圍 (Custom Range)"]
            )
        
        df_for_export = df_raw.copy()
        export_filename_tag = ""

        if not df_for_export.empty and "timestamp" in df_for_export.columns:
            df_for_export["parsed_dt"] = pd.to_datetime(df_for_export["timestamp"], errors="coerce")
            
            if "連動上方週期" in export_mode:
                df_final_export = df_filtered.copy()
                export_filename_tag = period_filter.split(" ")[1]
            else:
                with col_exp_dates:
                    d_col1, d_col2 = st.columns(2)
                    min_date = df_for_export["parsed_dt"].min().date() if df_for_export["parsed_dt"].notna().any() else datetime.date.today()
                    max_date = datetime.date.today()
                    with d_col1:
                        start_d = st.date_input("開始日期 (Start Date)", value=min_date)
                    with d_col2:
                        end_d = st.date_input("結束日期 (End Date)", value=max_date)

                start_dt = datetime.datetime.combine(start_d, datetime.time.min)
                end_dt = datetime.datetime.combine(end_d, datetime.time.max)
                df_final_export = df_for_export[(df_for_export["parsed_dt"] >= start_dt) & (df_for_export["parsed_dt"] <= end_dt)]
                export_filename_tag = f"{start_d}_to_{end_d}"
        else:
            df_final_export = pd.DataFrame()
            export_filename_tag = "empty"

        st.caption(f"📊 目前準備匯出筆數：**{len(df_final_export)}** 筆記錄")

        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if not df_final_export.empty:
                clean_csv_df = df_final_export.drop(columns=["parsed_dt"], errors="ignore")
                csv_bytes = clean_csv_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label=f"📥 匯出【指定期間審計日誌】CSV ({export_filename_tag})",
                    data=csv_bytes,
                    file_name=f"trayzero_audit_{export_filename_tag}.csv",
                    mime="text/csv",
                    type="primary"
                )
            else:
                st.button("📥 暫無資料可匯出", disabled=True)

        with btn_c2:
            if not df_final_export.empty:
                unique_members_in_period = [m for m in df_final_export["member_id"].dropna().unique().tolist() if m != "STAFF"]
                if unique_members_in_period:
                    crm_period_records = []
                    for mid in unique_members_in_period:
                        prof = analyze_member_loyalty_profile(mid, df_final_export)
                        crm_period_records.append({
                            "會員卡號 (Member ID)": prof["member_id"],
                            "客群分類 (Segment)": prof["crm_segment"],
                            "區間最愛餐點": prof["favorite_dish"],
                            "區間平均殘食率": f"{prof['avg_waste']:.1f}%",
                            "建議點餐預設": prof["pos_default_rice"],
                            "建議推播優惠券": prof["retarget_strategy"]
                        })
                    df_crm_period = pd.DataFrame(crm_period_records)
                    csv_crm_bytes = df_crm_period.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label=f"📥 匯出【會員 Retargeting 數據包】CSV ({len(unique_members_in_period)}人)",
                        data=csv_crm_bytes,
                        file_name=f"cdc_loyalty_{export_filename_tag}.csv",
                        mime="text/csv"
                    )
                else:
                    st.button("ℹ️ 該期間無會員記錄 (均為員工)", disabled=True)
            else:
                st.button("📥 暫無會員資料", disabled=True)

        st.markdown('</div>', unsafe_allow_html=True)

def render_mode3():
    st.markdown("### ⚙️ 基礎資料管理 (Master Data Management - Live Cloud Store)")
    st.caption("支援連接 Google Drive / Google Sheets 發佈的 Live CSV 網址，系統每次刷新皆從雲端即時拉取。")

    tab_cloud, tab1, tab2, tab3 = st.tabs([
        "☁️ Google Drive / Sheets Live CSV 連接",
        "🏢 分店清單 (Branches CSV)", 
        "🍱 餐點品項管理 (Menu CSV)",
        "🎁 殘食門檻獎勵階梯配置 (Incentive Tiers CSV)"
    ])

    with tab_cloud:
        st.markdown("#### ☁️ Google Drive / Google Sheets Live CSV 雲端數據源配置")
        current_urls = get_cloud_urls()
        c_base = st.text_input("Google Drive 主發佈 CSV 網址 (Base Live CSV URL)", value=current_urls.get("base_url", DEFAULT_BASE_GDRIVE_URL))
        c_url1 = st.text_input("1. 分店清單 Live CSV 網址", value=current_urls.get("branches_url", f"{c_base}&sheet=branches"))
        c_url2 = st.text_input("2. 菜單清單 Live CSV 網址", value=current_urls.get("dishes_url", f"{c_base}&sheet=dishes"))
        c_url3 = st.text_input("3. 獎勵規則 Live CSV 網址", value=current_urls.get("rewards_url", f"{c_base}&sheet=rewards"))

        if st.button("💾 儲存並啟用 Google Drive Live 連接 (Save Cloud Sync)", type="primary"):
            save_cloud_urls({
                "base_url": c_base,
                "branches_url": c_url1,
                "dishes_url": c_url2,
                "rewards_url": c_url3
            })
            st.success("✅ Google Drive 雲端 Live CSV 連結已成功儲存與生效！")
            st.rerun()

    with tab1:
        df_b_current = get_live_branches()
        st.markdown("#### 🏢 門市清單即時編輯 (Live Branches)")
        edit_b = st.data_editor(df_b_current, num_rows="dynamic", key="live_branch_editor")
        if st.button("💾 儲存修改至 master_branches.csv", type="primary"):
            save_live_branches(edit_b)
            st.success("✅ 門市資料已更新！")
            st.rerun()

    with tab2:
        df_d_current = get_live_dishes()
        st.markdown("#### 🍱 現有餐點清單即時編輯 (Live Menu CSV)")
        edit_d = st.data_editor(df_d_current, num_rows="dynamic", key="live_dish_editor")
        if st.button("💾 儲存修改至 master_dishes.csv", type="primary"):
            save_live_dishes(edit_d)
            st.success("✅ 菜單資料已更新！")
            st.rerun()

    with tab3:
        df_r_current = get_live_rewards()
        st.markdown("#### 🎁 會員還盤獎勵階梯配置 (Incentive Tiers Configuration)")
        edit_r = st.data_editor(df_r_current, num_rows="dynamic", key="live_reward_editor")
        if st.button("💾 儲存修改至 master_rewards.csv", type="primary"):
            save_live_rewards(edit_r)
            st.success("✅ 獎勵規則已更新！")
            st.rerun()

# ==============================================================================
# 8. Application Entry Point
# ==============================================================================
def main():
    inject_safe_css()
    init_db()

    with st.spinner("🚀 正在載入 TrayZero+ 專屬優化 Swin-Base 視覺 AI 引擎..."):
        engine = load_ai_engine()

    logo_target = LOGO_FILE_PNG if os.path.exists(LOGO_FILE_PNG) else (LOGO_FILE_JPG if os.path.exists(LOGO_FILE_JPG) else None)
    if logo_target:
        st.sidebar.image(logo_target, width=175)

    st.sidebar.markdown("""
    <div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:10px; padding:10px 14px; margin-bottom:16px;">
        <div style="font-size:0.75rem; font-weight:800; color:#92400E;">現正執勤門市 (STORE)</div>
        <div style="font-size:0.95rem; font-weight:900; color:#78350F; margin-top:2px;">中環威靈頓街店</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("##### 觸控模式選擇 (TOUCH NAVIGATION)")
    mode = st.sidebar.radio(
        label="模式選擇導航",
        options=[
            "Mode 1: 前線回收感應台",
            "Mode 2: 總部即時營運大盤",
            "Mode 3: 菜單與獎勵配置"
        ],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("##### 企業模組狀態 (MODULES)")
    mod_1 = st.sidebar.checkbox("M1: 營運監控 (Ops Core)", value=True)
    mod_2 = st.sidebar.checkbox("M2: 深度分析 (BI Analytics)", value=True)
    mod_3 = st.sidebar.checkbox("M3: 精準營銷 (Smart POS)", value=True)
    mod_4 = st.sidebar.checkbox("M4: 會員閉環 (Loyalty Loop)", value=True)

    active_modules = {
        "mod1": mod_1,
        "mod2": mod_2,
        "mod3": mod_3,
        "mod4": mod_4
    }

    render_header()

    if mode.startswith("Mode 1"): 
        render_mode1(engine, active_modules)
    elif mode.startswith("Mode 2"): 
        render_mode2(engine, active_modules)
    else: 
        render_mode3()

if __name__ == "__main__":
    main()
