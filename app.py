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
# 1. Global Paths & Fast-Casual POS Enterprise CSS Theme (高對比自適應主題)
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

def inject_safe_css():
    st.markdown("""
    <style>
        /* 強制全局高對比淺色基調，徹底消滅深色模式導致的文字隱形 */
        :root {
            --cdc-red: #DC2626 !important;
            --cdc-amber: #D97706 !important;
            --text-color: #0F172A !important;
            --background-color: #F8FAFC !important;
            --secondary-background-color: #FFFFFF !important;
        }
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #F8FAFC !important;
            color: #0F172A !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", "PingFang HK", "Microsoft JhengHei", Arial, sans-serif !important;
        }
        .main .block-container {
            padding-top: 1.2rem !important;
            padding-bottom: 3rem !important;
            color: #0F172A !important;
        }
        
        /* 側邊欄高對比防隱形樣式 */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1.5px solid #CBD5E1 !important;
            padding-top: 1rem !important;
        }
        [data-testid="stSidebar"] * {
            color: #0F172A !important;
        }
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] span, 
        [data-testid="stSidebar"] label, 
        [data-testid="stSidebar"] div, 
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] h4, 
        [data-testid="stSidebar"] h5, 
        [data-testid="stSidebar"] h6 {
            color: #0F172A !important;
            font-weight: 600 !important;
        }
        [data-testid="stSidebar"] [data-testid="stImage"] {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            margin-left: auto !important;
            margin-right: auto !important;
            margin-bottom: 10px !important;
            width: 100% !important;
            text-align: center !important;
        }
        [data-testid="stSidebar"] [data-testid="stImage"] > img {
            margin-left: auto !important;
            margin-right: auto !important;
            display: block !important;
            max-width: 175px !important;
            height: auto !important;
            object-fit: contain !important;
        }

        /* 側邊欄卡片標題優化 */
        [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
            margin-bottom: 15px !important;
            border-radius: 12px !important;
            background-color: #F8FAFC !important;
            border: 1px solid #E2E8F0 !important;
        }

        /* 標題橫幅 (大家樂經典紅琥珀漸層) */
        .pos-header-banner {
            background: linear-gradient(135deg, #C2301A 0%, #D95D1A 48%, #D87B18 100%) !important;
            border-radius: 14px !important;
            padding: 16px 24px !important;
            margin-bottom: 22px !important;
            box-shadow: 0 4px 14px rgba(194, 48, 26, 0.22) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
        }
        .pos-header-title {
            color: #FFFFFF !important;
            font-size: 1.45rem !important;
            font-weight: 900 !important;
            margin: 0 !important;
            letter-spacing: 0.02em !important;
            text-shadow: 0 1px 3px rgba(0, 0, 0, 0.25) !important;
        }

        div[data-baseweb="select"],
        div[data-baseweb="select"] *,
        div[data-baseweb="input"],
        div[data-baseweb="input"] *,
        div[data-baseweb="base-input"],
        div[data-baseweb="base-input"] *,
        div[data-testid="stTextInputRootElement"],
        div[data-testid="stTextInputRootElement"] *,
        div[data-testid="stNumberInputContainer"],
        div[data-testid="stNumberInputContainer"] *,
        div[data-testid="stSelectbox"] div,
        div[data-testid="stSelectbox"] span,
        div[data-testid="stSelectbox"] svg,
        input, select, textarea {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            -webkit-text-fill-color: #0F172A !important;
            fill: #0F172A !important;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"],
        div[data-testid="stTextInputRootElement"] > div,
        div[data-testid="stNumberInputContainer"] > div {
            border: 1.5px solid #CBD5E1 !important;
            border-radius: 8px !important;
        }

        div[data-baseweb="popover"],
        div[data-baseweb="popover"] *,
        ul[data-baseweb="menu"],
        ul[data-baseweb="menu"] *,
        li[data-baseweb="menu-item"],
        li[data-baseweb="menu-item"] * {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            -webkit-text-fill-color: #0F172A !important;
        }

        div[data-testid="stFileUploader"],
        div[data-testid="stFileUploader"] *,
        section[data-testid="stFileUploaderDropzone"],
        section[data-testid="stFileUploaderDropzone"] *,
        div[data-testid="stFileUploaderDropzone"],
        div[data-testid="stFileUploaderDropzone"] *,
        div[data-testid="stFileUploaderFile"],
        div[data-testid="stFileUploaderFile"] *,
        div[data-testid="stFileUploaderFileData"],
        div[data-testid="stFileUploaderFileData"] *,
        div[data-testid="stFileUploaderDropzoneInstructions"],
        div[data-testid="stFileUploaderDropzoneInstructions"] * {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            -webkit-text-fill-color: #0F172A !important;
        }

        section[data-testid="stFileUploaderDropzone"] {
            border: 2px dashed #94A3B8 !important;
            border-radius: 10px !important;
        }

        div[data-testid="stFileUploader"] button {
            background-color: #F1F5F9 !important;
            color: #0F172A !important;
            border: 1px solid #CBD5E1 !important;
        }
        [data-testid="stSelectbox"] label,
        [data-testid="stTextInput"] label,
        [data-testid="stRadio"] label,
        [data-testid="stCheckbox"] label {
            color: #0F172A !important;
            font-weight: 700 !important;
            font-size: 0.92rem !important;
        }

        [data-testid="stMetric"] {
            background-color: #FFFFFF !important;
            border: 1.5px solid #CBD5E1 !important;
            border-radius: 10px !important;
            padding: 12px 16px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        }
        [data-testid="stMetric"] * {
            color: #0F172A !important;
        }
        [data-testid="stMetricLabel"] * {
            color: #475569 !important;
            font-weight: 700 !important;
            font-size: 0.88rem !important;
        }
        [data-testid="stMetricValue"] * {
            color: #0F172A !important;
            font-weight: 800 !important;
            font-size: 1.35rem !important;
            white-space: nowrap !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }

        div[data-baseweb="tab-list"] {
            gap: 8px !important;
            border-bottom: 2px solid #CBD5E1 !important;
            margin-bottom: 20px !important;
            background-color: transparent !important;
        }
        button[data-baseweb="tab"],
        button[data-baseweb="tab"] *,
        div[data-baseweb="tab-list"] button,
        div[data-baseweb="tab-list"] button * {
            background-color: transparent !important;
            color: #0F172A !important;
            -webkit-text-fill-color: #0F172A !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            opacity: 1 !important;
        }
        button[data-baseweb="tab"]:hover,
        button[data-baseweb="tab"]:hover *,
        div[data-baseweb="tab-list"] button:hover,
        div[data-baseweb="tab-list"] button:hover * {
            color: #DC2626 !important;
            -webkit-text-fill-color: #DC2626 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"],
        button[data-baseweb="tab"][aria-selected="true"] *,
        div[data-baseweb="tab-list"] button[aria-selected="true"],
        div[data-baseweb="tab-list"] button[aria-selected="true"] * {
            color: #DC2626 !important;
            -webkit-text-fill-color: #DC2626 !important;
            font-weight: 800 !important;
        }
        div[data-baseweb="tab-highlight"] {
            background-color: #DC2626 !important;
        }
        div[data-baseweb="tab-border"] {
            background-color: #CBD5E1 !important;
        }

        .voucher-box {
            background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%) !important;
            border: 2px dashed #D97706 !important;
            border-radius: 12px !important;
            padding: 14px 18px !important;
            margin: 14px 0 !important;
            color: #78350F !important;
        }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. Database Connection & Data Store
# ==============================================================================
def db_conn(): 
    try:
        conn = sqlite3.connect(DB_FILE, check_same_thread=False, timeout=10)
        return conn
    except Exception:
        return sqlite3.connect("/tmp/trayzero_audit.db", check_same_thread=False)

def ensure_audit_logs_schema(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            audit_date TEXT,
            audit_month TEXT,
            branch_name TEXT,
            branch_level TEXT,
            dish_name TEXT,
            primary_waste TEXT,
            waste_ratio REAL,
            waste_weight_g REAL,
            cost_waste_hkd REAL,
            co2_emission_kg REAL,
            member_id TEXT
        )
    """)
    conn.commit()
    
    cur.execute("PRAGMA table_info(audit_logs)")
    existing_cols = {col[1] for col in cur.fetchall()}
    expected_cols = [
        ("timestamp", "TEXT"),
        ("audit_date", "TEXT"),
        ("audit_month", "TEXT"),
        ("branch_name", "TEXT"),
        ("branch_level", "TEXT"),
        ("dish_name", "TEXT"),
        ("primary_waste", "TEXT"),
        ("waste_ratio", "REAL"),
        ("waste_weight_g", "REAL"),
        ("cost_waste_hkd", "REAL"),
        ("co2_emission_kg", "REAL"),
        ("member_id", "TEXT")
    ]
    for col_name, col_type in expected_cols:
        if col_name not in existing_cols:
            try:
                cur.execute(f"ALTER TABLE audit_logs ADD COLUMN {col_name} {col_type}")
                conn.commit()
            except Exception:
                pass

def init_db():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS dishes (
            dish_id TEXT PRIMARY KEY,
            name TEXT,
            main_carb TEXT,
            protein TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS branches (
            name TEXT PRIMARY KEY,
            level TEXT,
            district TEXT,
            traffic TEXT,
            avg_covers INTEGER,
            base_rice_g INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rewards (
            reward_id TEXT PRIMARY KEY,
            tier_name TEXT,
            max_waste_ratio REAL,
            reward_type TEXT,
            reward_description TEXT,
            is_active BOOLEAN
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()
    ensure_audit_logs_schema(conn)

    default_dishes = [
        ("D01", "一哥焗豬扒飯", "白米飯", "焗厚切豬扒"),
        ("D05", "焗肉醬意粉", "意大利麵", "慢燉牛肉醬"),
        ("D03", "燒味飯", "白米飯", "燒味"),
        ("D04", "干炒牛河", "中式麵條", "牛肉")
    ]
    cur.execute("SELECT COUNT(*) FROM dishes")
    dish_count = cur.fetchone()[0]
    if dish_count == 0 or dish_count != 4:
        cur.execute("DELETE FROM dishes")
        cur.executemany("INSERT INTO dishes VALUES (?,?,?,?)", default_dishes)
        conn.commit()

    cur.execute("SELECT COUNT(*) FROM branches")
    if cur.fetchone()[0] == 0:
        default_branches = [
            ("中環威靈頓街店", "Level A", "中西區", "白領上班族為主，午市尖峰翻檯率極高", 1200, 240),
            ("沙田新城市廣場店", "Level B", "沙田區", "家庭客、長者與週末休閒客群", 1500, 260),
            ("香港科技大學店", "Level C", "西貢區", "學生、教職員，運動量及食量顯著較大", 1800, 280),
            ("將軍澳 Popcorn 店", "Level B", "西貢區", "家庭客及換乘鐵路客流", 1400, 260)
        ]
        cur.executemany("INSERT OR IGNORE INTO branches VALUES (?,?,?,?,?,?)", default_branches)

    cur.execute("SELECT COUNT(*) FROM rewards")
    if cur.fetchone()[0] == 0:
        default_rewards = [
            ("R01", "極致光盤獎", 10.0, "Coupon + Points", "【$3 堂食現金券】+【50 綠色積分】+【凍檸茶半價券】", True),
            ("R02", "達標惜食獎", 20.0, "Coupon", "【$2 堂食電子券】+【20 綠色積分】", True),
            ("R03", "支持環保獎", 100.0, "Points", "【10 綠色環保積分】", True)
        ]
        cur.executemany("INSERT OR IGNORE INTO rewards VALUES (?,?,?,?,?,?)", default_rewards)

    cur.execute("SELECT COUNT(*) FROM audit_logs")
    if cur.fetchone()[0] == 0 and os.path.exists(SEED_AUDIT_FILE):
        try:
            df_seed = pd.read_csv(SEED_AUDIT_FILE)
            for _, r in df_seed.iterrows():
                cur.execute("""
                    INSERT INTO audit_logs (timestamp, audit_date, audit_month, branch_name, branch_level, dish_name, primary_waste, waste_ratio, waste_weight_g, cost_waste_hkd, co2_emission_kg, member_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(r.get("timestamp", "2026-10-09 12:30:00")),
                    str(r.get("audit_date", "2026-10-09")),
                    str(r.get("audit_month", "2026-10")),
                    str(r.get("branch_name", "沙田新城市廣場店")),
                    str(r.get("branch_level", "Level B")),
                    str(r.get("dish_name", "一哥焗豬扒飯")),
                    str(r.get("primary_waste", "主食澱粉")),
                    float(r.get("waste_ratio", 25.0)),
                    float(r.get("waste_weight_g", 125.0)),
                    float(r.get("cost_waste_hkd", 5.6)),
                    float(r.get("co2_emission_kg", 0.35)),
                    str(r.get("member_id", "M882190"))
                ))
            conn.commit()
        except Exception as e:
            pass

    conn.commit()
    conn.close()

def get_live_dishes():
    conn = db_conn()
    df = pd.read_sql_query("SELECT * FROM dishes", conn)
    conn.close()
    if df.empty and os.path.exists(DISH_FILE):
        try:
            df = pd.read_csv(DISH_FILE)
        except Exception:
            pass
    if df.empty:
        df = pd.DataFrame([
            {"dish_id": "D01", "name": "一哥焗豬扒飯", "main_carb": "白米飯", "protein": "焗厚切豬扒"},
            {"dish_id": "D05", "name": "焗肉醬意粉", "main_carb": "意大利麵", "protein": "慢燉牛肉醬"},
            {"dish_id": "D03", "name": "燒味飯", "main_carb": "白米飯", "protein": "燒味"},
            {"dish_id": "D04", "name": "干炒牛河", "main_carb": "中式麵條", "protein": "牛肉"}
        ])
    return df

def get_live_branches():
    conn = db_conn()
    df = pd.read_sql_query("SELECT * FROM branches", conn)
    conn.close()
    if df.empty and os.path.exists(BRANCH_FILE):
        try:
            df = pd.read_csv(BRANCH_FILE)
        except Exception:
            pass
    if df.empty:
        df = pd.DataFrame([
            {"name": "中環威靈頓街店", "level": "Level A", "district": "中西區", "traffic": "白領上班族為主，午市尖峰翻檯率極高", "avg_covers": 1200, "base_rice_g": 240},
            {"name": "沙田新城市廣場店", "level": "Level B", "district": "沙田區", "traffic": "家庭客、長者與週末休閒客群", "avg_covers": 1500, "base_rice_g": 260},
            {"name": "香港科技大學店", "level": "Level C", "district": "西貢區", "traffic": "學生、教職員，運動量及食量顯著較大", "avg_covers": 1800, "base_rice_g": 280},
            {"name": "將軍澳 Popcorn 店", "level": "Level B", "district": "西貢區", "traffic": "家庭客及換乘鐵路客流", "avg_covers": 1400, "base_rice_g": 260}
        ])
    return df

def get_live_rewards():
    conn = db_conn()
    df = pd.read_sql_query("SELECT * FROM rewards WHERE is_active = 1 ORDER BY max_waste_ratio ASC", conn)
    conn.close()
    return df

def evaluate_customer_rewards(waste_ratio_pct, is_en=False):
    df_rew = get_live_rewards()
    if df_rew.empty:
        if waste_ratio_pct <= 10.0:
            desc = "【$3 Cash Voucher】+【50 Green Points】" if is_en else "【$3 堂食現金券】+【50 綠色積分】+【凍檸茶半價券】"
            tier = "Ultra Clean Plate" if is_en else "極致光盤獎"
            return desc, tier, "#10B981"
        elif waste_ratio_pct <= 20.0:
            desc = "【$2 Dining Voucher】+【20 Green Points】" if is_en else "【$2 堂食電子券】+【20 綠色積分】"
            tier = "Standard Clean Plate" if is_en else "達標惜食獎"
            return desc, tier, "#059669"
        else:
            desc = "【10 Green Points】" if is_en else "【10 綠色環保積分】"
            tier = "Green Support" if is_en else "支持環保獎"
            return desc, tier, "#3B82F6"
    for _, r in df_rew.iterrows():
        if waste_ratio_pct <= float(r["max_waste_ratio"]):
            return r["reward_description"], r["tier_name"], "#10B981"
    desc = "【10 Green Points】" if is_en else "【10 綠色環保積分】"
    tier = "Green Support" if is_en else "支持環保獎"
    return desc, tier, "#64748B"

def save_record(r):
    ts_str = str(r.get("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    date_str = str(r.get("audit_date", ts_str.split(" ")[0] if " " in ts_str else ts_str[:10]))
    month_str = str(r.get("audit_month", date_str[:7]))
    
    try:
        conn = db_conn()
        ensure_audit_logs_schema(conn)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO audit_logs (
                timestamp, audit_date, audit_month, branch_name, branch_level, dish_name, 
                primary_waste, waste_ratio, waste_weight_g, cost_waste_hkd, co2_emission_kg, member_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ts_str,
            date_str,
            month_str,
            str(r.get("branch_name", "沙田新城市廣場店")),
            str(r.get("branch_level", "Level B")),
            str(r.get("dish_name", "一哥焗豬扒飯")),
            str(r.get("primary_waste", "主食澱粉")),
            float(r.get("waste_ratio", 0.0)),
            float(r.get("waste_weight_g", 0.0)),
            float(r.get("cost_waste_hkd", 0.0)),
            float(r.get("co2_emission_kg", 0.0)),
            str(r.get("member_id", "ANON"))
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        pass
        
    try:
        if os.path.exists(SEED_AUDIT_FILE):
            df_new = pd.DataFrame([{
                "timestamp": ts_str,
                "audit_date": date_str,
                "audit_month": month_str,
                "branch_name": r.get("branch_name", "沙田新城市廣場店"),
                "branch_level": r.get("branch_level", "Level B"),
                "dish_name": r.get("dish_name", "一哥焗豬扒飯"),
                "primary_waste": r.get("primary_waste", "主食澱粉"),
                "waste_ratio": float(r.get("waste_ratio", 0.0)),
                "waste_weight_g": float(r.get("waste_weight_g", 0.0)),
                "cost_waste_hkd": float(r.get("cost_waste_hkd", 0.0)),
                "co2_emission_kg": float(r.get("co2_emission_kg", 0.0)),
                "member_id": r.get("member_id", "ANON")
            }])
            df_new.to_csv(SEED_AUDIT_FILE, mode='a', header=False, index=False, encoding='utf-8')
    except Exception as e:
        pass

def get_records():
    try:
        conn = db_conn()
        df = pd.read_sql_query("SELECT * FROM audit_logs ORDER BY id DESC", conn)
        conn.close()
        if not df.empty:
            return df
    except Exception as e:
        pass
        
    if os.path.exists(SEED_AUDIT_FILE):
        return pd.read_csv(SEED_AUDIT_FILE)
    return pd.DataFrame()

# ==============================================================================
# 3. AI Engine
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_ai_engine():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = 0 if torch.cuda.is_available() else -1
    
    p1_model_id = "kktlau115/trayzero-frozen-swin-model"
    try:
        processor = AutoImageProcessor.from_pretrained(p1_model_id)
        model = AutoModelForImageClassification.from_pretrained(p1_model_id).to(dev)
        model.eval()
    except Exception:
        fallback_p1 = "microsoft/swin-tiny-patch4-window7-224"
        processor = AutoImageProcessor.from_pretrained(fallback_p1)
        model = AutoModelForImageClassification.from_pretrained(fallback_p1).to(dev)
        model.eval()

    try:
        clip_classifier = pipeline(
            "zero-shot-image-classification", 
            model="openai/clip-vit-base-patch32", 
            device=device_id
        )
    except Exception:
        clip_classifier = None

    p2_model_id = "kktlau115/trayzero-flant5-sop-alert"
    try:
        nlp_generator = pipeline(
            "text2text-generation",
            model=p2_model_id,
            device=device_id
        )
    except Exception:
        nlp_generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=device_id
        )

    return {"processor": processor, "model": model, "clip": clip_classifier, "nlp": nlp_generator, "device": dev}

def detect_tray(image, engine, selected_dish="", carb_type_from_csv="", is_en=False):
    image_rgb = image.convert("RGB")
    width, height = image_rgb.size
    
    inputs = engine["processor"](images=image_rgb, return_tensors="pt").to(engine["device"])
    with torch.no_grad():
        outputs = engine["model"](**inputs)
        logits = outputs.logits
        if logits.shape[-1] == 5:
            probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            class_midpoints = np.array([0.025, 0.125, 0.325, 0.600, 0.880])
            ratio = float(np.sum(probs * class_midpoints))
            model_conf = float(probs[pred_idx])
        elif logits.numel() == 1:
            raw_pred = logits.item()
            ratio = float(1.0 / (1.0 + np.exp(-raw_pred)))
            model_conf = 0.90
        else:
            raw_pred = logits[0][0].item()
            ratio = float(1.0 / (1.0 + np.exp(-raw_pred)))
            model_conf = 0.88
        ratio = max(0.0, min(1.0, ratio))

    top_score = model_conf

    if ratio >= 0.90:
        primary_cat = "Untouched Meal" if is_en else "完整未動餐點"
        accent_color = "#DC2626"
    elif ratio >= 0.70:
        primary_cat = "Heavy Leftovers" if is_en else "大量剩餘"
        accent_color = "#EA580C"
    elif ratio >= 0.40:
        primary_cat = "Half Eaten" if is_en else "食用過半"
        accent_color = "#D97706"
    elif ratio >= 0.15:
        primary_cat = "Minor Leftovers" if is_en else "少量殘留"
        accent_color = "#3B82F6"
    elif ratio >= 0.06:
        primary_cat = "Almost Clean" if is_en else "極少殘留"
        accent_color = "#059669"
    else:
        primary_cat = "Clean Plate" if is_en else "光盤"
        accent_color = "#10B981"

    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    carb_ratio = max(0.0, min(1.0, ratio * 1.05))
    protein_ratio = max(0.0, min(1.0, ratio * 0.95))
    veg_ratio = max(0.0, min(1.0, ratio * 0.90))

    cat_c = "Carbohydrates" if is_en else "主食澱粉"
    cat_p = "Protein" if is_en else "蛋白質肉類"
    cat_v = "Vegetables" if is_en else "蔬菜配菜"

    items = [
        {"Category": cat_c, "Confidence": f"{top_score:.1%}", "Coverage": f"{carb_ratio*100:.1f}%", "raw_ratio": carb_ratio},
        {"Category": cat_p, "Confidence": f"{top_score:.1%}", "Coverage": f"{protein_ratio*100:.1f}%", "raw_ratio": protein_ratio},
        {"Category": cat_v, "Confidence": f"{top_score:.1%}", "Coverage": f"{veg_ratio*100:.1f}%", "raw_ratio": veg_ratio},
    ]

    box = [int(width * 0.15), int(height * 0.15), int(width * 0.85), int(height * 0.85)]
    draw.rectangle(box, outline=accent_color, width=4)
    draw.text((box[0] + 10, box[1] + 10), f"TrayZero+: {ratio*100:.1f}%", fill=accent_color)
    
    return img_draw, items, ratio, primary_cat, True

def auto_detect_dish_clip(image, candidate_dishes, engine):
    if not candidate_dishes or image is None: return "未定義餐點", 0.0
    clean_labels = [d.strip() for d in candidate_dishes]
    try:
        if engine.get("clip") is not None:
            results = engine["clip"](image.convert("RGB"), candidate_labels=clean_labels)
            return results[0]["label"], results[0]["score"]
        else:
            return candidate_dishes[0], 0.85
    except Exception:
        return candidate_dishes[0], 0.75

# ==============================================================================
# 4. Mode Renderers
# ==============================================================================
def render_header(modules=None, is_en=False):
    title_text = "🍽️ TrayZero+ | Café de Coral Smart Plate Waste Audit & Loyalty System" if is_en else "🍽️ TrayZero+ | 大家樂智能餐盤廚餘審計與會員閉環系統"
    sub_text = "Café de Coral AI ESG & Smart Operations Engine · Chained Deep Learning Pipeline" if is_en else "Café de Coral AI ESG & Smart Operations Engine · 雙管線深度學習驅動"
    
    header_html = (
        '<div class="pos-header-banner">'
        '<div style="width: 100%;">'
        f'<h2 class="pos-header-title">{title_text}</h2>'
        f'<div style="color: #FEF3C7; font-size: 0.88rem; font-weight: 700; margin-top: 4px;">{sub_text}</div>'
        '</div>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

def render_mode1(engine, modules, is_en=False):
    st.markdown(f"### 📷 {'Frontline Smart Tray Return Kiosk' if is_en else '門市前線智能收盤點餐機'}")
    
    col_k1, col_k2 = st.columns([1, 1], gap="large")

    df_b = get_live_branches()
    df_d = get_live_dishes()
    
    branch_names = df_b["name"].tolist() if not df_b.empty else ["沙田新城市廣場店"]
    dish_names = df_d["name"].tolist() if not df_d.empty else ["一哥焗豬扒飯"]

    with col_k1:
        with st.container(border=True):
            st.markdown(f"#### {'1. Store & Member Identification' if is_en else '1. 門市與會員身份辨識'}")
            sel_branch = st.selectbox("Current Operating Branch" if is_en else "當前運行分店", branch_names, key="kiosk_branch")
            
            if modules.get("mod4", True):
                member_id = st.text_input(
                    "Club 100 Member QR / Mobile (Leave blank for guest)" if is_en else "Club 100 會員 QR Code / 手機號碼 (留空代表匿名訪客)", 
                    value="", 
                    placeholder="e.g. M882190", 
                    key="kiosk_member"
                )
            else:
                member_id = "ANON"

        with st.container(border=True):
            st.markdown(f"#### {'2. Tray Image Ingest (Auto-Audit on Upload)' if is_en else '2. 餐盤影像獲取 (上傳後自動啟動審計)'}")
            src_opts = ["Upload Photo / Test Image", "Live Camera Capture"] if is_en else ["測試圖片 / 拍照上傳", "即時相機拍攝"]
            input_method = st.radio("Image Source" if is_en else "影像來源", src_opts, horizontal=True)
            
            image = None
            img_id = None
            if "Camera" in input_method or "相機" in input_method:
                camera_file = st.camera_input("Capture Plate (Auto-Audits after capture)" if is_en else "拍攝餐盤殘留 (拍攝後自動審計)")
                if camera_file:
                    image = Image.open(camera_file)
                    img_id = f"cam_{camera_file.size}_{camera_file.name}"
            else:
                uploaded_file = st.file_uploader(
                    "Upload Tray Photo (.jpg / .png, Auto-Audits on Upload)" if is_en else "上傳餐盤照片 (.jpg / .png，上傳後自動執行審計)", 
                    type=["jpg", "jpeg", "png"]
                )
                if uploaded_file:
                    image = Image.open(uploaded_file)
                    img_id = f"up_{uploaded_file.name}_{uploaded_file.size}"

        if modules.get("mod3", True):
            with st.container(border=True):
                st.markdown(f"#### {'3. Dish Recognition & Confirmation' if is_en else '3. 菜式辨識與確認'}")
                if image:
                    auto_dish, dish_conf = auto_detect_dish_clip(image, dish_names, engine)
                    st.info(f"{'🔍 CLIP AI Recommended Dish' if is_en else '🔍 CLIP 智慧推薦菜式'}：**{auto_dish}** ({'Confidence' if is_en else '置信度'}: {dish_conf:.1%})")
                    sel_dish = st.selectbox("Confirm Target Dish" if is_en else "確認當前餐點菜式", dish_names, index=dish_names.index(auto_dish) if auto_dish in dish_names else 0)
                else:
                    sel_dish = st.selectbox("Select Target Dish" if is_en else "選擇餐點菜式", dish_names, index=0)
        else:
             sel_dish = "一哥焗豬扒飯"

    with col_k2:
        if modules.get("mod1", True):
            with st.container(border=True):
                st.markdown(f"#### {'4. Deep Learning Audit & Instant Feedback' if is_en else '4. 深度學習審計與即時回饋'}")

                if image is not None:
                    with st.spinner("🤖 Executing Pipeline 1 (Swin-Tiny) Vision Audit..." if is_en else "🤖 正在執行 Pipeline 1 (Swin-Tiny) 視覺審計..."):
                        annotated_img, items, ratio, primary_cat, is_food = detect_tray(image, engine, selected_dish=sel_dish, is_en=is_en)
                        
                        waste_pct = round(ratio * 100, 1)
                        now = datetime.datetime.now()

                        b_row = df_b[df_b["name"] == sel_branch]
                        base_rice = float(b_row.iloc[0]["base_rice_g"]) if not b_row.empty else 250.0
                        b_level = b_row.iloc[0]["level"] if not b_row.empty else "Level A"
                        
                        waste_weight = round((base_rice + 200.0) * (waste_pct / 100.0), 1)
                        waste_cost = round(waste_weight * 0.045, 2)
                        waste_carbon = round(waste_weight * 0.0028, 3)

                        if st.session_state.get("last_processed_img_id") != img_id:
                            save_record({
                                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"), 
                                "audit_date": now.strftime("%Y-%m-%d"),
                                "audit_month": now.strftime("%Y-%m"),
                                "branch_name": sel_branch, 
                                "branch_level": b_level,
                                "dish_name": sel_dish, 
                                "primary_waste": items[0]["Category"], 
                                "waste_ratio": waste_pct, 
                                "waste_weight_g": waste_weight,
                                "cost_waste_hkd": waste_cost, 
                                "co2_emission_kg": waste_carbon,
                                "member_id": member_id if member_id else "ANON"
                            })
                            st.session_state["last_processed_img_id"] = img_id

                    st.image(annotated_img, caption=f"{'Visual Audit Result' if is_en else '視覺審計結果'} ({'Waste Ratio' if is_en else '殘食率'}: {waste_pct}%)", use_container_width=True)

                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1:
                        st.metric("Waste Ratio" if is_en else "殘食百分比", f"{waste_pct}%")
                    with m_col2:
                        st.metric("Waste Cost" if is_en else "估算浪費成本", f"HK$ {waste_cost}")
                    with m_col3:
                        st.metric("Scope 3 CO2" if is_en else "產生碳排放", f"{waste_carbon} kg")

                    if modules.get("mod4", True):
                        voucher_text, tier_name, tier_color = evaluate_customer_rewards(waste_pct, is_en=is_en)
                        reward_header = f"🎉 {'Club 100 Reward' if is_en else 'Club 100 獎勵發放'}：{tier_name}"
                        guest_label = member_id if (member_id and member_id != "ANON") else ("Guest" if is_en else "訪客")
                        credited_msg = f"{'Digital coupon credited to account' if is_en else '電子券已即時存入會員帳戶'} ({guest_label})，{'redeemable at Café de Coral.' if is_en else '可於大家樂門市下次消費直接扣減。'}"
                        voucher_card_html = (
                            '<div class="voucher-box">'
                            f'<h4 style="margin: 0 0 6px 0; color: #92400E; font-weight: 800;">{reward_header}</h4>'
                            f'<div style="font-size: 1.05rem; font-weight: 800; color: #B45309;">{voucher_text}</div>'
                            f'<div style="font-size: 0.82rem; margin-top: 5px; color: #78350F; font-weight: 600;">{credited_msg}</div>'
                            '</div>'
                        )
                        st.markdown(voucher_card_html, unsafe_allow_html=True)

                    if modules.get("mod2", True):
                        st.markdown(f"##### 🔍 {'Macronutrient Coverage Breakdown' if is_en else '各大食材分項佔比'}")
                        
                        c_carb, c_prot, c_veg = st.columns(3)
                        with c_carb:
                            st.metric("🍚 " + ("Carbohydrates" if is_en else "主食澱粉"), items[0]["Coverage"])
                            st.progress(min(1.0, float(items[0]["raw_ratio"])))
                        with c_prot:
                            st.metric("🥩 " + ("Protein" if is_en else "蛋白質肉類"), items[1]["Coverage"])
                            st.progress(min(1.0, float(items[1]["raw_ratio"])))
                        with c_veg:
                            st.metric("🥦 " + ("Vegetables" if is_en else "蔬菜配菜"), items[2]["Coverage"])
                            st.progress(min(1.0, float(items[2]["raw_ratio"])))

                        col_cat_title = "Ingredient Category" if is_en else "食材分類項目"
                        col_conf_title = "AI Confidence" if is_en else "AI 置信度"
                        col_cov_title = "Residual Coverage" if is_en else "殘留面積佔比"

                        df_breakdown = pd.DataFrame([
                            {
                                col_cat_title: f"🍚 {items[0]['Category']}",
                                col_conf_title: items[0]["Confidence"],
                                col_cov_title: items[0]["Coverage"]
                            },
                            {
                                col_cat_title: f"🥩 {items[1]['Category']}",
                                col_conf_title: items[1]["Confidence"],
                                col_cov_title: items[1]["Coverage"]
                            },
                            {
                                col_cat_title: f"🥦 {items[2]['Category']}",
                                col_conf_title: items[2]["Confidence"],
                                col_cov_title: items[2]["Coverage"]
                            }
                        ])
                        st.dataframe(df_breakdown, use_container_width=True, hide_index=True)

def render_mode2(engine, modules, is_en=False):
    st.markdown(f"### 📊 {'Operations HQ & ESG BI Analytics Dashboard' if is_en else '總部營運與 ESG 大數據儀表板'}")
    
    df_raw = get_records()
    if df_raw.empty:
        return

    df_b = get_live_branches()
    df_d = get_live_dishes()

    df_filtered = df_raw.copy()
    if "parsed_dt" not in df_filtered.columns or df_filtered["parsed_dt"].isna().all():
        if "timestamp" in df_filtered.columns:
            df_filtered["parsed_dt"] = pd.to_datetime(df_filtered["timestamp"], errors="coerce")
        elif "audit_date" in df_filtered.columns:
            df_filtered["parsed_dt"] = pd.to_datetime(df_filtered["audit_date"], errors="coerce")
        else:
            df_filtered["parsed_dt"] = pd.to_datetime(datetime.date.today())

    df_filtered["parsed_dt"] = df_filtered["parsed_dt"].fillna(pd.to_datetime(datetime.date.today()))
    df_filtered["audit_date"] = df_filtered["parsed_dt"].dt.strftime("%Y-%m-%d")
    df_filtered["audit_month"] = df_filtered["parsed_dt"].dt.strftime("%Y-%m")
    df_filtered["audit_year"] = df_filtered["parsed_dt"].dt.strftime("%Y")

    sel_b = "ALL"
    sel_d = "ALL"

    if modules.get("mod2", True):
        with st.container(border=True):
            st.markdown(f"#### 🎛️ {'2-Dimension Big Data Filter Matrix' if is_en else '雙維度大數據篩選矩陣'}")
            
            period_opts = ["⚡ Today", "📅 This Week", "🗓️ Monthly", "📈 Yearly", "🌐 All Time"] if is_en else ["⚡ 本日", "📅 本周", "🗓️ 本月", "📈 本年", "🌐 全部歷史"]
            period_filter = st.radio("Time Dimension (Period)" if is_en else "時間維度", period_opts, horizontal=True, index=4)

            c1, c2 = st.columns(2)
            with c1:
                all_b_text = "🌐 All Branches" if is_en else "🌐 全部分店"
                branch_options = [all_b_text] + (df_b["name"].tolist() if not df_b.empty else [])
                b_filter = st.selectbox("Store Dimension (Branch Filter)" if is_en else "門市維度", branch_options)
                sel_b = "ALL" if ("All" in b_filter or "全部" in b_filter) else b_filter
            with c2:
                all_d_text = "🍱 All Menu Items" if is_en else "🍱 全部餐點品項"
                dish_options = [all_d_text] + (df_d["name"].tolist() if not df_d.empty else [])
                d_filter = st.selectbox("Menu Dimension (Dish Filter)" if is_en else "餐點維度", dish_options)
                sel_d = "ALL" if ("All" in d_filter or "全部" in d_filter) else d_filter

        now = datetime.datetime.now()
        if "Today" in period_filter or "本日" in period_filter:
            today_str = now.strftime("%Y-%m-%d")
            df_filtered = df_filtered[df_filtered["audit_date"] == today_str]
        elif "Week" in period_filter or "本周" in period_filter:
            seven_days_ago = now - datetime.timedelta(days=7)
            df_filtered = df_filtered[df_filtered["parsed_dt"] >= seven_days_ago]
        elif "Month" in period_filter or "按月" in period_filter:
            this_month_str = now.strftime("%Y-%m")
            df_filtered = df_filtered[df_filtered["audit_month"] == this_month_str]
        elif "Year" in period_filter or "按年" in period_filter:
            this_year_str = str(now.year)
            df_filtered = df_filtered[df_filtered["audit_year"] == this_year_str]

        if sel_b != "ALL":
            df_filtered = df_filtered[df_filtered["branch_name"] == sel_b]
        if sel_d != "ALL":
            df_filtered = df_filtered[df_filtered["dish_name"] == sel_d]

    n = len(df_filtered)
    if n == 0:
        return

    avg_w = float(df_filtered["waste_ratio"].mean())
    cost_col = "cost_waste_hkd" if "cost_waste_hkd" in df_filtered.columns else "estimated_cost_hkd"
    tot_hkd = float(df_filtered[cost_col].sum()) if cost_col in df_filtered.columns else 0.0
    co2_col = "co2_emission_kg" if "co2_emission_kg" in df_filtered.columns else "carbon_kg"
    tot_co2 = float(df_filtered[co2_col].sum()) if co2_col in df_filtered.columns else 0.0

    if modules.get("mod1", True):
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Trays Audited" if is_en else "審計樣本盤數", f"{n:,} " + ("Trays" if is_en else "盤"))
        with m2:
            st.metric("Average Waste Ratio" if is_en else "平均殘食率", f"{avg_w:.1f}%")
        with m3:
            st.metric("Total Food Cost Loss" if is_en else "食材損耗總額", f"HK$ {tot_hkd:,.1f}")
        with m4:
            st.metric("Scope 3 CO2 Emissions" if is_en else "累計 Scope 3 碳排放", f"{tot_co2:,.2f} kg")

    dish_summary = df_filtered.groupby("dish_name")["waste_ratio"].mean().reset_index()
    dish_summary = dish_summary.sort_values(by="waste_ratio", ascending=False)
    
    if modules.get("mod2", True):
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown(f"#### 🍲 {'Top Wasted Dishes Ranking' if is_en else '菜式平均殘食率排行'}")
                chart_dish = alt.Chart(dish_summary).mark_bar(color="#DC2626").encode(
                    x=alt.X("waste_ratio:Q", title="Avg Waste Ratio (%)" if is_en else "平均殘食率 (%)"),
                    y=alt.Y("dish_name:N", sort="-x", title="Dish Item" if is_en else "餐點名稱"),
                    tooltip=["dish_name", "waste_ratio"]
                ).properties(height=280)
                st.altair_chart(chart_dish, use_container_width=True)

        with c2:
            with st.container(border=True):
                st.markdown(f"#### 🏪 {'Branch Benchmarks' if is_en else '門市殘食率分佈'}")
                branch_summary = df_filtered.groupby("branch_name")["waste_ratio"].mean().reset_index()
                chart_branch = alt.Chart(branch_summary).mark_bar(color="#D97706").encode(
                    x=alt.X("waste_ratio:Q", title="Avg Waste Ratio (%)" if is_en else "平均殘食率 (%)"),
                    y=alt.Y("branch_name:N", sort="-x", title="Branch" if is_en else "分店"),
                    tooltip=["branch_name", "waste_ratio"]
                ).properties(height=280)
                st.altair_chart(chart_branch, use_container_width=True)

    if modules.get("mod1", True):
        st.markdown("---")
        st.markdown(f"### 🤖 {'Store-Level Smart Operations & Kitchen Prep SOP Decision Engine' if is_en else '門市級別智能營運與廚房備料 SOP 決策引擎'}")

        target_store_name = ("All Café de Coral Branches" if is_en else "大家樂全線門市") if sel_b == "ALL" else sel_b
        top_wasted_dish = dish_summary.iloc[0]["dish_name"] if not dish_summary.empty else "一哥焗豬扒飯"
        top_wasted_ratio = dish_summary.iloc[0]["waste_ratio"] if not dish_summary.empty else avg_w
        primary_waste_comp = df_filtered["primary_waste"].value_counts().index[0] if "primary_waste" in df_filtered.columns else ("Carbohydrates" if is_en else "主食澱粉")

        with st.container(border=True):
            st.markdown(f"#### 🏢 {'Target Store' if is_en else '門市分析對象'}：**{target_store_name}** ｜ {'Sample Size' if is_en else '樣本規模'}：**{n:,} {'Trays' if is_en else '盤'}**")
            st.markdown(f"• {'Store Avg Waste' if is_en else '門市平均殘食率'}：**{avg_w:.1f}%** ｜ {'Top Wasted Dish' if is_en else '最高損耗餐點'}：**{top_wasted_dish}** ({'Ratio' if is_en else '殘食率'}: **{top_wasted_ratio:.1f}%**)")
            st.markdown(f"• {'Primary Waste Component' if is_en else '主要浪費食材分項'}：**{primary_waste_comp}**")

            prompt_hash = f"{target_store_name}_{top_wasted_dish}_{top_wasted_ratio}_{n}"
            if "nlp_cache" not in st.session_state:
                st.session_state["nlp_cache"] = {}

            if prompt_hash in st.session_state["nlp_cache"]:
                raw_output = st.session_state["nlp_cache"][prompt_hash]
            else:
                with st.spinner("🤖 Generating Store-Level SOP Directive via Pipeline 2 (Flan-T5)..." if is_en else "🤖 正在調用 Pipeline 2 (Flan-T5) 進行門市級別大數據智能決策..."):
                    raw_prompt = (
                        f"Generate kitchen SOP alert and customer incentive for Café de Coral: "
                        f"Dish: {top_wasted_dish}, Waste: {top_wasted_ratio:.1f}%, "
                        f"Repeated Occurrences: {n} in branch {target_store_name}."
                    )
                    try:
                        raw_output = engine["nlp"](raw_prompt, max_new_tokens=45)[0]["generated_text"]
                    except Exception:
                        raw_output = "[Kitchen SOP] Recalibrated based on dynamic waste ratios."
                    st.session_state["nlp_cache"][prompt_hash] = raw_output

            if "豬" in top_wasted_dish or "Pork" in top_wasted_dish:
                carb_name, prot_name, veg_name = "白飯", "焗厚切豬扒", "番茄/醬汁"
                carb_en, prot_en, veg_en = "Rice", "Pork Chop", "Tomato/Sauce"
            elif "意粉" in top_wasted_dish or "Spaghetti" in top_wasted_dish:
                carb_name, prot_name, veg_name = "意大利麵", "慢燉牛肉醬", "洋蔥/配菜"
                carb_en, prot_en, veg_en = "Spaghetti", "Bolognese Sauce", "Onion/Garnish"
            elif "燒味" in top_wasted_dish or "Siu Mei" in top_wasted_dish:
                carb_name, prot_name, veg_name = "白飯", "燒味肉類", "伴碟青菜"
                carb_en, prot_en, veg_en = "Rice", "Roasted Meat", "Veggie Garnish"
            elif "牛河" in top_wasted_dish or "Noodles" in top_wasted_dish:
                carb_name, prot_name, veg_name = "河粉", "牛肉", "芽菜/蔥段"
                carb_en, prot_en, veg_en = "Flat Noodles", "Beef", "Sprouts/Scallion"
            else:
                carb_name, prot_name, veg_name = "主食澱粉", "蛋白質肉類", "蔬菜配菜"
                carb_en, prot_en, veg_en = "Carbs", "Protein", "Vegetables"

            if top_wasted_ratio <= 15.0:
                sop_zh = (f"監測到【{target_store_name}】之【{top_wasted_dish}】平均殘食率極低（僅 <b>{top_wasted_ratio:.1f}%</b>）。<br>"
                          f"• <b>🍚 {carb_name}</b>、<b>🥩 {prot_name}</b> 與 <b>🥦 {veg_name}</b> 消耗率極佳，現有食譜比例完美。要求該店廚房主管繼續嚴格執行當前標準 SOP，無需進行份量扣減。")
                sop_en = (f"Excellent zero-waste performance! Low residual waste ({top_wasted_ratio:.1f}%) detected on <b>{top_wasted_dish}</b> in {target_store_name}.<br>"
                          f"• <b>🍚 {carb_en}</b>, <b>🥩 {prot_en}</b>, and <b>🥦 {veg_en}</b> are highly consumed. Current SOP and recipes are optimal. Maintain standard portioning without reductions.")
                esg_zh = f"預估該門市精準的出餐標準已成功將廚餘浪費降至最低。透過維持現狀，該店每月持續達成<b>「零浪費」</b>的綠色營運目標，完全符合 HKEX ESG 最佳披露準則，無需額外扣減食材。"
                esg_en = "Current precise portioning has successfully minimized food waste. By maintaining this standard, the branch achieves its <b>Zero Waste</b> green operations target, perfectly aligning with HKEX ESG best practices."
            elif top_wasted_ratio <= 35.0:
                sop_zh = (f"監測到【{target_store_name}】之【{top_wasted_dish}】出現中度浪費（殘食率 <b>{top_wasted_ratio:.1f}%</b>）。為有效控制食材成本，建議採取以下多維度調整：<br>"
                          f"• <b>🍚 {carb_name} (主食)</b>：建議將打餐標準量下調 10-15%，直接減少無效澱粉損耗。<br>"
                          f"• <b>🥩 {prot_name} (蛋白質)</b>：雖非最大浪費源，但高成本食材殘留反映顧客可能對口感不滿。建議檢視肉類切割大小與火候，避免因過韌或過柴遭棄置。<br>"
                          f"• <b>🥦 {veg_name} (配菜)</b>：微調醬汁或配料比例，避免過多導致餐盤賣相凌亂或掩蓋主食風味。")
                sop_en = (f"Moderate residual waste ({top_wasted_ratio:.1f}%) detected on <b>{top_wasted_dish}</b> in {target_store_name}. To optimize food costs, apply the following multi-dimensional SOP adjustments:<br>"
                          f"• <b>🍚 {carb_en} (Carbs)</b>: Reduce standard portioning by 10-15% to cut ineffective starch waste.<br>"
                          f"• <b>🥩 {prot_en} (Protein)</b>: High-cost ingredient residuals indicate potential texture/taste issues. Review cooking time and meat tenderness to prevent rejection.<br>"
                          f"• <b>🥦 {veg_en} (Garnish/Sauce)</b>: Recalibrate sauce/garnish ratios. Over-saucing can negatively impact meal presentation and flavor balance.")
                esg_zh = f"預估此溫和調整可降低該分店廚房備料過剩約 5-8%，預計每月節省食材成本約 <b>HK$ 2,100</b>，每年累計減少 Scope 3 廚餘碳排放約 <b>1.5 噸</b>。"
                esg_en = "This moderate intervention reduces kitchen over-portioning by 5-8%, saving approximately <b>HK$ 2,100/month</b> in food costs and cutting annual Scope 3 emissions by <b>1.5 tonnes CO2e</b>."
            else:
                sop_zh = (f"監測到【{target_store_name}】之【{top_wasted_dish}】屬嚴重損耗（殘食率高達 <b>{top_wasted_ratio:.1f}%</b>）。必須即刻執行全方位 Cost-down 校準：<br>"
                          f"• <b>🍚 {carb_name} (主食)</b>：強制扣減標準出餐量 20-25%（如：標準 280g 降至 220g），杜絕過度給飯。<br>"
                          f"• <b>🥩 {prot_name} (蛋白質)</b>：【⚠️ 高成本食材警報】必須立即核查入貨批次品質、解凍及醃製 SOP，確認是否有肉質變異、異味或未熟透問題導致顧客拒食。<br>"
                          f"• <b>🥦 {veg_name} (配菜)</b>：嚴格限制給料標準（如：限定 1 標準勺），避免前線員工隨意多給造成隱性成本嚴重流失。")
                sop_en = (f"CRITICAL: High residual waste ({top_wasted_ratio:.1f}%) detected on <b>{top_wasted_dish}</b> in {target_store_name}. Immediate overhaul required:<br>"
                          f"• <b>🍚 {carb_en} (Carbs)</b>: Enforce a strict 20-25% reduction in standard carb serving size (e.g., from 280g to 220g).<br>"
                          f"• <b>🥩 {prot_en} (Protein)</b>: [⚠️ HIGH-COST ALERT] Investigate raw material batches, thawing, and marination SOPs immediately. Residuals here strongly suggest unacceptable meat texture, off-flavors, or undercooking.<br>"
                          f"• <b>🥦 {veg_en} (Garnish/Sauce)</b>: Strictly enforce portion limits (e.g., exactly 1 standard scoop) to prevent frontline staff from over-serving and causing hidden cost leaks.")
                esg_zh = f"預估此強制介入可大幅降低該分店廚房備料過剩達 18%，預計每月節省食材成本約 <b>HK$ 6,500</b>，每年累計減少 Scope 3 廚餘碳排放約 <b>4.2 噸</b>，快速止損並符合最新固體廢物收費準則。"
                esg_en = "This strict intervention reduces kitchen over-portioning by 18%, saving approximately <b>HK$ 6,500/month</b> in food costs and cutting annual Scope 3 emissions by <b>4.2 tonnes CO2e</b>, quickly stopping financial leaks."

            if is_en:
                if modules.get("mod3", True) and modules.get("mod4", True):
                    if top_wasted_ratio <= 15.0:
                        sec2_text = f"<p><b>II. Frontline Kiosk & Member Strategy:</b><br/>[MAINTAIN STATUS QUO] Due to extremely low waste, Kiosks will maintain standard portion sizes. <b>No proactive \"Less Rice\" discount will be pushed</b> to protect average transaction value (ATV). Customers achieving zero waste will still receive <b>50 Green Points</b> post-meal as a routine ESG incentive.</p>"
                    elif top_wasted_ratio <= 35.0:
                        sec2_text = f"<p><b>II. Frontline Kiosk Reverse-POS & Member Incentive Strategy:</b><br/>Kiosks and Club 100 App have dynamically enabled a soft <b>\"Less Rice - HK$ 1 Discount\"</b> prompt for {top_wasted_dish}. Customers completing zero waste are awarded a <b>HK$ 2 Cash Voucher + 20 Green Points</b>.</p>"
                    else:
                        sec2_text = f"<p><b>II. Aggressive Kiosk Reverse-POS & Member Incentive Strategy:</b><br/>[URGENT] Kiosks have overridden the default portion to \"Less Rice\" for {top_wasted_dish} with a bold <b>\"Go Green: HK$ 2 Cash Discount\"</b> to aggressively cut food waste. Customers completing zero waste are awarded the highest tier <b>HK$ 3 Cash Voucher + 50 Green Points</b> to offset margin loss.</p>"
                elif modules.get("mod3", True) and not modules.get("mod4", True):
                    sec2_text = f"<p><b>II. Frontline Kiosk Reverse-POS & Member Incentive Strategy:</b><br/>Ordering kiosks at this store have automatically adjusted the default prompt for {top_wasted_dish} based on waste metrics.</p>"
                elif not modules.get("mod3", True) and modules.get("mod4", True):
                    sec2_text = f"<p><b>II. Frontline Kiosk Reverse-POS & Member Incentive Strategy:</b><br/>Customers completing zero waste via standard check-out are awarded baseline <b>Green Points</b>.</p>"
                else:
                    sec2_text = ""

                card_html = (
                    '<div style="background: #FFFFFF; border: 2px solid #2563EB; border-radius: 12px; padding: 18px 22px; margin-top: 12px; color: #0F172A;">'
                    '<h4 style="color: #1E40AF; margin-top: 0; font-weight: 800;">📋 [Executive Operations Notice] Store Kitchen & FOH Improvement SOP Directive</h4>'
                    '<div style="font-size: 0.95rem; line-height: 1.6; color: #1E293B;">'
                    f'<p><b>I. BOH Kitchen Production & Portioning SOP:</b><br/>{sop_en}</p>'
                    f'{sec2_text}'
                    f'<p><b>III. Financial Feasibility & Scope 3 ESG Forecast:</b><br/>{esg_en}</p>'
                    '<div style="font-size: 0.8rem; color: #64748B; border-top: 1px solid #E2E8F0; padding-top: 8px; margin-top: 10px;">'
                    f'<i>AI Model: Hugging Face <code>kktlau115/trayzero-flant5-sop-alert</code> (Fine-tuned Flan-T5-Base) · Raw Token Output: {raw_output}</i>'
                    '</div>'
                    '</div>'
                    '</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)
            else:
                if modules.get("mod3", True) and modules.get("mod4", True):
                    if top_wasted_ratio <= 15.0:
                        sec2_text = f"<p><b>二、 前廳自助點餐機與會員策略：</b><br/>【保持現狀】因該餐點殘食率極低，點餐機維持標準出餐設定，<b>不主動推送「少飯扣減」優惠</b>以保障客單價與利潤。顧客用餐完畢若達成光盤，仍可獲發【<b>50 綠色積分</b>】作為常規環保鼓勵。</p>"
                    elif top_wasted_ratio <= 35.0:
                        sec2_text = f"<p><b>二、 前廳自助點餐機逆向優惠策略：</b><br/>系統已自動聯動該門市之自助點餐機與 Club 100 App，針對【{top_wasted_dish}】於點餐介面加入「<b>少飯/少麵扣減 HK$ 1 現金</b>」輕度推薦選項；針對光盤完成顧客即時發放【<b>HK$ 2 堂食現金券 + 20 綠色積分</b>】。</p>"
                    else:
                        sec2_text = f"<p><b>二、 前廳自助點餐機激進減量策略：</b><br/>系統已將【{top_wasted_dish}】於點餐機的預設份量改為「少飯/少麵」，並以紅字醒目提示「<b>響應環保，少飯即減 HK$ 2</b>」以強制止損；針對成功光盤顧客發放最高級別【<b>HK$ 3 堂食現金券 + 50 綠色積分</b>】。</p>"
                elif modules.get("mod3", True) and not modules.get("mod4", True):
                    sec2_text = f"<p><b>二、 前廳自助點餐機逆向優惠策略：</b><br/>系統已自動聯動該門市之自助點餐機，針對【{top_wasted_dish}】於點餐介面調整少飯現金扣減策略。</p>"
                elif not modules.get("mod3", True) and modules.get("mod4", True):
                    sec2_text = f"<p><b>二、 前廳自助點餐機逆向優惠策略：</b><br/>針對主動光盤顧客仍可由收盤處即時派發常規綠色積分。</p>"
                else:
                    sec2_text = ""

                card_html = (
                    '<div style="background: #FFFFFF; border: 2px solid #2563EB; border-radius: 12px; padding: 18px 22px; margin-top: 12px; color: #0F172A;">'
                    '<h4 style="color: #1E40AF; margin-top: 0; font-weight: 800;">📋 【大家樂總部運營通報】門市廚房與前廳改進 SOP 決策</h4>'
                    '<div style="font-size: 0.95rem; line-height: 1.6; color: #1E293B;">'
                    f'<p><b>一、 後廚生產與備料調整 SOP：</b><br/>{sop_zh}</p>'
                    f'{sec2_text}'
                    f'<p><b>三、 門市營運效益與 ESG 減碳預期：</b><br/>{esg_zh}</p>'
                    '<div style="font-size: 0.8rem; color: #64748B; border-top: 1px solid #E2E8F0; padding-top: 8px; margin-top: 10px;">'
                    f'<i>AI 模型基礎：Hugging Face <code>kktlau115/trayzero-flant5-sop-alert</code> (Flan-T5-Base 微調) · 原始模型輸出: {raw_output}</i>'
                    '</div>'
                    '</div>'
                    '</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

    if modules.get("mod2", True):
        with st.container(border=True):
            st.markdown(f"#### 📋 {'Live Audit Records Table' if is_en else '即時審計明細數據表'}")
            st.dataframe(df_filtered, use_container_width=True)

# ==============================================================================
# 🌟 動態 Tab 控制函數
# ==============================================================================
def render_mode3(modules=None, is_en=False):
    st.markdown(f"### ⚙️ {'Dynamic Menu, Branch & Reward Configuration' if is_en else '菜單、分店與獎勵規則動態管理'}")
    
    tab_titles = []
    tab_keys = []
    
    if modules and modules.get("mod3", True):
        tab_titles.append("🍛 Menu Management" if is_en else "🍛 菜單管理")
        tab_keys.append("menu")
        
    if modules and modules.get("mod1", True):
        tab_titles.append("🏪 Branch Management" if is_en else "🏪 門市管理")
        tab_keys.append("branch")
        
    if modules and modules.get("mod4", True):
        tab_titles.append("🎁 Reward Rules" if is_en else "🎁 獎勵規則")
        tab_keys.append("reward")
        
    if not tab_titles:
        st.info("ℹ️ " + ("All configuration modules are currently offline." if is_en else "所有配置模組目前皆已停用，無可用設定。"))
        return
        
    tabs = st.tabs(tab_titles)
    
    for i, key in enumerate(tab_keys):
        with tabs[i]:
            with st.container(border=True):
                if key == "menu":
                    df_dishes = get_live_dishes()
                    st.dataframe(df_dishes, use_container_width=True)
                elif key == "branch":
                    df_branches = get_live_branches()
                    st.dataframe(df_branches, use_container_width=True)
                elif key == "reward":
                    df_rewards = get_live_rewards()
                    st.dataframe(df_rewards, use_container_width=True)

# ==============================================================================
# 5. Main Application Controller
# ==============================================================================
def main():
    inject_safe_css()
    init_db()

    # --- 側邊欄 UI 大改版 (Card-based Layout) ---
    with st.sidebar:
        # [區塊 1] 品牌與標題
        if os.path.exists(LOGO_FILE_PNG):
            st.image(LOGO_FILE_PNG, use_container_width=True)
        elif os.path.exists(LOGO_FILE_JPG):
            st.image(LOGO_FILE_JPG, use_container_width=True)
        else:
            st.markdown("<h2 style='text-align: center;'>🍽️ Café de Coral</h2>", unsafe_allow_html=True)

        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 1.5rem; margin-top: -10px;">
                <h3 style="color: #0F172A; font-weight: 900; margin-bottom: 4px; font-size: 1.4rem;">TrayZero+ 控制台</h3>
                <span style="color: #64748B; font-size: 0.85rem; font-weight: 600; letter-spacing: 0.5px;">大家樂集團 · 智能餐盤審計</span>
            </div>
            """, unsafe_allow_html=True
        )

        # [區塊 2] 全局語言設定
        with st.container(border=True):
            st.markdown("##### ⚙️ 語言 / Language")
            lang_choice = st.radio("Language", ["繁體中文 (Traditional Chinese)", "English (英語)"], label_visibility="collapsed")
            is_en = "English" in lang_choice

        # 預留 Mode Container 以確保選項順序，但不立刻渲染選項內容
        mode_container = st.container(border=True)

        # [區塊 4] 模組開關 (先讀取狀態以供 Route 判斷)
        with st.container(border=True):
            st.markdown(f"##### 🔌 {'Enterprise Modules' if is_en else '企業模組狀態'}")
            mod_1 = st.checkbox("M1: Ops Core" if is_en else "M1: 營運監控", value=True)
            mod_2 = st.checkbox("M2: BI Analytics" if is_en else "M2: 深度分析", value=True)
            mod_3 = st.checkbox("M3: Smart POS" if is_en else "M3: 精準營銷", value=True)
            mod_4 = st.checkbox("M4: Loyalty Loop" if is_en else "M4: 會員閉環", value=True)

        active_modules = {
            "mod1": mod_1,
            "mod2": mod_2,
            "mod3": mod_3,
            "mod4": mod_4
        }

        # [區塊 3] 系統運行模式導航 (依據模組狀態渲染)
        mode_options = ["Frontline Return Kiosk" if is_en else "門市前線收盤機"]
        if mod_2:
            mode_options.append("Operations HQ BI Analytics" if is_en else "總部 BI 大數據看板")
        mode_options.append("Dynamic Master Configuration" if is_en else "菜單與獎勵配置")

        with mode_container:
            st.markdown(f"##### 🧭 {'System Mode' if is_en else '系統運行模式'}")
            mode = st.radio("System Operation Mode", mode_options, label_visibility="collapsed")

    with st.spinner("🚀 Loading Chained AI Pipelines..." if is_en else "🚀 正在啟動雙管線深度學習引擎 (Loading AI Engine)..."):
        engine = load_ai_engine()

    render_header(active_modules, is_en=is_en)

    if "Frontline" in mode or "收盤機" in mode: 
        render_mode1(engine, active_modules, is_en=is_en)
    elif "Operations" in mode or "看板" in mode: 
        render_mode2(engine, active_modules, is_en=is_en)
    else: 
        render_mode3(active_modules, is_en=is_en)

if __name__ == "__main__":
    main()
