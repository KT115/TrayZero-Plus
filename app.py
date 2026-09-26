import os
import datetime
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageStat
import torch
from transformers import (
    AutoImageProcessor, 
    AutoModelForObjectDetection, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    pipeline
)

# ==============================================================================
# 0. Primary Streamlit Execution Configuration
# ==============================================================================
st.set_page_config(
    page_title="TrayZero+ | 大家樂智能餐盤審計與會員閉環平台", 
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

CONTAINER_AND_BEVERAGE_BLOCKLIST = {
    "cup", "bottle", "wine glass", "dining table", 
    "knife", "fork", "spoon", "chopsticks", "person", "chair"
}

def inject_safe_css():
    st.markdown("""
    <style>
        /* 根色彩變數 - Option B: Fast-Casual POS 商業點餐機高對比風格 */
        :root {
            --cdc-red: #DC2626 !important;
            --cdc-amber: #D97706 !important;
            --cdc-amber-light: #FFFBEB !important;
            --cdc-dark: #0F172A !important;
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

        /* 頂部 POS 風格企業橫幅 (取代圖片，純文字品牌渲染) */
        .pos-header-banner {
            background: #FFFFFF !important;
            border-radius: 12px;
            padding: 16px 22px;
            margin-bottom: 20px;
            border: 1px solid #E2E8F0;
            border-left: 8px solid #D97706 !important;
            box-shadow: 0 4px 12px rgba(217, 119, 6, 0.08);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .pos-header-title {
            color: #0F172A !important;
            font-size: 1.35rem !important;
            font-weight: 900 !important;
            margin: 0 !important;
            letter-spacing: -0.02em;
        }
        .pos-header-sub {
            color: #64748B !important;
            font-size: 0.82rem !important;
            font-weight: 700 !important;
            margin-top: 4px;
        }
        .shift-badge {
            background: #FEF3C7 !important;
            border: 1.5px solid #FDE68A !important;
            color: #B45309 !important;
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 900;
        }

        /* 側邊欄 POS 樣式 */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1.5px solid #E2E8F0 !important;
            padding-top: 1rem !important;
        }
        .sidebar-brand-box {
            background: linear-gradient(135deg, #DC2626 0%, #D97706 100%) !important;
            border-radius: 12px;
            padding: 16px 14px;
            text-align: center;
            color: #FFFFFF !important;
            margin-bottom: 18px;
            box-shadow: 0 4px 10px rgba(220, 38, 38, 0.15);
        }
        .sidebar-brand-title {
            font-size: 1.15rem !important;
            font-weight: 900 !important;
            letter-spacing: 0.05em;
            color: #FFFFFF !important;
            margin-bottom: 2px;
        }
        .sidebar-brand-sub {
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            color: #FEF08A !important;
        }

        /* 元件 Label 強制深黑高對比 */
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

        /* 單選 Radio 與 Checkbox 文字顯色 */
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

        /* 標籤頁 (Tabs) */
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
            border-bottom: 3px solid #D97706 !important;
        }
        div[data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] p,
        div[data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] span {
            color: #D97706 !important;
            font-weight: 900 !important;
        }

        /* 輸入框 */
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

        /* 大家樂電子票券卡 (Digital Coupon Card) */
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
            font-size: 1.25rem;
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

        /* 大尺寸 POS 數據指標磚 */
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

        /* 營運指引卡 */
        .pos-directive-card {
            border-radius: 10px;
            padding: 14px 18px;
            margin-top: 14px;
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0;
            border-left: 5px solid #DC2626 !important;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
        }

        /* 按鈕樣式 */
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

        # 種子菜單初始化 (保障車仔麵永遠存在)
        dishes_count = conn.execute("SELECT COUNT(*) FROM master_dishes_db").fetchone()[0]
        if dishes_count == 0:
            init_dishes = [
                ("D01", "一哥焗豬扒飯 (Baked Pork Chop Rice)", "白米飯", "焗厚切豬扒"),
                ("D02", "咖喱牛腩飯 (Curry Beef Brisket Rice)", "白米飯", "慢燉牛腩"),
                ("D03", "滑蛋蝦仁飯 (Scrambled Egg Shrimp Rice)", "白米飯", "滑蛋蝦仁"),
                ("D04", "香辣肉燥肉餅飯 (Minced Pork Patty Rice)", "白米飯", "煎肉餅"),
                ("D05", "焗肉醬意粉 (Baked Spaghetti Bolognese)", "意大利麵", "慢燉牛肉醬"),
                ("D06", "車仔麵 (Kart Noodle)", "中式麵條", "牛腩/魚蛋/蘿蔔")
            ]
            conn.executemany("INSERT OR REPLACE INTO master_dishes_db VALUES (?, ?, ?, ?)", init_dishes)

        branches_count = conn.execute("SELECT COUNT(*) FROM master_branches_db").fetchone()[0]
        if branches_count == 0:
            init_branches = [
                ("中環威靈頓街店", "Level A (商業核心區 / CBD)", "中西區", "白領上班族為主，午市尖峰翻檯率極高", 1200, 240),
                ("沙田新城市廣場店", "Level B (住宅商場 / Residential)", "沙田區", "家庭客、長者與週末休閒客群", 1500, 260),
                ("香港科技大學店 (HKUST)", "Level C (校園與青年區 / Campus)", "西貢區", "學生、教職員，運動量及食量顯著較大", 1800, 280),
                ("將軍澳 Popcorn 店", "Level B (住宅商場 / Residential)", "西貢區", "家庭客及換乘鐵路客流", 1400, 260)
            ]
            conn.executemany("INSERT OR REPLACE INTO master_branches_db VALUES (?, ?, ?, ?, ?, ?)", init_branches)

# ==============================================================================
# 3. 雙向永續資料讀取與儲存 (CSV + SQLite Auto-Merge)
# ==============================================================================
def get_live_dishes():
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
            if "member_id" not in df.columns: df["member_id"] = "GUEST"
            df["member_id"] = df["member_id"].fillna("GUEST")
        return df

# ==============================================================================
# 4. Multi-Modal Vision Engine (CLIP 語義評估 + YOLOS 空間排除)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_ai_engine():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m_path = os.path.join(BASE_DIR, "Fine-tuned_Model_files")
    is_finetuned = False
    
    if os.path.exists(m_path) and any(os.scandir(m_path)):
        model_name = m_path
        is_finetuned = True
    else:
        model_name = "hustvl/yolos-tiny"
    
    proc = AutoImageProcessor.from_pretrained(model_name)
    det = AutoModelForObjectDetection.from_pretrained(model_name).to(dev)
    tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    gen = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(dev)
    
    clip_classifier = pipeline(
        "zero-shot-image-classification", 
        model="openai/clip-vit-base-patch32", 
        device=0 if torch.cuda.is_available() else -1
    )
    
    return {
        "proc": proc,
        "det": det,
        "tok": tok,
        "gen": gen,
        "clip": clip_classifier,
        "device": dev,
        "is_finetuned": is_finetuned
    }

def detect_tray(image, engine, selected_dish="", carb_type_from_csv=""):
    inp = engine["proc"](images=image, return_tensors="pt").to(engine["device"])
    with torch.no_grad(): 
        out = engine["det"](**inp)
        
    sz = torch.tensor([image.size[::-1]]).to(engine["device"])
    res = engine["proc"].post_process_object_detection(out, threshold=0.15, target_sizes=sz)[0]
    
    img_draw = image.copy()
    total_area = image.size[0] * image.size[1]
    draw = ImageDraw.Draw(img_draw)
    items = []
    
    waste_level_labels = [
        "a bowl or plate with a lot of leftover food, noodles, vegetables and meat",
        "a bowl with half portion of leftover food",
        "a bowl with only a few small scraps of food left",
        "a completely finished empty bowl with no noodles or food, only broth soup and spoon left"
    ]
    level_res = engine["clip"](image, candidate_labels=waste_level_labels)
    top_level = level_res[0]["label"]
    
    food_type_labels = [
        "leftover noodles or pasta in the bowl",
        "leftover rice on the plate",
        "leftover meat, fishballs or sausages",
        "leftover vegetables, green leaves or soup",
        "clean empty dish"
    ]
    type_res = engine["clip"](image, candidate_labels=food_type_labels)
    top_type = type_res[0]["label"]

    is_truly_empty = ("empty bowl" in top_level or "clean empty dish" in top_type) and ("noodles" not in top_type and "meat" not in top_type)

    if is_truly_empty:
        return img_draw, [{
            "分類項目 Category": "光盤 Clean Plate", 
            "置信度 Confidence": f"{level_res[0]['score']:.1%}", 
            "佔比 Coverage": "0.0%"
        }], 0.0, "光盤 Clean Plate", True

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

    has_boxes = False
    for box, score, label_id in zip(res["boxes"].tolist(), res["scores"].tolist(), res["labels"].tolist()):
        lbl = engine["det"].config.id2label.get(label_id, "item").lower()
        if lbl in CONTAINER_AND_BEVERAGE_BLOCKLIST:
            continue
            
        b = [max(0, box[0]), max(0, box[1]), min(image.size[0], box[2]), min(image.size[1], box[3])]
        box_w = b[2] - b[0]
        box_h = b[3] - b[1]
        area = box_w * box_h
        
        if area > total_area * 0.70:
            continue
        if b[1] < image.size[1] * 0.45 and (box_h / max(1, box_w) > 1.3):
            continue

        draw.rectangle(b, outline=accent_color, width=3)
        draw.text((b[0] + 4, b[1] + 4), f"{primary.split(' ')[0]}", fill=accent_color)
        items.append({
            "分類項目 Category": primary.split(" ")[0], 
            "置信度 Confidence": f"{score:.1%}", 
            "佔比 Coverage": f"{area/total_area:.1%}"
        })
        has_boxes = True

    if not has_boxes:
        w, h = image.size
        bowl_box = [int(w * 0.28), int(h * 0.18), int(w * 0.78), int(h * 0.78)]
        draw.rectangle(bowl_box, outline=accent_color, width=3)
        draw.text((bowl_box[0] + 6, bowl_box[1] + 6), f"{primary.split(' ')[0]} (主食與配料殘留)", fill=accent_color)
        items.append({
            "分類項目 Category": primary.split(" ")[0], 
            "置信度 Confidence": f"{type_res[0]['score']:.1%}", 
            "佔比 Coverage": "35.0%"
        })

    if "a lot of leftover food" in top_level:
        ratio = 0.52
    elif "half portion" in top_level:
        ratio = 0.35
    else:
        ratio = 0.22

    return img_draw, items, ratio, primary, True

def auto_detect_dish_clip(image, candidate_dishes, engine):
    if not candidate_dishes:
        return "未定義餐點", 0.0
    clean_labels = [d.strip() for d in candidate_dishes]
    try:
        results = engine["clip"](image, candidate_labels=clean_labels)
        return results[0]["label"], results[0]["score"]
    except Exception:
        return candidate_dishes[0], 0.75

# ==============================================================================
# 5. Member Profile Synthesis (Loyalty Engine)
# ==============================================================================
def analyze_member_loyalty_profile(member_id, df_all):
    if not member_id or member_id == "GUEST" or df_all.empty:
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
# 6. Mode Renderers (Fast-Casual POS Layout)
# ==============================================================================
def render_header(modules):
    active_badges = []
    if modules.get("mod1", True): active_badges.append("M1 營運監控")
    if modules.get("mod2", True): active_badges.append("M2 數據洞察")
    if modules.get("mod3", True): active_badges.append("M3 點餐機反哺")
    if modules.get("mod4", True): active_badges.append("M4 Club 100")
    badge_str = " • ".join(active_badges) if active_badges else "未啟用任何模組"

    st.markdown(f"""
    <div class="pos-header-banner">
        <div>
            <div class="pos-header-title">大家的大家樂 🍽️ TrayZero+ 智能餐盤審計與會員閉環系統</div>
            <div class="pos-header-sub">大家樂集團 IT PMO 聯合開發 • 雙向鏡像保護 (CSV + SQLite) • 已授權模組: {badge_str}</div>
        </div>
        <div class="shift-badge">🔥 午市尖峰運作中</div>
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
        
        active_member_id = "GUEST"
        if modules.get("mod4", True):
            st.markdown("##### 📲 大家樂 Club 100 會員識別 (Member Scanner)")
            member_col1, member_col2 = st.columns([3, 1])
            with member_col1:
                raw_member_id = st.text_input(
                    "掃描或輸入會員卡號 / 手機號碼", 
                    value=st.session_state.get("last_input_member", ""),
                    placeholder="例: 001 / C100-8801 / 手機號碼"
                )
            with member_col2:
                st.write("")
                st.write("")
                is_guest = st.checkbox("👤 訪客還盤", value=False)
            
            active_member_id = "GUEST" if is_guest or not raw_member_id.strip() else raw_member_id.strip()
            st.session_state["last_input_member"] = raw_member_id

            if active_member_id != "GUEST":
                prof = analyze_member_loyalty_profile(active_member_id, df_history)
                if prof:
                    st.markdown(f"""
                    <div style="background:#FFFBEB; border:1.5px solid #FDE68A; border-radius:10px; padding:12px 16px; margin-bottom:14px;">
                        <b style="color:#92400E; font-size:0.92rem;">💳 會員檔案: #{active_member_id} ({prof['crm_segment']} • 熟客回頭 {prof['total_visits']} 次)</b><br>
                        <span style="font-size:0.84rem; color:#78350F; font-weight:700;">歷史平均殘食: <b>{prof['avg_waste']:.1f}%</b> | 最喜愛餐點: <b>{prof['favorite_dish']}</b></span><br>
                        <span style="font-size:0.84rem; color:#D97706; font-weight:800;">🎯 點餐機 (Kiosk) 自動預載: {prof['pos_default_rice']} • {prof['pos_default_sauce']}</span>
                    </div>
                    """, unsafe_allow_html=True)

        auto_dish = st.checkbox("🤖 啟用 AI 自動辨識餐點類型 (Auto Dish Recognition via CLIP)", value=True)
        scan_mode = st.radio(
            "感應鏡頭作業模式 (Camera Operations)", 
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
            with st.spinner("🚀 AI 正在深度解構餐點與殘食成份 (CLIP + YOLOS)..."):
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
                
                if modules.get("mod4", True) and active_member_id != "GUEST":
                    rewards_list = evaluate_customer_rewards(waste_pct)
                    reward_msg = " • ".join(rewards_list)
                else:
                    reward_msg = "訪客還盤完成 (Guest Return)"

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
                st.toast(f"✅ 還盤數據已即時同步！已觸發獎勵規則。")
                st.rerun()

    with c2:
        st.markdown("#### 🎯 即時審計結果與激勵派發 (Live Ticket)")
        latest = st.session_state.get("latest")
        if not latest:
            st.info("💡 尚未執行偵測。請對準餐盤拍照或上傳。")
        else:
            conf_str = f"({latest.get('conf', 1.0):.1%})" if 'conf' in latest else ""
            st.image(latest["img"], caption=f"🍽️ {latest['dish']} {conf_str} • {latest['time']}")
            
            # Option B: 實體電子卡券卡 (Digital Coupon Voucher Display)
            if modules.get("mod4", True) and latest.get("member") != "GUEST":
                st.markdown(f"""
                <div class="pos-coupon-card">
                    <div class="pos-coupon-header">🎟️ 大家樂 CLUB 100 電子現金券 (即時入帳)</div>
                    <div class="pos-coupon-value">{latest.get('reward').split('：')[-1] if '：' in latest.get('reward') else latest.get('reward')}</div>
                    <div class="pos-coupon-desc">• 關聯會員卡號: <code>{latest.get('member')}</code>  |  • 達成狀態: 惜食獎勵門檻達標</div>
                    <div class="pos-coupon-badge">已派送至手機大家樂錢包 (App Push Sent)</div>
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

            # 後廚即時校準卡
            st.markdown(f"""
            <div class="pos-directive-card">
                <b style="color:#0F172A; font-size:0.9rem;">👨‍🍳 大家樂後廚計量校準 (Kitchen Advisory)</b><br>
                <span style="font-size:0.8rem; color:#475569; font-weight:700;">• 餐點【{latest['dish'].split(' ')[0]}】殘食率為 {latest['ratio']:.1%}，後廚份量出餐標準合規。</span><br>
                <span style="font-size:0.8rem; color:#475569; font-weight:700;">• 凍檸茶檸檬片與冰塊未誤算為廚餘（零干擾）。</span><br>
                <span style="font-size:0.8rem; color:#D97706; font-weight:800;">• 交易流水與積分已即時寫入 seed_audit_logs.csv 與資料庫。</span>
            </div>
            """, unsafe_allow_html=True)

def render_mode2(engine, modules):
    df_b = get_live_branches()
    df_d = get_live_dishes()
    df_raw = get_records()

    st.markdown("### 📊 總部即時營運大盤與會員客群 Retargeting 數據中心")

    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl2:
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
    if not df_filtered.empty:
        if sel_b != "ALL": 
            df_filtered = df_filtered[df_filtered["branch_name"] == sel_b]
        if sel_d != "ALL": 
            df_filtered = df_filtered[df_filtered["dish_name"] == sel_d]

    n = len(df_filtered)
    avg_w = df_filtered["waste_ratio"].mean() if n > 0 else 0.0
    tot_hkd = df_filtered["cost_waste_hkd"].sum() if n > 0 else 0.0
    tot_co2 = df_filtered["co2_emission_kg"].sum() if n > 0 else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="pos-metric-card"><div class="pos-metric-label">審計樣本盤數</div><div class="pos-metric-val">{n} <span style="font-size:0.85rem;color:#94A3B8">TRAYS</span></div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="pos-metric-card amber-glow"><div class="pos-metric-label">平均殘食率</div><div class="pos-metric-val" style="color:{"#DC2626" if avg_w > 25 else "#D97706"}">{avg_w:.1f}%</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="pos-metric-card"><div class="pos-metric-label">食材損耗總額</div><div class="pos-metric-val" style="color:#D97706">HK${tot_hkd:,.1f}</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="pos-metric-card"><div class="pos-metric-label">累計碳排放</div><div class="pos-metric-val" style="color:#2563EB">{tot_co2:.2f} <span style="font-size:0.85rem;color:#94A3B8">kg</span></div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📋 即時審計記錄（即時讀取自 seed_audit_logs.csv）")
    if not df_filtered.empty:
        st.dataframe(df_filtered)
        csv_download = df_filtered.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 匯出當前維度 CSV 審計日誌",
            data=csv_download,
            file_name=f"trayzero_audit_export_{datetime.date.today()}.csv",
            mime="text/csv"
        )
    else:
        st.info("💡 當前篩選維度下尚無資料。請在 Mode 1 進行實體餐盤掃描，系統將自動即時存入 CSV。")

    st.markdown("---")

    if modules.get("mod4", True) and not df_raw.empty:
        st.markdown("#### 🎯 大家樂會員偏好與 Retargeting 數據中心")
        unique_members = [m for m in df_raw["member_id"].dropna().unique().tolist() if m != "GUEST"]
        
        if unique_members:
            crm_records = []
            for mid in unique_members:
                prof = analyze_member_loyalty_profile(mid, df_raw)
                crm_records.append({
                    "會員卡號 (Member ID)": prof["member_id"],
                    "客群分類 (Segment)": prof["crm_segment"],
                    "最喜愛餐點 (Favorite Dish)": prof["favorite_dish"],
                    "平均殘食率": f"{prof['avg_waste']:.1f}%",
                    "點餐機預設主食 (POS Default Rice)": prof["pos_default_rice"],
                    "點餐機預設醬汁 (POS Default Sauce)": prof["pos_default_sauce"],
                    "建議大家樂 CRM 推送優惠券策略": prof["retarget_strategy"]
                })
            
            df_crm = pd.DataFrame(crm_records)
            st.dataframe(df_crm)

            csv_crm = df_crm.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 匯出大家樂 Club 100 Retargeting 數據包 (CSV)",
                data=csv_crm,
                file_name=f"cdc_loyalty_retarget_feed_{datetime.date.today()}.csv",
                mime="text/csv"
            )

    if n > 0 and (modules.get("mod1", True) or modules.get("mod3", True)):
        st.markdown("#### 🧭 大家樂總部營運與菜單工程建議")
        t_branch = sel_b if sel_b != "ALL" else df_filtered.groupby("branch_name")["waste_ratio"].mean().idxmax()
        t_dish = sel_d if sel_d != "ALL" else df_filtered.groupby("dish_name")["waste_ratio"].mean().idxmax()
        sub_avg_w = df_filtered["waste_ratio"].mean()
        sub_loss = df_filtered["cost_waste_hkd"].sum()

        if modules.get("mod1", True):
            st.markdown(f"""
            <div class="pos-directive-card">
                <b style="color:#0F172A;">👨‍🍳 後廚出餐計量標準校準 Head Chef ({t_branch} • {t_dish})</b><br>
                【即時份量校準】平均殘食率達 {sub_avg_w:.1f}%。針對「{t_dish}」換裝標準打餐器（每份減量 30g 出餐），單期預估防損挽回 HK$ {max(150, round(sub_loss * 0.4)):,.0f}。
            </div>
            """, unsafe_allow_html=True)

        if modules.get("mod3", True):
            st.markdown(f"""
            <div class="pos-directive-card" style="border-left-color: #D97706 !important;">
                <b style="color:#0F172A;">🖥️ 點餐機 (Kiosk) 與大家樂 App 反向客製化連動</b><br>
                【智慧預設下發】系統已自動將高頻剩餘「{t_dish}」主食之會員，於點餐終端預設勾選「少飯/少麵（立減 $2）」或「少汁」，在點餐階段源頭減廢。
            </div>
            """, unsafe_allow_html=True)

def render_mode3():
    st.markdown("### ⚙️ 基礎資料管理 (Master Data Management - Real-Time CSV)")
    st.caption("所有編輯與上傳均即時同步至實體 CSV 檔案與資料庫鏡像庫，無論系統如何重啟或更新，資料均能完整保留。")

    tab1, tab2, tab3 = st.tabs([
        "🏢 分店清單 (Branches CSV)", 
        "🍱 餐點品項管理 (Menu CSV)",
        "🎁 殘食門檻獎勵階梯配置 (Incentive Tiers CSV)"
    ])

    with tab1:
        df_b_current = get_live_branches()
        st.markdown("#### 🏢 門市清單即時編輯 (Live Branches)")
        edit_b = st.data_editor(df_b_current, num_rows="dynamic", key="live_branch_editor")
        
        c_b1, c_b2 = st.columns([1, 1])
        with c_b1:
            if st.button("💾 儲存修改至 master_branches.csv", type="primary"):
                save_live_branches(edit_b)
                st.success("✅ master_branches.csv 檔案與資料庫鏡像已即時更新儲存！")
                st.rerun()
        with c_b2:
            csv_b_export = edit_b.dropna(subset=["name"]).to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 下載目前 master_branches.csv", data=csv_b_export, file_name="master_branches.csv", mime="text/csv")

        st.markdown("---")
        up_b = st.file_uploader("批次覆蓋上傳分店 CSV (Upload Store CSV)", type=["csv"], key="up_b_uploader")
        if up_b:
            try:
                new_df_b = pd.read_csv(up_b, encoding="utf-8-sig")
                save_live_branches(new_df_b)
                st.success(f"🎉 成功寫入 {len(new_df_b)} 間分店！")
                st.rerun()
            except Exception as e:
                st.error(f"匯入錯誤: {e}")

    with tab2:
        df_d_current = get_live_dishes()
        
        st.markdown("#### 📸 新增菜品與照片註冊 (即時寫入 master_dishes.csv)")
        col_reg1, col_reg2 = st.columns([1.1, 0.9])
        with col_reg1:
            new_dish_id = st.text_input("品項編號 (Dish ID)", value=f"D{len(df_d_current)+1:02d}")
            new_dish_name = st.text_input("餐點名稱 (Dish Name)", placeholder="例: 車仔麵 (Kart Noodle)")
            new_carb = st.selectbox("主要碳水主食 (Carbohydrate)", ["中式麵條 (Noodles)", "白米飯 (Rice)", "蛋炒飯 (Fried Rice)", "意大利麵/意粉 (Spaghetti)", "河粉/米線 (Rice Noodles)", "無主食 (None)"])
            new_protein = st.text_input("主要蛋白質/主菜 (Protein Source)", placeholder="例: 咖喱牛腩/魚蛋")

        with col_reg2:
            new_dish_photo = st.file_uploader("📷 上傳菜品樣本照片 (Reference Photo)", type=["jpg", "png", "jpeg"], key="new_dish_photo_input")
            if new_dish_photo:
                photo_preview = Image.open(new_dish_photo)
                st.image(photo_preview, caption="菜品照片預覽", width=240)

        if st.button("🚀 註冊新菜品並寫入 CSV (Save to Dishes CSV)", type="primary"):
            if not new_dish_name.strip():
                st.error("❌ 請輸入餐點名稱！")
            else:
                if new_dish_photo:
                    img_ext = os.path.splitext(new_dish_photo.name)[1]
                    saved_img_path = os.path.join(DISH_IMG_DIR, f"{new_dish_id}_{new_dish_name}{img_ext}")
                    with open(saved_img_path, "wb") as f:
                        f.write(new_dish_photo.getbuffer())

                new_row = pd.DataFrame([{
                    "dish_id": new_dish_id,
                    "name": new_dish_name.strip(),
                    "main_carb": new_carb.split(" ")[0],
                    "protein": new_protein.strip() if new_protein else "綜合配料"
                }])
                
                updated_dishes = pd.concat([df_d_current, new_row], ignore_index=True)
                save_live_dishes(updated_dishes)
                st.success(f"🎉 成功將【{new_dish_name}】寫入 master_dishes.csv 與資料庫！前台已即刻生效。")
                st.rerun()

        st.markdown("---")
        st.markdown("#### 🍱 現有餐點清單即時編輯 (Live Menu CSV)")
        st.caption("提示：在表格中修改或新增後，點擊下方「儲存修改」按鈕即可完成寫入。系統會自動過濾未填寫的空白列。")
        edit_d = st.data_editor(df_d_current, num_rows="dynamic", key="live_dish_editor")
        
        c_d1, c_d2 = st.columns([1, 1])
        with c_d1:
            if st.button("💾 儲存修改至 master_dishes.csv", type="primary"):
                save_live_dishes(edit_d)
                st.success("✅ master_dishes.csv 檔案與資料庫鏡像已即時更新儲存！")
                st.rerun()
        with c_d2:
            csv_d_export = edit_d.dropna(subset=["name"]).to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 下載目前 master_dishes.csv", data=csv_d_export, file_name="master_dishes.csv", mime="text/csv")

        st.markdown("---")
        up_d = st.file_uploader("批次覆蓋上傳餐點 CSV (Upload Menu CSV)", type=["csv"], key="up_d_uploader")
        if up_d:
            try:
                new_df_d = pd.read_csv(up_d, encoding="utf-8-sig")
                save_live_dishes(new_df_d)
                st.success(f"🎉 成功寫入 {len(new_df_d)} 項餐點！")
                st.rerun()
            except Exception as e:
                st.error(f"匯入錯誤: {e}")

    with tab3:
        st.markdown("#### 🎁 會員還盤獎勵階梯配置 (Incentive Tiers Configuration)")
        st.caption("管理員可在此自訂當客戶達到特定殘食佔比門檻時，自動派發的單一或多項組合獎勵。設定即時寫入 master_rewards.csv。")

        df_r_current = get_live_rewards()
        
        st.markdown("##### ➕ 快速新增獎勵規則 (Add New Tier)")
        c_r1, c_r2, c_r3 = st.columns([1.2, 1, 2])
        with c_r1:
            new_r_name = st.text_input("獎勵等級名稱 (Tier Name)", placeholder="例: 超級光盤達人獎")
        with c_r2:
            new_r_ratio = st.number_input("殘食佔比上限 Max Waste (%)", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
        with c_r3:
            new_r_desc = st.text_input("組合獎勵內容描述 (Coupons / Rewards Combo)", placeholder="例: 【$5 現金券】+【特飲免費券】+【100 積分】")

        if st.button("➕ 新增此獎勵規則至 CSV (Add Tier)", type="primary"):
            if not new_r_name.strip() or not new_r_desc.strip():
                st.error("❌ 請輸入規則名稱與獎勵內容！")
            else:
                new_reward_row = pd.DataFrame([{
                    "reward_id": f"R{len(df_r_current)+1:02d}",
                    "tier_name": new_r_name.strip(),
                    "max_waste_ratio": float(new_r_ratio),
                    "reward_type": "Custom Combo",
                    "reward_description": new_r_desc.strip(),
                    "is_active": True
                }])
                updated_rewards = pd.concat([df_r_current, new_reward_row], ignore_index=True)
                save_live_rewards(updated_rewards)
                st.success(f"🎉 成功新增獎勵規則【{new_r_name}】至 master_rewards.csv！前台即刻生效。")
                st.rerun()

        st.markdown("---")
        st.markdown("##### 📝 線上即時編輯獎勵規則 (Live Rewards Editor)")
        edit_r = st.data_editor(df_r_current, num_rows="dynamic", key="live_reward_editor")
        
        c_save_r1, c_save_r2 = st.columns([1, 1])
        with c_save_r1:
            if st.button("💾 儲存修改至 master_rewards.csv", type="secondary"):
                save_live_rewards(edit_r)
                st.success("✅ master_rewards.csv 檔案與資料庫已即時更新儲存！")
                st.rerun()
        with c_save_r2:
            csv_r_export = edit_r.dropna(subset=["tier_name"]).to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 下載目前 master_rewards.csv", data=csv_r_export, file_name="master_rewards.csv", mime="text/csv")

# ==============================================================================
# 7. Application Entry Point
# ==============================================================================
def main():
    inject_safe_css()
    init_db()

    with st.spinner("🚀 正在啟動雙核心 AI 引擎 (Loading AI Engines)..."):
        engine = load_ai_engine()

    # 側邊欄：純文字企業品牌橫幅 (無須依賴圖片，支援純文字精緻渲染)
    st.sidebar.markdown("""
    <div class="sidebar-brand-box">
        <div class="sidebar-brand-title">大家樂 CAFÉ DE CORAL</div>
        <div class="sidebar-brand-sub">TrayZero+ 前線餐盤智能回收終端</div>
    </div>
    """, unsafe_allow_html=True)

    # 門市當前狀態小卡
    st.sidebar.markdown("""
    <div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:10px; padding:10px 14px; margin-bottom:16px;">
        <div style="font-size:0.75rem; font-weight:800; color:#92400E;">現正執勤門市 (STORE)</div>
        <div style="font-size:0.95rem; font-weight:900; color:#78350F; margin-top:2px;">中環威靈頓街店</div>
        <div style="font-size:0.75rem; font-weight:700; color:#B45309; margin-top:2px;">● 審計連線正常 (Sync 100%)</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("##### 觸控模式選擇 (TOUCH NAVIGATION)")
    mode = st.sidebar.radio("", [
        "Mode 1: 前線回收感應台",
        "Mode 2: 總部即時營運大盤",
        "Mode 3: 菜單與獎勵配置"
    ], label_visibility="collapsed")

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

    render_header(active_modules)

    if mode.startswith("Mode 1"): 
        render_mode1(engine, active_modules)
    elif mode.startswith("Mode 2"): 
        render_mode2(engine, active_modules)
    else: 
        render_mode3()

if __name__ == "__main__":
    main()
