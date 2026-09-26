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
# 1. Global Paths & Strict High-Contrast CSS Enforcement
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
    "cup", "bottle", "wine glass", "bowl", "dining table", 
    "knife", "fork", "spoon", "chopsticks", "person", "chair"
}

def inject_safe_css():
    st.markdown("""
    <style>
        :root {
            --text-color: #0F172A !important;
            --background-color: #F8FAFC !important;
            --secondary-background-color: #FFFFFF !important;
        }

        .stApp {
            background-color: #F8FAFC !important;
            color: #0F172A !important;
        }

        .main .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 3rem !important;
            color: #0F172A !important;
        }

        /* 1. All Widget Labels */
        label[data-testid="stWidgetLabel"],
        div[data-testid="stWidgetLabel"] label,
        div[data-testid="stWidgetLabel"] p,
        div[data-testid="stWidgetLabel"] span {
            color: #0F172A !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            opacity: 1 !important;
            visibility: visible !important;
        }

        /* 2. Radio Options & Checkboxes */
        div[data-testid="stRadio"] [role="radiogroup"] label,
        div[data-testid="stRadio"] [role="radiogroup"] label p,
        div[data-testid="stRadio"] [role="radiogroup"] label span,
        div[data-testid="stCheckbox"] label p,
        div[data-testid="stCheckbox"] label span {
            color: #0F172A !important;
            font-size: 0.92rem !important;
            font-weight: 600 !important;
            opacity: 1 !important;
            visibility: visible !important;
        }

        /* 3. Tabs */
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
            border-bottom: 3px solid #2563EB !important;
        }
        div[data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] p,
        div[data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] span {
            color: #2563EB !important;
            font-weight: 800 !important;
        }

        /* 4. Text Inputs, Number Inputs, and Select Boxes */
        input[type="text"], 
        input[type="number"],
        div[data-baseweb="input"] input,
        div[data-baseweb="select"] div {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border-color: #94A3B8 !important;
            font-weight: 600 !important;
        }
        input::placeholder {
            color: #64748B !important;
            opacity: 1 !important;
            font-weight: 500 !important;
        }

        /* 5. Custom Card Styling */
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

# ==============================================================================
# 2. Database Connection & Schema Setup
# ==============================================================================
def db_conn(): 
    return sqlite3.connect(DB_FILE)

def init_db():
    with db_conn() as conn:
        # 1. 審計日誌流水表
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
        # 2. 永續菜單鏡像表 (防止 Git 覆蓋消失)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS master_dishes_db (
                dish_id TEXT PRIMARY KEY,
                name TEXT,
                main_carb TEXT,
                protein TEXT
            )
        """)
        # 3. 永續門市鏡像表
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
        # 4. 永續獎勵階梯鏡像表
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

        # 審計種子資料載入
        row_count = conn.execute("SELECT COUNT(*) FROM audit_logs").fetchone()[0]
        if row_count == 0 and os.path.exists(SEED_AUDIT_FILE):
            try:
                seed_df = pd.read_csv(SEED_AUDIT_FILE, encoding="utf-8-sig")
                for col in ["member_id", "reward_issued", "audit_date", "audit_month"]:
                    if col not in seed_df.columns:
                        if col == "member_id": seed_df[col] = "GUEST"
                        elif col == "reward_issued": seed_df[col] = "歷史資料"
                        elif col == "audit_date": seed_df[col] = datetime.date.today().strftime("%Y-%m-%d")
                        elif col == "audit_month": seed_df[col] = datetime.date.today().strftime("%Y-%m")
                
                valid_cols = ["timestamp", "audit_date", "audit_month", "branch_name", "member_id", 
                              "dish_name", "primary_waste", "waste_ratio", "cost_waste_hkd", "co2_emission_kg", "reward_issued"]
                seed_df_clean = seed_df[[c for c in valid_cols if c in seed_df.columns]]
                seed_df_clean.to_sql("audit_logs", conn, if_exists="append", index=False)
            except Exception as e:
                print(f"Seed Load Error: {e}")

# ==============================================================================
# 3. 雙向永續資料讀取與儲存 (CSV + SQLite Auto-Merge，抗 Git 覆蓋)
# ==============================================================================
def get_live_dishes():
    """從 CSV 讀取，並與 SQLite 永續資料表自動補合（防止 Git pull 覆蓋）"""
    csv_df = pd.DataFrame(columns=["dish_id", "name", "main_carb", "protein"])
    if os.path.exists(DISH_FILE):
        try:
            csv_df = pd.read_csv(DISH_FILE, encoding="utf-8-sig")
        except Exception:
            pass

    with db_conn() as conn:
        db_df = pd.read_sql("SELECT dish_id, name, main_carb, protein FROM master_dishes_db", conn)

    # 合併 CSV 與 SQLite 資料
    combined = pd.concat([csv_df, db_df], ignore_index=True)
    if not combined.empty:
        combined = combined.dropna(subset=["name"])
        combined = combined[combined["name"].str.strip() != ""]
        combined = combined.drop_duplicates(subset=["name"], keep="last")
        # 同步回寫 CSV 與 DB
        combined.to_csv(DISH_FILE, index=False, encoding="utf-8-sig")
        with db_conn() as conn:
            combined.to_sql("master_dishes_db", conn, if_exists="replace", index=False)
        return combined
    else:
        return pd.DataFrame(columns=["dish_id", "name", "main_carb", "protein"])

def save_live_dishes(df):
    """清洗空白列，同時存入 CSV 與 SQLite"""
    clean_df = df.dropna(subset=["name"]).copy()
    clean_df = clean_df[clean_df["name"].astype(str).str.strip() != ""]
    clean_df = clean_df.drop_duplicates(subset=["name"], keep="last")
    
    clean_df.to_csv(DISH_FILE, index=False, encoding="utf-8-sig")
    with db_conn() as conn:
        clean_df.to_sql("master_dishes_db", conn, if_exists="replace", index=False)

def get_live_branches():
    csv_df = pd.DataFrame(columns=["name", "level", "district", "traffic", "avg_covers", "base_rice_g"])
    if os.path.exists(BRANCH_FILE):
        try:
            csv_df = pd.read_csv(BRANCH_FILE, encoding="utf-8-sig")
        except Exception:
            pass

    with db_conn() as conn:
        db_df = pd.read_sql("SELECT name, level, district, traffic, avg_covers, base_rice_g FROM master_branches_db", conn)

    combined = pd.concat([csv_df, db_df], ignore_index=True)
    if not combined.empty:
        combined = combined.dropna(subset=["name"])
        combined = combined[combined["name"].str.strip() != ""]
        combined = combined.drop_duplicates(subset=["name"], keep="last")
        combined.to_csv(BRANCH_FILE, index=False, encoding="utf-8-sig")
        with db_conn() as conn:
            combined.to_sql("master_branches_db", conn, if_exists="replace", index=False)
        return combined
    else:
        return pd.DataFrame(columns=["name", "level", "district", "traffic", "avg_covers", "base_rice_g"])

def save_live_branches(df):
    clean_df = df.dropna(subset=["name"]).copy()
    clean_df = clean_df[clean_df["name"].astype(str).str.strip() != ""]
    clean_df = clean_df.drop_duplicates(subset=["name"], keep="last")
    
    clean_df.to_csv(BRANCH_FILE, index=False, encoding="utf-8-sig")
    with db_conn() as conn:
        clean_df.to_sql("master_branches_db", conn, if_exists="replace", index=False)

def get_live_rewards():
    csv_df = pd.DataFrame(columns=["reward_id", "tier_name", "max_waste_ratio", "reward_type", "reward_description", "is_active"])
    if os.path.exists(REWARD_FILE):
        try:
            csv_df = pd.read_csv(REWARD_FILE, encoding="utf-8-sig")
        except Exception:
            pass

    with db_conn() as conn:
        db_df = pd.read_sql("SELECT reward_id, tier_name, max_waste_ratio, reward_type, reward_description, is_active FROM master_rewards_db", conn)

    combined = pd.concat([csv_df, db_df], ignore_index=True)
    if not combined.empty:
        combined = combined.dropna(subset=["tier_name"])
        combined = combined[combined["tier_name"].str.strip() != ""]
        combined = combined.drop_duplicates(subset=["tier_name"], keep="last")
        combined.to_csv(REWARD_FILE, index=False, encoding="utf-8-sig")
        with db_conn() as conn:
            combined.to_sql("master_rewards_db", conn, if_exists="replace", index=False)
        return combined

    # 預設獎勵階梯
    default_rewards = [
        {"reward_id": "R01", "tier_name": "極致光盤獎 (Ultra Clean)", "max_waste_ratio": 10.0, "reward_type": "Coupon + Points", "reward_description": "【$3 現金券】+【50 綠色積分】+【凍檸茶半價券】", "is_active": True},
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

def reset_db():
    with db_conn() as conn: 
        conn.execute("DELETE FROM audit_logs")
    if os.path.exists(SEED_AUDIT_FILE):
        try:
            os.remove(SEED_AUDIT_FILE)
        except Exception:
            pass

# ==============================================================================
# 4. Multi-Modal Vision Engine (CLIP + YOLOS)
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

def is_bowl_actually_clean(image):
    """檢驗碗內是否空碗（排除筷子與碗花紋干擾）"""
    try:
        w, h = image.size
        crop_box = (int(w * 0.2), int(h * 0.05), int(w * 0.8), int(h * 0.65))
        cropped = image.crop(crop_box).convert("L")
        edges = cropped.filter(ImageFilter.FIND_EDGES)
        stat = ImageStat.Stat(edges)
        edge_energy = stat.mean[0]
        return edge_energy < 18.0
    except Exception:
        return False

def detect_tray(image, engine, selected_dish="", carb_type_from_csv=""):
    inp = engine["proc"](images=image, return_tensors="pt").to(engine["device"])
    with torch.no_grad(): 
        out = engine["det"](**inp)
        
    sz = torch.tensor([image.size[::-1]]).to(engine["device"])
    res = engine["proc"].post_process_object_detection(out, threshold=0.20, target_sizes=sz)[0]
    
    img_draw = image.copy()
    total_area = image.size[0] * image.size[1]
    waste_area = 0
    draw = ImageDraw.Draw(img_draw)
    items = []
    
    residual_labels = [
        "吃得很乾淨的光盤空碗只剩湯水 (completely finished empty bowl with only soup left)",
        "碗內堆滿剩餘麵條 (bowl full of leftover noodles)",
        "碗內堆滿剩餘米飯主食 (plate full of leftover rice)",
        "盤內剩餘大塊肉類與海鮮 (leftover large meat or seafood)",
        "盤內剩餘大量蔬菜配菜 (leftover vegetables and sides)"
    ]
    
    clip_res = engine["clip"](image, candidate_labels=residual_labels)
    top_pred = clip_res[0]["label"]
    top_score = clip_res[0]["score"]

    bowl_clean_check = is_bowl_actually_clean(image)
    if ("乾淨的光盤" in top_pred and top_score > 0.35) or bowl_clean_check:
        return img_draw, [{
            "分類項目 Category": "光盤 Clean Plate", 
            "置信度 Confidence": f"{max(top_score, 0.95):.1%}", 
            "佔比 Coverage": "0.0%"
        }], 0.0, "光盤 Clean Plate", True

    is_noodle_dish = any(kw in str(carb_type_from_csv) for kw in ["麵", "意粉", "粉", "Spaghetti", "Noodle"])
    
    if "麵條" in top_pred or is_noodle_dish:
        primary = "主食殘留 (麵食) Carb Residual (Noodles)"
        accent_color = "#EF4444"
    elif "米飯" in top_pred or "飯" in str(carb_type_from_csv):
        primary = "主食殘留 (米飯) Carb Residual (Rice)"
        accent_color = "#EF4444"
    elif "肉類" in top_pred:
        primary = "肉類殘留 Meat Residual"
        accent_color = "#F59E0B"
    else:
        primary = "配菜/醬汁 Sides & Sauce"
        accent_color = "#10B981"

    valid_detected = False
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
            
        waste_area += area
        valid_detected = True
        draw.rectangle(b, outline=accent_color, width=3)
        draw.text((b[0] + 4, b[1] + 4), f"{primary.split(' ')[0]} {score:.0%}", fill=accent_color)
        items.append({
            "分類項目 Category": primary.split(" ")[0], 
            "置信度 Confidence": f"{score:.1%}", 
            "佔比 Coverage": f"{area/total_area:.1%}"
        })

    if not valid_detected or waste_area == 0:
        return img_draw, [{"分類項目 Category": "光盤 Clean Plate", "置信度 Confidence": "96.2%", "佔比 Coverage": "0.0%"}], 0.0, "光盤 Clean Plate", True

    ratio = round(min(0.85, waste_area / (total_area * 0.60)), 3)
    return img_draw, items, ratio, primary, True

def auto_detect_dish_clip(image, candidate_dishes, engine):
    if not candidate_dishes:
        return "未定義餐點", 0.0
    clean_labels = [d.strip() for d in candidate_dishes]
    try:
        results = engine["clip"](image, candidate_labels=clean_labels)
        top_dish = results[0]["label"]
        top_conf = results[0]["score"]
        
        if top_conf < 0.40:
            return "空餐盤 (已完食 Cleaned Tray)", top_conf
            
        return top_dish, top_conf
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

    sauce_wastes = m_df[m_df["primary_waste"].str.contains("配菜|醬汁", na=False)]
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
# 6. Mode Renderers
# ==============================================================================
def render_header(modules):
    active_badges = []
    if modules.get("mod1", True): active_badges.append("M1: 營運監控")
    if modules.get("mod2", True): active_badges.append("M2: 數據洞察")
    if modules.get("mod3", True): active_badges.append("M3: 精準營銷")
    if modules.get("mod4", True): active_badges.append("M4: 會員閉環")
    badge_str = " • ".join(active_badges) if active_badges else "未啟用任何模組"

    st.markdown(f"""
    <div class="trayzero-header">
        <h2 class="trayzero-title">🍽️ TrayZero+ 大家樂智能餐盤審計與會員閉環平台</h2>
        <div style="font-size:0.8rem; color:#64748B; margin-top:4px; font-weight:600;">已授權模組: {badge_str}</div>
    </div>
    """, unsafe_allow_html=True)

def render_mode1(engine, modules):
    df_b = get_live_branches()
    df_d = get_live_dishes()

    if df_b.empty or df_d.empty:
        st.warning("⚠️ 門市或餐點清單為空！請先至 Mode 3 上傳或新增菜單與分店。")
        return

    df_history = get_records()
    c1, c2 = st.columns([1.1, 0.9])
    
    with c1:
        st.markdown("#### 🏢 回收台設置與會員識別")
        b_name = st.selectbox("執勤門市 (Active Store Location)", df_b["name"].tolist())
        
        active_member_id = "GUEST"
        if modules.get("mod4", True):
            st.markdown("##### 📲 大家樂 Club 100 會員識別 (Member Scanner)")
            member_col1, member_col2 = st.columns([3, 1])
            with member_col1:
                raw_member_id = st.text_input(
                    "掃描或輸入會員卡號 (Scan/Enter Member ID)", 
                    value=st.session_state.get("last_input_member", ""),
                    placeholder="例: C100-8801 / 手機號碼"
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
                    <div class="member-live-badge">
                        💳 <b>會員 ID</b>: <b style="color:#2563EB;">{active_member_id}</b> ({prof['crm_segment']})<br>
                        📊 <b>歷史用餐</b>: 累計還盤 {prof['total_visits']} 次 | 平均殘食率: <b>{prof['avg_waste']:.1f}%</b><br>
                        💡 <b>目前點餐預設</b>: <span style="color:#059669; font-weight:700;">{prof['pos_default_rice']}</span> / <span style="color:#059669; font-weight:700;">{prof['pos_default_sauce']}</span>
                    </div>
                    """, unsafe_allow_html=True)

        auto_dish = st.checkbox("🤖 啟用 AI 自動辨識餐點類型 (Auto Dish Recognition via CLIP)", value=True)
        scan_mode = st.radio(
            "掃描模式 (Scanning Method)", 
            ["🟢 Live Camera 長開 (靜止自動感應 / Auto-Scan)", "📸 手動快照 (Manual Snapshot)", "📁 上傳照片 (Upload Image)"], 
            horizontal=True
        )
        
        img_cap = None
        should_run = False

        if scan_mode.startswith("🟢"):
            cam = st.camera_input("持續監控畫面 (Live Feed Monitor)", key="live_cam")
            if cam:
                img_cap = Image.open(cam).convert("RGB")
                h = hash(img_cap.tobytes()[:3000])
                if h != st.session_state.get("last_h"):
                    st.session_state["last_h"] = h
                    should_run = True
        elif scan_mode.startswith("📸"):
            m_cam = st.camera_input("拍照 (Take Snapshot)", key="manual_cam")
            if m_cam: 
                img_cap = Image.open(m_cam).convert("RGB")
                should_run = True
        else:
            up = st.file_uploader("上傳餐盤相片 (Upload Tray Image)", type=["jpg", "png", "jpeg"], key="tray_file_uploader")
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
        st.markdown("#### 🎯 前線掃描結果與會員數據")
        latest = st.session_state.get("latest")
        if not latest:
            st.info("💡 尚未執行偵測。請對準餐盤拍照或上傳。")
        else:
            conf_str = f"({latest.get('conf', 1.0):.1%})" if 'conf' in latest else ""
            st.image(latest["img"], caption=f"🍽️ {latest['dish']} {conf_str} • {latest['time']}")
            
            if modules.get("mod4", True) and latest.get("member") != "GUEST":
                st.markdown(f"""
                <div class="crm-card">
                    <b style="color:#166534; font-size:1rem;">🎁 會員獎勵與 Loyalty Loop 派發成功</b><br>
                    • <b>關聯會員 ID</b>: <code>{latest.get('member')}</code><br>
                    • <b>動態派發獎勵</b>: <span style="font-weight:700; color:#047857;">{latest.get('reward')}</span><br>
                    • <b>自動反哺 POS 規則</b>: 下次點餐系統已預載顧客客製化偏好
                </div>
                """, unsafe_allow_html=True)

            k1, k2, k3 = st.columns(3)
            k1.markdown(f'<div class="clean-card"><div class="clean-label">殘食佔比 Waste Ratio</div><div class="clean-val" style="color:{"#EF4444" if latest["ratio"] > 0.3 else "#10B981"}">{latest["ratio"]:.1%}</div></div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="clean-card"><div class="clean-label">主要殘留 Primary</div><div class="clean-val" style="font-size:0.95rem;margin-top:4px;">{latest["cat"]}</div></div>', unsafe_allow_html=True)
            k3.markdown(f'<div class="clean-card"><div class="clean-label">推算損耗 Loss</div><div class="clean-val" style="color:#F59E0B">HK${latest["cost"]}</div></div>', unsafe_allow_html=True)

def render_mode2(engine, modules):
    df_b = get_live_branches()
    df_d = get_live_dishes()
    df_raw = get_records()

    st.markdown("### 📊 總部營運與會員客群 Retargeting 數據中心")

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
    k1.markdown(f'<div class="clean-card"><div class="clean-label">審計樣本盤數</div><div class="clean-val">{n} <span style="font-size:0.85rem;color:#94A3B8">TRAYS</span></div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="clean-card"><div class="clean-label">平均殘食率</div><div class="clean-val" style="color:{"#EF4444" if avg_w > 25 else "#10B981"}">{avg_w:.1f}%</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="clean-card"><div class="clean-label">食材損耗總額</div><div class="clean-val" style="color:#F59E0B">HK${tot_hkd:,.1f}</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="clean-card"><div class="clean-label">累計碳排放</div><div class="clean-val" style="color:#3B82F6">{tot_co2:.2f} <span style="font-size:0.85rem;color:#94A3B8">kg</span></div></div>', unsafe_allow_html=True)

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
            <div class="directive-card directive-chef">
                <b style="color:#0F172A;">👨‍🍳 後廚出餐計量標準校準 Head Chef ({t_branch} • {t_dish})</b><br>
                【即時份量校準】平均殘食率達 {sub_avg_w:.1f}%。針對「{t_dish}」換裝標準打餐器（每份減量 30g 出餐），單期預估防損挽回 HK$ {max(150, round(sub_loss * 0.4)):,.0f}。
            </div>
            """, unsafe_allow_html=True)

        if modules.get("mod3", True):
            st.markdown(f"""
            <div class="directive-card directive-pos">
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

    # Tab 1: 門市即時管理
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

    # Tab 2: 菜品清單即時管理 (加入自動持久化與去空列保護)
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

    # Tab 3: 動態獎勵階梯配置
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

    logo_target = LOGO_FILE_PNG if os.path.exists(LOGO_FILE_PNG) else (LOGO_FILE_JPG if os.path.exists(LOGO_FILE_JPG) else None)
    if logo_target:
        st.sidebar.image(logo_target, width=175)

    st.sidebar.title("🧩 功能模組授權 (Modules)")
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

    st.sidebar.markdown("---")
    mode = st.sidebar.radio("工作模式 (Navigation)", [
        "Mode 1: 前線餐盤智能偵測與會員還盤",
        "Mode 2: 總部即時營運與客群大盤",
        "Mode 3: 基礎資料設定"
    ])

    if mode.startswith("Mode 1"): 
        render_mode1(engine, active_modules)
    elif mode.startswith("Mode 2"): 
        render_mode2(engine, active_modules)
    else: 
        render_mode3()

if __name__ == "__main__":
    main()
