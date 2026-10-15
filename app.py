import re
import time
import os
import datetime
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoImageProcessor,
    AutoModelForImageClassification,
    pipeline
)
import altair as alt
)
import altair as alt

# ==============================================================================
# 1. Page Configuration & Custom Styling
# ==============================================================================
st.set_page_config(
    page_title="TrayZero+ | 智能餐盤審計與會員獎勵系統",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
            margin-bottom: 16px !important;
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
# 2. Database & Data Store Layer (Fully Decoupled Chinese / English)
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

def init_db():
    conn = db_conn()
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS dishes (
            dish_id TEXT PRIMARY KEY,
            name_zh TEXT,
            name_en TEXT,
            main_carb_zh TEXT,
            main_carb_en TEXT,
            protein_zh TEXT,
            protein_en TEXT
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS branches (
            branch_id TEXT PRIMARY KEY,
            name_zh TEXT,
            name_en TEXT,
            level_zh TEXT,
            level_en TEXT,
            district_zh TEXT,
            district_en TEXT,
            traffic_zh TEXT,
            traffic_en TEXT,
            avg_covers INTEGER,
            base_rice_g INTEGER,
            strategy_zh TEXT,
            strategy_en TEXT
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rewards (
            reward_id TEXT PRIMARY KEY,
            tier_name_zh TEXT,
            tier_name_en TEXT,
            max_waste_ratio REAL,
            reward_type TEXT,
            reward_description_zh TEXT,
            reward_description_en TEXT,
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
        ("D01", "一哥焗豬扒飯", "Baked Pork Chop Rice", "白米飯", "Steamed Jasmine Rice", "焗厚切豬扒", "Baked Thick-Cut Pork Chop"),
        ("D05", "焗肉醬意粉", "Baked Spaghetti Bolognese", "意大利麵", "Spaghetti Pasta", "慢燉牛肉醬", "Slow-Cooked Beef Bolognese"),
        ("D03", "燒味飯", "Siu Mei Rice", "白米飯", "Steamed Jasmine Rice", "燒味", "Roasted BBQ Meat"),
        ("D04", "干炒牛河", "Fried Beef Noodles", "中式麵條", "Flat Rice Noodles", "牛肉", "Tender Sliced Beef"),
        ("D02", "咖喱牛腩飯", "Curry Beef Brisket Rice", "白米飯", "Steamed Jasmine Rice", "牛腩", "Stewed Beef Brisket"),
        ("D06", "滑蛋蝦仁飯", "Scrambled Egg Shrimp Rice", "白米飯", "Steamed Jasmine Rice", "蝦仁", "Fresh Shrimps with Egg"),
        ("D07", "香辣肉燥肉餅飯", "Minced Pork Patty Rice", "白米飯", "Steamed Jasmine Rice", "肉餅", "Minced Pork Patty"),
        ("D08", "車仔麵", "Cart Noodles", "中式麵條", "Hong Kong Noodles", "什錦配料", "Assorted Delicacies")
    ]
    cur.execute("SELECT COUNT(*) FROM dishes")
    if cur.fetchone()[0] == 0:
        cur.executemany("INSERT OR REPLACE INTO dishes VALUES (?,?,?,?,?,?,?)", default_dishes)
        conn.commit()

    default_branches = [
        ("B01", "中環威靈頓街店", "Wellington Street Branch", "Level A (商業核心區)", "Level A (CBD Core)", "中西區", "Central & Western", "白領上班族為主，午市尖峰翻檯率極高", "Corporate white-collars; peak lunch turnover", 1200, 240, "白領控醣需求高，建議推動小份量出餐與少飯扣減優惠。", "High low-carb demand; promote smaller servings and Less Rice discounts."),
        ("B02", "沙田新城市廣場店", "New Town Plaza Branch", "Level B (住宅商場)", "Level B (Residential Mall)", "沙田區", "Sha Tin", "家庭客、長者與週末休閒客群", "Families, seniors, and weekend leisure diners", 1500, 260, "家庭用餐與兒童共享比例高，建議推動彈性配菜組合。", "High family sharing ratio; promote flexible side dish combos."),
        ("B03", "香港科技大學店", "HKUST Campus Branch", "Level C (校園青年區)", "Level C (University Campus)", "西貢區", "Sai Kung", "學生、教職員，運動量及食量顯著較大", "Students and faculty with active appetite", 1800, 280, "維持大份量以維持飽足感，重點監控肉類醬汁口味與炸物品質。", "Maintain large portions; monitor protein tenderness and gravy quality."),
        ("B04", "將軍澳 Popcorn 店", "PopCorn Mall Branch", "Level B (住宅商場)", "Level B (Residential Mall)", "西貢區", "Sai Kung", "家庭客及換乘鐵路客流", "Families and transit commuters", 1400, 260, "重點監控主食及焗烤類餐點份量。", "Focus on portion control for rice and baked pasta entrees.")
    ]
    cur.execute("SELECT COUNT(*) FROM branches")
    if cur.fetchone()[0] == 0:
        cur.executemany("INSERT OR REPLACE INTO branches VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", default_branches)
        conn.commit()

    default_rewards = [
        ("R01", "極致光盤獎", "Ultra Clean Plate", 10.0, "Coupon + Points", "【$3 堂食現金券】+【50 綠色積分】+【凍檸茶半價券】", "【$3 Cash Voucher】+【50 Green Points】+【Half-Price Iced Lemon Tea】", True),
        ("R02", "達標惜食獎", "Standard Clean Plate", 20.0, "Coupon", "【$2 堂食電子券】+【20 綠色積分】", "【$2 Dining Voucher】+【20 Green Points】", True),
        ("R03", "支持環保獎", "Green Support", 100.0, "Points", "【10 綠色環保積分】", "【10 Green Points】", True)
    ]
    cur.execute("SELECT COUNT(*) FROM rewards")
    if cur.fetchone()[0] == 0:
        cur.executemany("INSERT OR REPLACE INTO rewards VALUES (?,?,?,?,?,?,?,?)", default_rewards)
        conn.commit()

    cur.execute("SELECT COUNT(*) FROM audit_logs")
    if cur.fetchone()[0] == 0 and os.path.exists(SEED_AUDIT_FILE):
        try:
            df_seed = pd.read_csv(SEED_AUDIT_FILE, encoding="utf-8-sig")
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
        except Exception:
            pass

    conn.commit()
    conn.close()

def strip_bilingual_parentheses(text, is_en=False):
    if not text:
        return ""
    s = str(text).strip()
    if " (" in s and s.endswith(")"):
        parts = s.split(" (")
        zh_part = parts[0].strip()
        en_part = parts[1][:-1].strip()
        return en_part if is_en else zh_part
    return s

def get_live_dishes(is_en=False):
    conn = db_conn()
    try:
        df = pd.read_sql_query("SELECT * FROM dishes", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    
    if df.empty and os.path.exists(DISH_FILE):
        try:
            df = pd.read_csv(DISH_FILE, encoding="utf-8-sig")
        except Exception:
            try:
                df = pd.read_csv(DISH_FILE, encoding="utf-8")
            except Exception:
                pass

    if df.empty:
        df = pd.DataFrame([
            {"dish_id": "D01", "name_zh": "一哥焗豬扒飯", "name_en": "Baked Pork Chop Rice", "main_carb_zh": "白米飯", "main_carb_en": "Steamed Jasmine Rice", "protein_zh": "焗厚切豬扒", "protein_en": "Baked Thick-Cut Pork Chop"},
            {"dish_id": "D05", "name_zh": "焗肉醬意粉", "name_en": "Baked Spaghetti Bolognese", "main_carb_zh": "意大利麵", "main_carb_en": "Spaghetti Pasta", "protein_zh": "慢燉牛肉醬", "protein_en": "Slow-Cooked Beef Bolognese"},
            {"dish_id": "D03", "name_zh": "燒味飯", "name_en": "Siu Mei Rice", "main_carb_zh": "白米飯", "main_carb_en": "Steamed Jasmine Rice", "protein_zh": "燒味", "protein_en": "Roasted BBQ Meat"},
            {"dish_id": "D04", "name_zh": "干炒牛河", "name_en": "Fried Beef Noodles", "main_carb_zh": "中式麵條", "main_carb_en": "Flat Rice Noodles", "protein_zh": "牛肉", "protein_en": "Tender Sliced Beef"}
        ])

    if "name_zh" not in df.columns and "name" in df.columns:
        df["name_zh"] = df["name"].apply(lambda x: strip_bilingual_parentheses(x, is_en=False))
        df["name_en"] = df["name"].apply(lambda x: strip_bilingual_parentheses(x, is_en=True))
    if "main_carb_zh" not in df.columns and "main_carb" in df.columns:
        df["main_carb_zh"] = df["main_carb"].apply(lambda x: strip_bilingual_parentheses(x, is_en=False))
        df["main_carb_en"] = df["main_carb"].apply(lambda x: strip_bilingual_parentheses(x, is_en=True))
    if "protein_zh" not in df.columns and "protein" in df.columns:
        df["protein_zh"] = df["protein"].apply(lambda x: strip_bilingual_parentheses(x, is_en=False))
        df["protein_en"] = df["protein"].apply(lambda x: strip_bilingual_parentheses(x, is_en=True))

    df["name"] = df["name_en"] if is_en else df["name_zh"]
    df["main_carb"] = df["main_carb_en"] if is_en else df["main_carb_zh"]
    df["protein"] = df["protein_en"] if is_en else df["protein_zh"]
    return df

def get_live_branches(is_en=False):
    conn = db_conn()
    try:
        df = pd.read_sql_query("SELECT * FROM branches", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    
    if df.empty and os.path.exists(BRANCH_FILE):
        try:
            df = pd.read_csv(BRANCH_FILE, encoding="utf-8-sig")
        except Exception:
            try:
                df = pd.read_csv(BRANCH_FILE, encoding="utf-8")
            except Exception:
                pass

    if df.empty:
        df = pd.DataFrame([
            {"branch_id": "B01", "name_zh": "中環威靈頓街店", "name_en": "Wellington Street Branch", "level_zh": "Level A (商業核心區)", "level_en": "Level A (CBD Core)", "district_zh": "中西區", "district_en": "Central & Western", "traffic_zh": "白領上班族為主，午市尖峰翻檯率極高", "traffic_en": "Corporate white-collars; peak lunch turnover", "avg_covers": 1200, "base_rice_g": 240, "strategy_zh": "白領控醣需求高，建議推動小份量出餐與少飯扣減優惠。", "strategy_en": "High low-carb demand; promote smaller servings and Less Rice discounts."},
            {"branch_id": "B02", "name_zh": "沙田新城市廣場店", "name_en": "New Town Plaza Branch", "level_zh": "Level B (住宅商場)", "level_en": "Level B (Residential Mall)", "district_zh": "沙田區", "district_en": "Sha Tin", "traffic_zh": "家庭客、長者與週末休閒客群", "traffic_en": "Families, seniors, and weekend leisure diners", "avg_covers": 1500, "base_rice_g": 260, "strategy_zh": "家庭用餐與兒童共享比例高，建議推動彈性配菜組合。", "strategy_en": "High family sharing ratio; promote flexible side dish combos."},
            {"branch_id": "B03", "name_zh": "香港科技大學店", "name_en": "HKUST Campus Branch", "level_zh": "Level C (校園青年區)", "level_en": "Level C (University Campus)", "district_zh": "西貢區", "district_en": "Sai Kung", "traffic_zh": "學生、教職員，運動量及食量顯著較大", "traffic_en": "Students and faculty with active appetite", "avg_covers": 1800, "base_rice_g": 280, "strategy_zh": "維持大份量以維持飽足感，重點監控肉類醬汁口味與炸物品質。", "strategy_en": "Maintain large portions; monitor protein tenderness and gravy quality."},
            {"branch_id": "B04", "name_zh": "將軍澳 Popcorn 店", "name_en": "PopCorn Mall Branch", "level_zh": "Level B (住宅商場)", "level_en": "Level B (Residential Mall)", "district_zh": "西貢區", "district_en": "Sai Kung", "traffic_zh": "家庭客及換乘鐵路客流", "traffic_en": "Families and transit commuters", "avg_covers": 1400, "base_rice_g": 260, "strategy_zh": "重點監控主食及焗烤類餐點份量。", "strategy_en": "Focus on portion control for rice and baked pasta entrees."}
        ])

    if "name_zh" not in df.columns and "name" in df.columns:
        df["name_zh"] = df["name"].apply(lambda x: strip_bilingual_parentheses(x, is_en=False))
        df["name_en"] = df["name"].apply(lambda x: strip_bilingual_parentheses(x, is_en=True))
    if "level_zh" not in df.columns and "level" in df.columns:
        df["level_zh"] = df["level"].apply(lambda x: strip_bilingual_parentheses(x, is_en=False))
        df["level_en"] = df["level"].apply(lambda x: strip_bilingual_parentheses(x, is_en=True))

    df["name"] = df["name_en"] if is_en else df["name_zh"]
    df["level"] = df["level_en"] if is_en else df["level_zh"]
    df["district"] = df["district_en"] if is_en else df["district_zh"]
    df["traffic"] = df["traffic_en"] if is_en else df["traffic_zh"]
    df["strategy"] = df["strategy_en"] if is_en else df["strategy_zh"]
    return df

def get_live_rewards(is_en=False):
    conn = db_conn()
    try:
        df = pd.read_sql_query("SELECT * FROM rewards WHERE is_active = 1 ORDER BY max_waste_ratio ASC", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()

    if df.empty and os.path.exists(REWARD_FILE):
        try:
            df = pd.read_csv(REWARD_FILE, encoding="utf-8-sig")
        except Exception:
            try:
                df = pd.read_csv(REWARD_FILE, encoding="utf-8")
            except Exception:
                pass

    if df.empty:
        df = pd.DataFrame([
            {"reward_id": "R01", "tier_name_zh": "極致光盤獎", "tier_name_en": "Ultra Clean Plate", "max_waste_ratio": 10.0, "reward_type": "Coupon + Points", "reward_description_zh": "【$3 堂食現金券】+【50 綠色積分】+【凍檸茶半價券】", "reward_description_en": "【$3 Cash Voucher】+【50 Green Points】+【Half-Price Iced Lemon Tea】", "is_active": True},
            {"reward_id": "R02", "tier_name_zh": "達標惜食獎", "tier_name_en": "Standard Clean Plate", "max_waste_ratio": 20.0, "reward_type": "Coupon", "reward_description_zh": "【$2 堂食電子券】+【20 綠色積分】", "reward_description_en": "【$2 Dining Voucher】+【20 Green Points】", "is_active": True},
            {"reward_id": "R03", "tier_name_zh": "支持環保獎", "tier_name_en": "Green Support", "max_waste_ratio": 100.0, "reward_type": "Points", "reward_description_zh": "【10 綠色環保積分】", "reward_description_en": "【10 Green Points】", "is_active": True}
        ])

    df["tier_name"] = df["tier_name_en"] if is_en else df["tier_name_zh"]
    df["reward_description"] = df["reward_description_en"] if is_en else df["reward_description_zh"]
    return df

def evaluate_customer_rewards(waste_ratio_pct, is_en=False):
    df_rew = get_live_rewards(is_en=is_en)
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
            ts_str, date_str, month_str,
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
        get_records.clear()
    except Exception:
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
            df_new.to_csv(SEED_AUDIT_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
    except Exception:
        pass

@st.cache_data(ttl=60)
def get_records():
    try:
        conn = db_conn()
        df = pd.read_sql_query("SELECT * FROM audit_logs ORDER BY id DESC", conn)
        conn.close()
        if not df.empty:
            return df
    except Exception:
        pass
        
    if os.path.exists(SEED_AUDIT_FILE):
        try:
            return pd.read_csv(SEED_AUDIT_FILE, encoding="utf-8-sig")
        except Exception:
            return pd.read_csv(SEED_AUDIT_FILE, encoding="utf-8")
    return pd.DataFrame()

# ==============================================================================
# 3. Model Engine Loaders & Inference (with Food Verification Gatekeeper)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_vision_engine():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
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
    return {"processor": processor, "model": model, "device": dev}

@st.cache_resource(show_spinner=False)
def load_clip_engine():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = 0 if dev == "cuda" else -1
    try:
        return pipeline(
            "zero-shot-image-classification", 
            model="openai/clip-vit-base-patch32", 
            device=device_id
        )
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_nlp_engine():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = 0 if dev == "cuda" else -1
    p2_model_id = "kktlau115/trayzero-flant5-sop-alert"
    
    if dev == "cpu":
        try:
            torch.set_num_threads(4)
        except Exception:
            pass

    try:
        tokenizer = AutoTokenizer.from_pretrained(p2_model_id, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            p2_model_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32
        )
        if dev == "cpu":
            try:
                import torch.quantization
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
            except Exception:
                pass
        model.to(dev)
        model.eval()
        return pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device_id
        )
    except Exception:
        try:
            return pipeline("text2text-generation", model=p2_model_id, device=device_id)
        except Exception:
            try:
                return pipeline("text2text-generation", model="google/flan-t5-base", device=device_id)
            except Exception:
                return None

@st.cache_resource(show_spinner=False)
def load_ai_engine():
    vision = load_vision_engine()
    return {"processor": vision["processor"], "model": vision["model"], "device": vision["device"], "clip": None, "nlp": None}

def verify_food_tray_image(image, engine):
    """
    Quality gatekeeper: Verifies whether the image contains a food plate or meal tray
    versus a selfie, person's face, or non-food background before executing Pipeline 1.
    """
    if image is None:
        return False, "EMPTY_IMAGE", 0.0

    clip = engine.get("clip")
    if clip is None:
        clip = load_clip_engine()
        engine["clip"] = clip

    if clip is None:
        return True, "FOOD_TRAY", 1.0

    try:
        candidate_labels = [
            "a photo of a dining tray, meal, plate of food or fast-food dish",
            "a portrait photo of a person, human face, selfie or clothing",
            "a photo of an empty room, furniture, document, pet or non-food item"
        ]
        results = clip(image.convert("RGB"), candidate_labels=candidate_labels)
        top_label = results[0]["label"]
        top_score = float(results[0]["score"])

        if "selfie" in top_label and top_score >= 0.38:
            return False, "SELFIE_DETECTED", top_score
        elif "non-food" in top_label and top_score >= 0.45:
            return False, "NON_FOOD_DETECTED", top_score
        else:
            return True, "FOOD_CONFIRMED", top_score
    except Exception:
        return True, "FALLBACK_PASS", 1.0

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
    cat_v = "Vegetables & Sauce" if is_en else "蔬菜配菜"

    items = [
        {"Category": cat_c, "Confidence": f"{top_score:.1%}", "Coverage": f"{carb_ratio*100:.1f}%", "raw_ratio": carb_ratio},
        {"Category": cat_p, "Confidence": f"{top_score:.1%}", "Coverage": f"{protein_ratio*100:.1f}%", "raw_ratio": protein_ratio},
        {"Category": cat_v, "Confidence": f"{top_score:.1%}", "Coverage": f"{veg_ratio*100:.1f}%", "raw_ratio": veg_ratio},
    ]

    box = [int(width * 0.15), int(height * 0.15), int(width * 0.85), int(height * 0.85)]
    draw.rectangle(box, outline=accent_color, width=4)
    draw.text((box[0] + 10, box[1] + 10), f"TrayZero+: {ratio*100:.1f}%", fill=accent_color)
    
    return img_draw, items, ratio, primary_cat, True

def auto_detect_dish_clip(image, df_dishes, engine, is_en=False):
    if df_dishes.empty or image is None:
        return ("Undetected Dish" if is_en else "未定義餐點"), 0.0
        
    candidate_labels_en = df_dishes["name_en"].tolist()
    
    clip = engine.get("clip")
    if clip is None:
        clip = load_clip_engine()
        engine["clip"] = clip

    try:
        if clip is not None:
            results = clip(image.convert("RGB"), candidate_labels=candidate_labels_en)
            best_match_en = results[0]["label"]
            score = results[0]["score"]
            matched_row = df_dishes[df_dishes["name_en"] == best_match_en]
            if not matched_row.empty:
                chosen_name = matched_row.iloc[0]["name_en"] if is_en else matched_row.iloc[0]["name_zh"]
            else:
                chosen_name = best_match_en
            return chosen_name, score
        else:
            first_row = df_dishes.iloc[0]
            return (first_row["name_en"] if is_en else first_row["name_zh"]), 0.85
    except Exception:
        first_row = df_dishes.iloc[0]
        return (first_row["name_en"] if is_en else first_row["name_zh"]), 0.75

# ==============================================================================
# 4. Mode 1: Frontline Smart Tray Return Kiosk
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

    df_b = get_live_branches(is_en=is_en)
    df_d = get_live_dishes(is_en=is_en)
    
    branch_names = df_b["name"].tolist() if not df_b.empty else [("New Town Plaza Branch" if is_en else "沙田新城市廣場店")]
    dish_names = df_d["name"].tolist() if not df_d.empty else [("Baked Pork Chop Rice" if is_en else "一哥焗豬扒飯")]

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

        is_food_valid = True
        if image is not None:
            is_food_valid, check_reason, check_conf = verify_food_tray_image(image, engine)
            if not is_food_valid:
                if check_reason == "SELFIE_DETECTED":
                    warn_msg = (
                        f"⚠️ Notice: Human selfie or portrait detected ({check_conf:.1%}). "
                        "Please point the camera directly at a meal tray."
                    ) if is_en else (
                        f"⚠️ 系統提示：檢測到人像自拍或人物面部照片 (置信度: {check_conf:.1%})，非有效餐盤食材。"
                        "請對準大家樂餐盤重新拍攝。"
                    )
                else:
                    warn_msg = (
                        f"⚠️ Notice: Non-food object or background detected ({check_conf:.1%}). "
                        "Please place a valid food tray to proceed."
                    ) if is_en else (
                        f"⚠️ 系統提示：未檢測到有效餐盤或食材（置信度: {check_conf:.1%})。"
                        "請放置盛有餐點的大家樂餐盤後重試。"
                    )
                st.warning(warn_msg)

        if modules.get("mod3", True):
            with st.container(border=True):
                st.markdown(f"#### {'3. Dish Recognition & Confirmation' if is_en else '3. 菜式辨識與確認'}")
                if image and is_food_valid:
                    auto_dish, dish_conf = auto_detect_dish_clip(image, df_d, engine, is_en=is_en)
                    st.info(f"{'🔍 CLIP AI Recommended Dish' if is_en else '🔍 CLIP 智慧推薦菜式'}：**{auto_dish}** ({'Confidence' if is_en else '置信度'}: {dish_conf:.1%})")
                    sel_dish = st.selectbox("Confirm Target Dish" if is_en else "確認當前餐點菜式", dish_names, index=dish_names.index(auto_dish) if auto_dish in dish_names else 0)
                else:
                    sel_dish = st.selectbox("Select Target Dish" if is_en else "選擇餐點菜式", dish_names, index=0)
        else:
            sel_dish = dish_names[0] if dish_names else ("Baked Pork Chop Rice" if is_en else "一哥焗豬扒飯")

    with col_k2:
        if modules.get("mod1", True):
            with st.container(border=True):
                st.markdown(f"#### {'4. Deep Learning Audit & Instant Feedback' if is_en else '4. 深度學習審計與即時回饋'}")

                if image is not None and is_food_valid:
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
                            st.metric("🥦 " + ("Vegetables & Sauce" if is_en else "蔬菜配菜"), items[2]["Coverage"])
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
                elif image is not None and not is_food_valid:
                    st.info(
                        "ℹ️ Standby: Pipeline 1 vision processing suspended. Please upload a meal tray image to activate waste auditing."
                        if is_en else
                        "ℹ️ 待機中：視覺分析已暫停。請上傳包含有效餐點之大家樂餐盤照片以啟動審計。"
                    )

# ==============================================================================
# 5. Mode 2: Operations HQ & ESG Analytics Dashboard
# ==============================================================================
def render_mode2(engine, modules, is_en=False):
    st.markdown(f"### 📊 {'Operations HQ & ESG BI Analytics Dashboard' if is_en else '總部營運與 ESG 大數據儀表板'}")
    
    df_raw = get_records()
    if df_raw.empty:
        st.info("No audit records available." if is_en else "無可顯示之審計紀錄。")
        return

    df_b = get_live_branches(is_en=is_en)
    df_d = get_live_dishes(is_en=is_en)

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

    df_filtered["branch_name"] = df_filtered["branch_name"].apply(lambda x: strip_bilingual_parentheses(x, is_en=is_en))
    df_filtered["dish_name"] = df_filtered["dish_name"].apply(lambda x: strip_bilingual_parentheses(x, is_en=is_en))

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
        elif "Month" in period_filter or "按月" in period_filter or "本月" in period_filter:
            this_month_str = now.strftime("%Y-%m")
            df_filtered = df_filtered[df_filtered["audit_month"] == this_month_str]
        elif "Year" in period_filter or "按年" in period_filter or "本年" in period_filter:
            this_year_str = str(now.year)
            df_filtered = df_filtered[df_filtered["audit_year"] == this_year_str]

        if sel_b != "ALL":
            df_filtered = df_filtered[df_filtered["branch_name"] == sel_b]
        if sel_d != "ALL":
            df_filtered = df_filtered[df_filtered["dish_name"] == sel_d]

    n = len(df_filtered)
    if n == 0:
        st.info("No records matched the selected filters." if is_en else "所選篩選條件下無符合之紀錄。")
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

        dish_agg = df_filtered.groupby("dish_name").agg(
            waste_ratio=("waste_ratio", "mean"),
            tray_count=("waste_ratio", "count"),
            cost_waste=("cost_waste_hkd" if "cost_waste_hkd" in df_filtered.columns else "estimated_cost_hkd", "sum"),
            primary_waste=("primary_waste", lambda x: x.mode()[0] if (hasattr(x, "mode") and not x.empty) else ("Carbohydrates" if is_en else "主食澱粉"))
        ).reset_index().sort_values(by="waste_ratio", ascending=False)

        if dish_agg.empty:
            st.info("ℹ️ " + ("No audit records available for analysis." if is_en else "無相應餐點數據可供分析。"))
            return

        st.markdown(f"##### 🍲 {'Grouped Dish Analysis Matrix' if is_en else '按菜式分類聚合分析矩陣'} ({len(dish_agg)} {'Core Dishes' if is_en else '款主力菜式'})")

        dish_list = dish_agg["dish_name"].tolist()
        c_sel_dish, c_dish_kpi = st.columns([2, 1])
        with c_sel_dish:
            selected_agg_dish = st.selectbox(
                "Select Grouped Core Dish for Pipeline 2 Analysis" if is_en else "選擇欲讓 Pipeline 2 進行大數據深度診斷的聚合菜式：",
                dish_list,
                index=0
            )

        curr_row = dish_agg[dish_agg["dish_name"] == selected_agg_dish].iloc[0]
        curr_dish = curr_row["dish_name"]
        curr_waste = float(curr_row["waste_ratio"])
        curr_count = int(curr_row["tray_count"])
        curr_comp = str(curr_row["primary_waste"])
        curr_cost = float(curr_row["cost_waste"])

        with c_dish_kpi:
            st.metric(
                label=f"【{curr_dish}】" + ("Avg Waste Ratio" if is_en else "平均殘食率"),
                value=f"{curr_waste:.1f}%",
                delta=f"{curr_count:,} " + ("Trays" if is_en else "盤樣本"),
                delta_color="inverse"
            )

        with st.container(border=True):
            st.markdown(f"#### 🏢 {'Target Store' if is_en else '門市分析對象'}：**{target_store_name}** ｜ {'Target Dish' if is_en else '聚合診斷餐點'}：**{curr_dish}**")
            st.markdown(f"• {'Aggregated Waste Ratio' if is_en else '大數據聚合殘食率'}：**{curr_waste:.1f}%** ｜ {'Grouped Sample Trays' if is_en else '菜式樣本總盤數'}：**{curr_count:,} 盤**")
            st.markdown(f"• {'Primary Waste Component' if is_en else '主要損耗食材分類'}：**{curr_comp}** ｜ {'Total Waste Cost Loss' if is_en else '累計食材損耗金額'}：**HK$ {curr_cost:,.1f}**")

            prompt_hash = f"{target_store_name}_{curr_dish}_{round(curr_waste, 1)}"
            if "nlp_cache" not in st.session_state:
                st.session_state["nlp_cache"] = {}

            if prompt_hash in st.session_state["nlp_cache"]:
                raw_output = st.session_state["nlp_cache"][prompt_hash]
            else:
                with st.spinner("🤖 Pipeline 2 (Flan-T5) 正在進行大數據聚合特徵深度推理..." if not is_en else "🤖 Pipeline 2 (Flan-T5) inferring operational directives..."):
                    raw_prompt = (
                        f"Generate kitchen SOP and kiosk strategy for Café de Coral: "
                        f"Branch: {target_store_name}, "
                        f"Dish: {curr_dish}, "
                        f"Waste: {curr_waste:.1f}%, "
                        f"Primary Component: {curr_comp}, "
                        f"Occurrences: {curr_count}."
                    )
                    try:
                        nlp = engine.get("nlp")
                        if nlp is None:
                            nlp = load_nlp_engine()
                            engine["nlp"] = nlp
                        if nlp is not None:
                            out = nlp(
                                raw_prompt, 
                                max_new_tokens=45,
                                num_beams=1,
                                do_sample=False,
                                early_stopping=True
                            )
                            raw_output = out[0]["generated_text"]
                        else:
                            raw_output = f"[Kitchen BOH SOP] Residuals within normal tolerance ({curr_waste:.1f}%). [Frontline Strategy] Standard ordering prompts; reward 20 Green Points."
                    except Exception:
                        raw_output = f"[Kitchen BOH SOP] Residuals within normal tolerance ({curr_waste:.1f}%). [Frontline Strategy] Standard ordering prompts; reward 20 Green Points."
                    st.session_state["nlp_cache"][prompt_hash] = raw_output

            if "[Frontline Strategy]" in raw_output:
                parts = raw_output.split("[Frontline Strategy]")
                ai_boh = parts[0].strip()
                ai_foh = parts[1].strip()
            else:
                ai_boh = raw_output.strip()
                ai_foh = "Standard ordering prompts; reward 20 Green Points."

            ai_boh_clean = ai_boh.replace("[Kitchen BOH SOP]", "").replace("[Kitchen BOH SOP Alert]", "").replace("[CRITICAL BOH Alert]", "").strip()
            ai_foh_clean = ai_foh.replace("[Frontline Strategy]", "").strip()

            if "豬" in curr_dish or "Pork" in curr_dish:
                carb_zh, prot_zh, veg_zh = "白米飯 (蛋炒飯底 280g)", "焗厚切豬扒", "鮮茄醬汁與洋蔥"
                carb_en, prot_en, veg_en = "Egg Fried Rice (Standard 280g)", "Baked Thick-Cut Pork Chop", "Tomato Gravy & Onion"
            elif "意粉" in curr_dish or "Spaghetti" in curr_dish:
                carb_zh, prot_zh, veg_zh = "意大利麵 (標準 260g)", "慢燉牛肉醬", "洋蔥與香草配料"
                carb_en, prot_en, veg_en = "Spaghetti Pasta (Standard 260g)", "Slow-cooked Beef Bolognese", "Onion & Herb Garnish"
            elif "牛河" in curr_dish or "Noodles" in curr_dish:
                carb_zh, prot_zh, veg_zh = "河粉 (標準 250g)", "牛肉片", "芽菜與韭黃"
                carb_en, prot_en, veg_en = "Flat Rice Noodles (Standard 250g)", "Tender Beef Slices", "Bean Sprouts & Chives"
            elif "燒味" in curr_dish or "Siu Mei" in curr_dish:
                carb_zh, prot_zh, veg_zh = "白米飯 (標準 260g)", "燒味肉類 (叉燒/油雞)", "伴碟青菜與薑蓉"
                carb_en, prot_en, veg_en = "Steamed Jasmine Rice (260g)", "Roasted Meats (Char Siu / Chicken)", "Veggie Garnish & Scallion Oil"
            else:
                carb_zh, prot_zh, veg_zh = "主食澱粉", "蛋白質肉類", "蔬菜配菜醬汁"
                carb_en, prot_en, veg_en = "Carbohydrates", "Protein", "Vegetables & Sauce"

            is_critical = "[CRITICAL" in ai_boh or curr_waste >= 65.0
            is_alert = "Alert" in ai_boh or (curr_waste >= 25.0 and not is_critical)

            if is_critical:
                tag_badge_zh = '<span style="background-color: #DC2626; color: white; padding: 2px 7px; border-radius: 4px; font-weight: 700; font-size: 0.8rem;">緊急損耗告警</span>'
                tag_badge_en = '<span style="background-color: #DC2626; color: white; padding: 2px 7px; border-radius: 4px; font-weight: 700; font-size: 0.8rem;">CRITICAL BOH ALERT</span>'
                
                carb_act_zh = "強制扣減標準出餐量 20%–25%（如：標準出餐量扣減至 220g），杜絕過度給飯與無效澱粉損耗。"
                carb_act_en = "Enforce a mandatory 20%–25% reduction in standard carb serving size (down to 220g) to eliminate excessive starch waste."
                
                prot_act_zh = "必須立即核查入貨批次品質、解凍及醃製 SOP，確認是否有肉質過韌、過柴或調味過淡問題導致顧客拒食。"
                prot_act_en = "Immediately inspect raw material batches, thawing, and marination SOPs for texture defects, toughness, or flavor inconsistencies."
                
                veg_act_zh = "嚴格限定 1 標準勺給料，避免醬汁過多導致餐盤賣相凌亂或掩蓋主食風味。"
                veg_act_en = "Strictly enforce portion limits (exactly 1 standard ladle) to prevent over-saucing and presentation breakdown."
                
                kiosk_zh = f"點餐機已自動將【{curr_dish}】預設份量改為「少飯/少麵」，並以醒目標示「<b>響應環保，少飯即減 HK$ 2</b>」；主動光盤顧客發放最高級別【<b>HK$ 3 堂食現金券 + 50 綠色積分</b>】。"
                kiosk_en = f"Kiosks have overridden default portion to 'Less Rice' with an active '<b>Go Green: HK$ 2 Cash Discount</b>' prompt; zero-waste diners earn <b>HK$ 3 Cash Voucher + 50 Green Points</b>."
                
                sav_hkd = max(4500.0, curr_cost * 18.0)
                sav_co2 = max(3.2, curr_waste * 0.065)
            elif is_alert:
                tag_badge_zh = '<span style="background-color: #D97706; color: white; padding: 2px 7px; border-radius: 4px; font-weight: 700; font-size: 0.8rem;">中度損耗優化</span>'
                tag_badge_en = '<span style="background-color: #D97706; color: white; padding: 2px 7px; border-radius: 4px; font-weight: 700; font-size: 0.8rem;">BOH SOP ALERT</span>'
                
                carb_act_zh = "建議將打餐標準量微調下調 10%–15%，精準平衡顧客飽足感與食材損耗。"
                carb_act_en = "Calibrate standard carb portion downward by 10%–15% to balance customer satiety with cost control."
                
                prot_act_zh = "雖非最大浪費源，但高成本食材殘留反映顧客可能對口感不滿。建議檢視烹調火候與嫩度，避免因過硬遭棄置。"
                prot_act_en = "High-cost ingredient residuals suggest texture issues. Review oven baking duration and meat tenderness to prevent rejection."
                
                veg_act_zh = "微調醬汁或配料比例，確保黃金賣相與口味濃淡適中。"
                veg_act_en = "Recalibrate sauce and garnish ratio to ensure optimal meal presentation and flavor balance."
                
                kiosk_zh = f"點餐機與 Club 100 App 已動態加入「<b>少飯/少麵扣減 HK$ 1 現金</b>」輕度推薦；達成光盤顧客即時發放【<b>HK$ 2 堂食現金券 + 20 綠色積分</b>】。"
                kiosk_en = f"Kiosks and Club 100 App have enabled a soft '<b>Less Rice - HK$ 1 Discount</b>' prompt; zero-waste diners receive a <b>HK$ 2 Cash Voucher + 20 Green Points</b>."
                
                sav_hkd = max(2100.0, curr_cost * 8.5)
                sav_co2 = max(1.5, curr_waste * 0.038)
            else:
                tag_badge_zh = '<span style="background-color: #10B981; color: white; padding: 2px 7px; border-radius: 4px; font-weight: 700; font-size: 0.8rem;">常規標準維持</span>'
                tag_badge_en = '<span style="background-color: #10B981; color: white; padding: 2px 7px; border-radius: 4px; font-weight: 700; font-size: 0.8rem;">OPTIMAL BOH SOP</span>'
                
                carb_act_zh = "主食消耗率極佳，現有食譜份量黃金平衡，要求廚房繼續嚴格執行當前標準打餐量，無需進行份量扣減。"
                carb_act_en = "Carbohydrate consumption is optimal. Maintain current recipe standards and portioning without reductions."
                
                prot_act_zh = "肉類口感與調味獲顧客高度認可，維持現有標準解凍、醃製與烹調程序。"
                prot_act_en = "Protein acceptance is high. Maintain existing standard thawing, marination, and cooking procedures."
                
                veg_act_zh = "配菜與醬汁比例得宜，繼續保持現有出品標準。"
                veg_act_en = "Garnish and sauce proportions are well-received; maintain consistent kitchen standards."
                
                kiosk_zh = "點餐機維持標準出餐設定，<b>不主動推送「少飯扣減」優惠</b>以保障門市平均客單價；光盤顧客自動獲發【<b>50 綠色積分</b>】常規環保獎勵。"
                kiosk_en = "Kiosks maintain standard portion settings with <b>no proactive Less Rice discount</b> to protect average transaction value; zero-waste diners receive <b>50 Green Points</b>."
                
                sav_hkd = 850.0
                sav_co2 = 0.65

            if is_en:
                card_html = (
                    '<div style="background: #FFFFFF; border: 2px solid #2563EB; border-radius: 12px; padding: 20px 24px; margin-top: 14px; color: #0F172A; box-shadow: 0 4px 14px rgba(37,99,235,0.08);">'
                    f'<h4 style="color: #1E40AF; margin-top: 0; margin-bottom: 12px; font-weight: 800;">📋 [Executive Operations Notice] Store Kitchen & FOH Improvement SOP Directive</h4>'
                    f'<div style="font-size: 0.95rem; line-height: 1.65; color: #1E293B;">'
                    f'<p style="margin-bottom: 12px;"><b>I. BOH Kitchen Production & Portioning SOP:</b><br/>'
                    f'Monitored aggregated waste ratio of <b>{curr_waste:.1f}%</b> on <b>{curr_dish}</b> in {target_store_name} ({curr_count:,} trays audited).<br/>'
                    f'• <b>🍳 AI Core Directive</b>: {tag_badge_en} <i>"{ai_boh_clean}"</i><br/>'
                    f'• <b>🍚 {carb_en} (Carbs)</b>: {carb_act_en}<br/>'
                    f'• <b>🥩 {prot_en} (Protein)</b>: {prot_act_en}<br/>'
                    f'• <b>🥦 {veg_en} (Garnish/Sauce)</b>: {veg_act_en}</p>'
                    f'<p style="margin-bottom: 12px;"><b>II. Frontline Kiosk Reverse-POS & Member Incentive Strategy:</b><br/>'
                    f'• <b>📱 AI Kiosk Directive</b>: <i>"{ai_foh_clean}"</i><br/>'
                    f'• <b>Implementation Detail</b>: {kiosk_en}</p>'
                    f'<p style="margin-bottom: 12px;"><b>III. Financial Feasibility & Scope 3 ESG Forecast:</b><br/>'
                    f'This intervention is projected to reduce avoidable kitchen prep waste, saving approximately <b>HK$ {sav_hkd:,.0f} / month</b> in direct food costs, and cutting annual Scope 3 emissions by <b>{sav_co2:.2f} tonnes CO2e</b>, fully aligning with HKEX ESG Reporting Guidelines.</p>'
                    f'<div style="font-size: 0.8rem; color: #64748B; border-top: 1px solid #E2E8F0; padding-top: 9px; margin-top: 12px;">'
                    f'<i>AI Model: Hugging Face <code>kktlau115/trayzero-flant5-sop-alert</code> (Fine-tuned Flan-T5-Base) · Raw Token Output: {raw_output}</i>'
                    f'</div>'
                    f'</div>'
                    f'</div>'
                )
            else:
                card_html = (
                    '<div style="background: #FFFFFF; border: 2px solid #2563EB; border-radius: 12px; padding: 20px 24px; margin-top: 14px; color: #0F172A; box-shadow: 0 4px 14px rgba(37,99,235,0.08);">'
                    f'<h4 style="color: #1E40AF; margin-top: 0; margin-bottom: 12px; font-weight: 800;">📋 【大家樂總部運營通報】門市廚房與前廳改進 SOP 決策</h4>'
                    f'<div style="font-size: 0.95rem; line-height: 1.65; color: #1E293B;">'
                    f'<p style="margin-bottom: 12px;"><b>一、 後廚生產與備料調整 SOP：</b><br/>'
                    f'監測到【{target_store_name}】之【{curr_dish}】大數據聚合殘食率為 <b>{curr_waste:.1f}%</b>（樣本盤數：{curr_count:,} 盤）。<br/>'
                    f'• <b>🍳 AI 核心後廚指令</b>：{tag_badge_zh} <i>「{ai_boh_clean}」</i><br/>'
                    f'• <b>🍚 {carb_zh} (主食)</b>：{carb_act_zh}<br/>'
                    f'• <b>🥩 {prot_zh} (蛋白質)</b>：{prot_act_zh}<br/>'
                    f'• <b>🥦 {veg_zh} (配菜醬汁)</b>：{veg_act_zh}</p>'
                    f'<p style="margin-bottom: 12px;"><b>二、 前廳自助點餐機與會員策略：</b><br/>'
                    f'• <b>📱 AI 點餐機策略</b>：<i>「{ai_foh_clean}」</i><br/>'
                    f'• <b>落實執行細則</b>：{kiosk_zh}</p>'
                    f'<p style="margin-bottom: 12px;"><b>三、 門市營運效益與 ESG 減碳預期：</b><br/>'
                    f'預估本項 AI 營運決策介入後，可有效降低廚房過度備料損耗，預計每月節省食材成本約 <b>HK$ {sav_hkd:,.0f}</b>，每年累計減少 Scope 3 廚餘碳排放約 <b>{sav_co2:.2f} 噸 CO2e</b>，快速止損並全面符合香港交易所 (HKEX) ESG 最佳披露準則。</p>'
                    f'<div style="font-size: 0.8rem; color: #64748B; border-top: 1px solid #E2E8F0; padding-top: 9px; margin-top: 12px;">'
                    f'<i>AI 模型基礎：Hugging Face <code>kktlau115/trayzero-flant5-sop-alert</code> (Flan-T5-Base 微調) · 原始模型輸出: {raw_output}</i>'
                    f'</div>'
                    f'</div>'
                    f'</div>'
                )
            st.markdown(card_html, unsafe_allow_html=True)

    if modules.get("mod2", True):
        with st.container(border=True):
            st.markdown(f"#### 📋 {'Live Audit Records Table' if is_en else '即時審計明細數據表'}")
            st.dataframe(df_filtered, use_container_width=True)

# ==============================================================================
# 6. Mode 3: Master Data Configuration (Separated Chinese / English UI)
# ==============================================================================
def render_mode3(modules=None, is_en=False):
    st.markdown(f"### ⚙️ {'Dynamic Master Configuration (Dishes, Branches & Rewards)' if is_en else '菜單、分店與獎勵規則動態管理'}")
    
    tab_titles = []
    tab_keys = []
    
    if modules and modules.get("mod3", True):
        tab_titles.append("🍛 Menu Configuration" if is_en else "🍛 菜單設定")
        tab_keys.append("menu")
        
    if modules and modules.get("mod1", True):
        tab_titles.append("🏪 Store Configuration" if is_en else "🏪 門市設定")
        tab_keys.append("branch")
        
    if modules and modules.get("mod4", True):
        tab_titles.append("🎁 Reward Rules" if is_en else "🎁 獎勵規則")
        tab_keys.append("reward")
        
    if not tab_titles:
        st.info("All configuration modules are currently offline." if is_en else "所有配置模組目前皆已停用，無可用設定。")
        return
        
    tabs = st.tabs(tab_titles)
    
    for i, key in enumerate(tab_keys):
        with tabs[i]:
            if key == "menu":
                with st.container(border=True):
                    st.markdown(f"#### {'Curated Dishes Database' if is_en else '當前生效菜式總庫'}")
                    df_dishes = get_live_dishes(is_en=is_en)
                    
                    if is_en:
                        display_df = df_dishes[["dish_id", "name_en", "main_carb_en", "protein_en"]].rename(columns={
                            "dish_id": "Dish ID",
                            "name_en": "Dish Name",
                            "main_carb_en": "Main Carbohydrate",
                            "protein_en": "Main Protein Component"
                        })
                    else:
                        display_df = df_dishes[["dish_id", "name_zh", "main_carb_zh", "protein_zh"]].rename(columns={
                            "dish_id": "菜式編號",
                            "name_zh": "菜式名稱",
                            "main_carb_zh": "主食澱粉",
                            "protein_zh": "主要蛋白質"
                        })
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                    with st.expander("➕ " + ("Add or Update Dish Item" if is_en else "新增或更新菜式項目")):
                        c_d1, c_d2 = st.columns(2)
                        with c_d1:
                            in_id = st.text_input("Dish ID (e.g. D09)" if is_en else "菜式編號 (例如 D09)", key="add_dish_id")
                            in_name_zh = st.text_input("Chinese Name" if is_en else "中文名稱 (例如 咖喱牛腩飯)", key="add_dish_nzh")
                            in_carb_zh = st.text_input("Main Carb (ZH)" if is_en else "主食澱粉 (中文，例如 白米飯)", key="add_dish_czh")
                            in_prot_zh = st.text_input("Main Protein (ZH)" if is_en else "主要蛋白質 (中文，例如 慢燉牛腩)", key="add_dish_pzh")
                        with c_d2:
                            st.write("")
                            in_name_en = st.text_input("English Name" if is_en else "英文名稱 (例如 Curry Beef Brisket Rice)", key="add_dish_nen")
                            in_carb_en = st.text_input("Main Carb (EN)" if is_en else "主食澱粉 (英文，例如 Steamed Jasmine Rice)", key="add_dish_cen")
                            in_prot_en = st.text_input("Main Protein (EN)" if is_en else "主要蛋白質 (英文，例如 Braised Beef Brisket)", key="add_dish_pen")
                            
                        if st.button("💾 " + ("Save Dish to Database" if is_en else "保存菜式至資料庫"), type="primary", key="btn_save_dish"):
                            if in_id and in_name_zh:
                                conn = db_conn()
                                cur = conn.cursor()
                                cur.execute("""
                                    INSERT OR REPLACE INTO dishes (dish_id, name_zh, name_en, main_carb_zh, main_carb_en, protein_zh, protein_en)
                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                """, (in_id.strip(), in_name_zh.strip(), in_name_en.strip() or in_name_zh.strip(), 
                                      in_carb_zh.strip() or "白米飯", in_carb_en.strip() or "Steamed Rice", 
                                      in_prot_zh.strip() or "肉類", in_prot_en.strip() or "Meat"))
                                conn.commit()
                                conn.close()
                                get_live_dishes.clear()
                                st.success("Dish saved successfully!" if is_en else "菜式已成功更新至資料庫！")
                                st.rerun()
                            else:
                                st.error("Please fill in Dish ID and Chinese Name." if is_en else "請填妥菜式編號與名稱。")

            elif key == "branch":
                with st.container(border=True):
                    st.markdown(f"#### {'Curated Store Network Database' if is_en else '當前門市營運分店總庫'}")
                    df_branches = get_live_branches(is_en=is_en)
                    
                    if is_en:
                        display_df = df_branches[["branch_id", "name_en", "level_en", "district_en", "avg_covers", "base_rice_g"]].rename(columns={
                            "branch_id": "Branch ID",
                            "name_en": "Branch Name",
                            "level_en": "Store Tier",
                            "district_en": "District",
                            "avg_covers": "Daily Covers",
                            "base_rice_g": "Standard Rice (g)"
                        })
                    else:
                        display_df = df_branches[["branch_id", "name_zh", "level_zh", "district_zh", "avg_covers", "base_rice_g"]].rename(columns={
                            "branch_id": "分店代號",
                            "name_zh": "分店名稱",
                            "level_zh": "門市等級",
                            "district_zh": "所屬區域",
                            "avg_covers": "日均來客數",
                            "base_rice_g": "標準出餐飯量 (g)"
                        })
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                    with st.expander("➕ " + ("Add or Update Store Branch" if is_en else "新增或更新門市分店")):
                        c_b1, c_b2 = st.columns(2)
                        with c_b1:
                            in_bid = st.text_input("Branch ID (e.g. B05)" if is_en else "分店代號 (例如 B05)", key="add_b_id")
                            in_bname_zh = st.text_input("Branch Name (ZH)" if is_en else "分店名稱 (中文，例如 旺角亞皆老街店)", key="add_b_nzh")
                            in_blevel_zh = st.selectbox("Tier (ZH)" if is_en else "門市等級 (中文)", ["Level A (商業核心區)", "Level B (住宅商場)", "Level C (校園青年區)"], key="add_b_lzh")
                            in_bdist_zh = st.text_input("District (ZH)" if is_en else "行政區域 (中文，例如 油尖旺區)", key="add_b_dzh")
                        with c_b2:
                            in_bcovers = st.number_input("Daily Covers" if is_en else "預估日均來客數", min_value=100, max_value=5000, value=1500, step=100, key="add_b_cov")
                            in_bname_en = st.text_input("Branch Name (EN)" if is_en else "分店名稱 (英文，例如 Mong Kok Branch)", key="add_b_nen")
                            in_blevel_en = st.selectbox("Tier (EN)" if is_en else "門市等級 (英文)", ["Level A (CBD Core)", "Level B (Residential Mall)", "Level C (University Campus)"], key="add_b_len")
                            in_bdist_en = st.text_input("District (EN)" if is_en else "行政區域 (英文，例如 Yau Tsim Mong)", key="add_b_den")

                        if st.button("💾 " + ("Save Branch to Database" if is_en else "保存分店至資料庫"), type="primary", key="btn_save_branch"):
                            if in_bid and in_bname_zh:
                                conn = db_conn()
                                cur = conn.cursor()
                                cur.execute("""
                                    INSERT OR REPLACE INTO branches (branch_id, name_zh, name_en, level_zh, level_en, district_zh, district_en, traffic_zh, traffic_en, avg_covers, base_rice_g, strategy_zh, strategy_en)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (in_bid.strip(), in_bname_zh.strip(), in_bname_en.strip() or in_bname_zh.strip(),
                                      in_blevel_zh, in_blevel_en, in_bdist_zh.strip() or "市區", in_bdist_en.strip() or "Urban",
                                      "人流密集", "High traffic", int(in_bcovers), 260, "標準出餐", "Standard operations"))
                                conn.commit()
                                conn.close()
                                get_live_branches.clear()
                                st.success("Store branch saved successfully!" if is_en else "分店資料已成功更新！")
                                st.rerun()
                            else:
                                st.error("Please fill in Branch ID and Name." if is_en else "請填妥分店代號與名稱。")

            elif key == "reward":
                with st.container(border=True):
                    st.markdown(f"#### {'Curated Member Rewards Rules' if is_en else '當前會員光盤激勵獎勵矩陣'}")
                    df_rewards = get_live_rewards(is_en=is_en)
                    
                    if is_en:
                        display_df = df_rewards[["reward_id", "tier_name_en", "max_waste_ratio", "reward_type", "reward_description_en"]].rename(columns={
                            "reward_id": "Reward ID",
                            "tier_name_en": "Reward Tier",
                            "max_waste_ratio": "Max Residual (%)",
                            "reward_type": "Benefit Type",
                            "reward_description_en": "Reward Details"
                        })
                    else:
                        display_df = df_rewards[["reward_id", "tier_name_zh", "max_waste_ratio", "reward_type", "reward_description_zh"]].rename(columns={
                            "reward_id": "獎勵編號",
                            "tier_name_zh": "獎勵等級",
                            "max_waste_ratio": "最高殘食率 (%)",
                            "reward_type": "權益類型",
                            "reward_description_zh": "獎勵說明"
                        })
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ==============================================================================
# 7. Main Application Entrypoint
# ==============================================================================
def main():
    inject_safe_css()
    init_db()

    if os.path.exists(LOGO_FILE_PNG):
        st.sidebar.image(LOGO_FILE_PNG, use_container_width=True)
    elif os.path.exists(LOGO_FILE_JPG):
        st.sidebar.image(LOGO_FILE_JPG, use_container_width=True)
    else:
        st.sidebar.markdown("## 🍽️ Café de Coral")

    lang_choice = st.sidebar.radio("🌐 語言 / Language", ["繁體中文 (Traditional Chinese)", "English (英語)"], horizontal=True)
    is_en = "English" in lang_choice

    st.sidebar.markdown(f"### **{'TrayZero+ Console' if is_en else 'TrayZero+ 控制台'}**")
    st.sidebar.markdown(f"**{'Café de Coral · Smart Plate Audit' if is_en else '大家樂集團 · 智能餐盤審計'}**")
    
    mode_container = st.sidebar.container()

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"##### {'ENTERPRISE MODULES' if is_en else '企業模組狀態'}")
    
    mod_1 = st.sidebar.checkbox("M1: Ops Core" if is_en else "M1: 營運監控", value=True)
    mod_2 = st.sidebar.checkbox("M2: BI Analytics" if is_en else "M2: 深度分析", value=True)
    mod_3 = st.sidebar.checkbox("M3: Smart POS" if is_en else "M3: 精準營銷", value=True)
    mod_4 = st.sidebar.checkbox("M4: Loyalty Loop" if is_en else "M4: 會員閉環", value=True)

    active_modules = {
        "mod1": mod_1,
        "mod2": mod_2,
        "mod3": mod_3,
        "mod4": mod_4
    }

    mode_options = [
        "Frontline Return Kiosk" if is_en else "門市前線收盤機"
    ]
    if mod_2:
        mode_options.append("Operations HQ BI Analytics" if is_en else "總部 BI 大數據看板")
    
    mode_options.append("Dynamic Master Configuration" if is_en else "菜單與獎勵配置")

    mode_container.markdown("---")
    mode = mode_container.radio("System Operation Mode" if is_en else "系統運行模式", mode_options)

    with st.spinner("🚀 Loading Edge Vision AI Engine (<2s)..." if is_en else "🚀 正在快速啟動輕量邊緣視覺 AI 引擎 (Swin-Tiny 邊緣加載中)..."):
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
