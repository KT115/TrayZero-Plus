import os
import datetime
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import pipeline
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
LOGO_FILE_PNG = os.path.join(BASE_DIR, "CDC_810.png")
LOGO_FILE_JPG = os.path.join(BASE_DIR, "CDC_810.jpg")

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
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        }
        .main .block-container {
            padding-top: 1.2rem !important;
            padding-bottom: 3rem !important;
        }
        .pos-header-banner {
            background: linear-gradient(135deg, #C2301A 0%, #D95D1A 48%, #D87B18 100%) !important;
            border-radius: 14px !important;
            padding: 16px 24px !important;
            margin-bottom: 22px !important;
            box-shadow: 0 4px 14px rgba(194, 48, 26, 0.22) !important;
            color: white !important;
        }
        .pos-header-title {
            color: #FFFFFF !important;
            font-size: 1.45rem !important;
            font-weight: 900 !important;
            margin: 0 !important;
        }
        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1.5px solid #E2E8F0 !important;
        }
        .pos-metric-card {
            background: #FFFFFF !important;
            border-radius: 12px;
            padding: 14px 16px;
            border: 1.5px solid #E2E8F0;
            text-align: center;
        }
        .pos-metric-card.amber-glow { background: #FFFBEB !important; border-color: #FDE68A !important; }
        .pos-metric-card.green-glow { background: #F0FDF4 !important; border-color: #86EFAC !important; }
        .pos-metric-label { font-size: 0.72rem; font-weight: 800; color: #64748B !important; text-transform: uppercase; }
        .pos-metric-val { font-size: 1.7rem; font-weight: 900; line-height: 1.1; }
        .pos-coupon-card {
            background: #FFFBEB !important;
            border: 2px dashed #D97706 !important;
            border-radius: 12px;
            padding: 16px 18px;
            margin-bottom: 16px;
        }
        .pos-directive-card {
            border-radius: 10px;
            padding: 14px 18px;
            margin-top: 14px;
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0;
            border-left: 5px solid #DC2626 !important;
        }
        .empty-state-box {
            background: #FFFFFF;
            border: 2px dashed #CBD5E1;
            border-radius: 14px;
            padding: 40px;
            text-align: center;
            color: #64748B;
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
        
        init_dishes = [
            ("D01", "一哥焗豬扒飯 (Baked Pork Chop Rice)", "白米飯", "焗厚切豬扒"),
            ("D02", "咖喱牛腩飯 (Curry Beef Brisket Rice)", "白米飯", "慢燉牛腩"),
            ("D03", "滑蛋蝦仁飯 (Scrambled Egg Shrimp Rice)", "白米飯", "滑蛋蝦仁"),
            ("D04", "香辣肉燥肉餅飯 (Minced Pork Patty Rice)", "白米飯", "煎肉餅"),
            ("D05", "焗肉醬意粉 (Baked Spaghetti Bolognese)", "意大利麵", "慢燉牛肉醬"),
            ("D06", "車仔麵 (Kart Noodle)", "中式麵條", "牛腩/魚蛋/蘿蔔")
        ]
        conn.executemany("INSERT OR REPLACE INTO master_dishes_db VALUES (?, ?, ?, ?)", init_dishes)

        init_branches = [
            ("中環威靈頓街店", "Level A (商業核心區 / CBD)", "中西區", "白領上班族為主", 1200, 240),
            ("沙田新城市廣場店", "Level B (住宅商場 / Residential)", "沙田區", "家庭客與長者", 1500, 260)
        ]
        conn.executemany("INSERT OR REPLACE INTO master_branches_db VALUES (?, ?, ?, ?, ?, ?)", init_branches)

        init_rewards = [
            ("R01", "極致光盤獎 (Ultra Clean)", 10.0, "Coupon + Points", "【$3 堂食現金券】+【50 綠色積分】", 1),
            ("R02", "達標惜食獎 (Standard Clean)", 20.0, "Coupon", "【$2 堂食電子券】+【20 綠色積分】", 1),
            ("R03", "支持環保獎 (Green Return)", 100.0, "Points", "【10 綠色環保積分】", 1)
        ]
        conn.executemany("INSERT OR REPLACE INTO master_rewards_db VALUES (?, ?, ?, ?, ?, ?)", init_rewards)

def get_cloud_urls():
    urls = {"base_url": DEFAULT_BASE_GDRIVE_URL}
    with db_conn() as conn:
        rows = conn.execute("SELECT key, url FROM cloud_config_db").fetchall()
        for k, u in rows:
            if u and u.strip(): urls[k] = u.strip()
    return urls

def get_live_dishes():
    with db_conn() as conn:
        return pd.read_sql("SELECT dish_id, name, main_carb, protein FROM master_dishes_db", conn)

def get_live_branches():
    with db_conn() as conn:
        return pd.read_sql("SELECT name, level, district, traffic, avg_covers, base_rice_g FROM master_branches_db", conn)

def get_live_rewards():
    with db_conn() as conn:
        return pd.read_sql("SELECT reward_id, tier_name, max_waste_ratio, reward_type, reward_description, is_active FROM master_rewards_db", conn)

def evaluate_customer_rewards(waste_ratio_pct):
    df_r = get_live_rewards()
    if df_r.empty: return ["已累積 10 綠色環保積分"]
    active_rules = df_r[df_r["is_active"] == 1].copy()
    matched = active_rules[active_rules["max_waste_ratio"] >= waste_ratio_pct].sort_values(by="max_waste_ratio", ascending=True)
    if not matched.empty:
        best_tier = matched.iloc[0]
        return [f"🎉 【{best_tier['tier_name']}】已派發至大家樂 App：{best_tier['reward_description']}"]
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

def get_records():
    with db_conn() as conn: 
        return pd.read_sql("SELECT * FROM audit_logs ORDER BY id DESC", conn)

# ==============================================================================
# 5. High-Precision CLIP Zero-Shot Engine (取代失准的微調模型)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_ai_engine():
    dev = 0 if torch.cuda.is_available() else -1
    print("📥 正在載入高精度 CLIP 零樣本視覺語意引擎...")
    clip_classifier = pipeline(
        "zero-shot-image-classification", 
        model="openai/clip-vit-base-patch32", 
        device=dev
    )
    return {"clip": clip_classifier}

def detect_tray(image, engine, selected_dish="", carb_type_from_csv=""):
    image_rgb = image.convert("RGB")
    width, height = image_rgb.size
    
    # 🌟 使用精準的語意對比清單，直接識別大家樂餐盤狀態
    state_labels = [
        "a full untouched meal with rice and meat, perfectly clean plate with no food eaten",
        "half eaten meal with some leftover food on the plate",
        "completely empty clean dish after eating everything, zero waste",
        "heavy food waste with lots of unfinished rice and meat left on the plate"
    ]
    
    res = engine["clip"](image_rgb, candidate_labels=state_labels)
    top_label = res[0]["label"]
    
    # 根據語意匹配精準判定殘食率 (0.0 ~ 1.0)
    if "untouched" in top_label or "full" in top_label:
        ratio = 0.95  # 完整未動 (未食用浪費)
        primary_cat = "完整未動餐點 (未食用浪費)"
        accent_color = "#DC2626"
    elif "empty" in top_label or "zero waste" in top_label:
        ratio = 0.0   # 光盤
        primary_cat = "光盤 Clean Plate"
        accent_color = "#10B981"
    elif "half" in top_label:
        ratio = 0.5   # 吃了一半
        primary_cat = "主食與配料半數殘留"
        accent_color = "#D97706"
    else:
        ratio = 0.8   # 大量剩餘
        primary_cat = "大量主食與肉類浪費"
        accent_color = "#DC2626"

    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    items = []

    box = [int(width * 0.1), int(height * 0.1), int(width * 0.9), int(height * 0.9)]
    draw.rectangle(box, outline=accent_color, width=4)
    draw.text((box[0] + 10, box[1] + 10), f"TrayZero+ CLIP 智慧識別殘食率: {ratio*100:.1f}%", fill=accent_color)
    
    items.append({
        "分類項目 Category": primary_cat, 
        "置信度 Confidence": f"{res[0]['score']:.1%}", 
        "佔比 Coverage": f"{ratio*100:.1f}%"
    })

    return img_draw, items, ratio, primary_cat, True

def auto_detect_dish_clip(image, candidate_dishes, engine):
    if not candidate_dishes: return "未定義餐點", 0.0
    clean_labels = [d.strip() for d in candidate_dishes]
    try:
        results = engine["clip"](image.convert("RGB"), candidate_labels=clean_labels)
        return results[0]["label"], results[0]["score"]
    except Exception:
        return candidate_dishes[0], 0.75

def analyze_member_loyalty_profile(member_id, df_all):
    if not member_id or member_id == "STAFF" or df_all.empty: return None
    m_df = df_all[df_all["member_id"] == member_id]
    if m_df.empty: return None
    return {
        "total_visits": len(m_df),
        "avg_waste": m_df["waste_ratio"].mean(),
        "favorite_dish": m_df["dish_name"].mode()[0] if not m_df["dish_name"].empty else "一哥焗豬扒飯",
        "crm_segment": "精明惜食會員 (Green Member)"
    }

# ==============================================================================
# 6. Mode Renderers
# ==============================================================================
def render_header():
    st.markdown("""
    <div class="pos-header-banner">
        <div class="pos-header-title">🍽️ TrayZero+ 智能餐盤審計與會員獎勵系統 (High-Precision CLIP Powered)</div>
    </div>
    """, unsafe_allow_html=True)

def render_mode1(engine, modules):
    df_b = get_live_branches()
    df_d = get_live_dishes()
    if df_b.empty or df_d.empty:
        st.warning("⚠️ 門市或餐點清單為空！")
        return

    df_history = get_records()
    c1, c2 = st.columns([1.15, 0.85])
    
    with c1:
        st.markdown("#### 🏢 會員識別與還盤掃描 (Member Scan & Ingest)")
        b_name = st.selectbox("執勤門市 (Active Store Location)", df_b["name"].tolist())
        
        active_member_id = "STAFF"
        if modules.get("mod4", True):
            raw_member_id = st.text_input("掃描或輸入會員卡號 / 手機號碼 (選填)", placeholder="例: C100-8801")
            active_member_id = raw_member_id.strip() if raw_member_id.strip() else "STAFF"

        auto_dish = st.checkbox("🤖 啟用 AI 自動辨識餐點類型", value=True)
        up = st.file_uploader("上傳大家樂餐盤相片 (Upload Tray Image)", type=["jpg", "png", "jpeg"])

        if up is not None:
            img_cap = Image.open(up).convert("RGB")
            with st.spinner("🚀 TrayZero+ 正在進行高精度視覺審計..."):
                candidate_names = df_d["name"].tolist()
                sel_dish, dish_conf = auto_detect_dish_clip(img_cap, candidate_names, engine) if auto_dish else (candidate_names[0], 1.0)

                matched_rows = df_d[df_d["name"] == sel_dish]
                carb_type = matched_rows.iloc[0]["main_carb"] if not matched_rows.empty else "白米飯"

                anno_img, items, ratio, primary_cat, _ = detect_tray(img_cap, engine, selected_dish=sel_dish, carb_type_from_csv=carb_type)

                loss_hkd = round(ratio * 25 * 0.45, 1) if ratio > 0 else 0.0
                now = datetime.datetime.now()
                waste_pct = round(ratio * 100, 1)
                
                reward_msg = " • ".join(evaluate_customer_rewards(waste_pct)) if active_member_id != "STAFF" else "員工還盤完成"

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
                    "img": anno_img, "dish": sel_dish, "conf": dish_conf,
                    "time": now.strftime("%H:%M:%S"), "ratio": ratio, "cat": primary_cat, 
                    "cost": loss_hkd, "member": active_member_id, "reward": reward_msg
                }
                st.toast("✅ 還盤審計數據已即時寫入！")
                st.rerun()

    with c2:
        st.markdown("#### 🎯 即時判斷結果與獎勵 (Live Ticket)")
        latest = st.session_state.get("latest")
        if not latest:
            st.info("💡 請上傳大家樂餐盤相片以執行審計。")
        else:
            st.image(latest["img"], caption=f"🍽️ {latest['dish']} • {latest['time']}")
            
            k1, k2, k3 = st.columns(3)
            with k1:
                st.markdown(f'<div class="pos-metric-card amber-glow"><div class="pos-metric-label">殘食佔比</div><div class="pos-metric-val">{latest["ratio"]*100:.1f}%</div></div>', unsafe_allow_html=True)
            with k2:
                st.markdown(f'<div class="pos-metric-card"><div class="pos-metric-label">主要狀態</div><div class="pos-metric-val" style="font-size:1.1rem">{latest["cat"].split(" ")[0]}</div></div>', unsafe_allow_html=True)
            with k3:
                st.markdown(f'<div class="pos-metric-card"><div class="pos-metric-label">推算損耗</div><div class="pos-metric-val">HK${latest["cost"]}</div></div>', unsafe_allow_html=True)

def render_mode2(engine, modules):
    st.markdown("### 📊 總部即時營運大盤")
    df_raw = get_records()
    if df_raw.empty:
        st.markdown('<div class="empty-state-box">📭 尚無審計數據，請先至 Mode 1 執行掃描。</div>', unsafe_allow_html=True)
        return
    st.dataframe(df_raw, use_container_width=True)

def render_mode3():
    st.markdown("### ⚙️ 基礎資料管理")
    st.info("支援連動系統資料。")

def main():
    inject_safe_css()
    init_db()
    engine = load_ai_engine()

    logo_target = LOGO_FILE_PNG if os.path.exists(LOGO_FILE_PNG) else (LOGO_FILE_JPG if os.path.exists(LOGO_FILE_JPG) else None)
    if logo_target: st.sidebar.image(logo_target, width=175)

    mode = st.sidebar.radio("模式選擇", ["Mode 1: 前線回收感應台", "Mode 2: 總部即時營運大盤", "Mode 3: 資料管理"], label_visibility="collapsed")
    render_header()

    if mode.startswith("Mode 1"): render_mode1(engine, {"mod4": True})
    elif mode.startswith("Mode 2"): render_mode2(engine, {"mod4": True})
    else: render_mode3()

if __name__ == "__main__":
    main()
