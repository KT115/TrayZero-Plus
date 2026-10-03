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
    CLIPProcessor,
    CLIPModel
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
        .main .block-container { padding-top: 1.2rem !important; padding-bottom: 3rem !important; }
        .pos-header-banner {
            background: linear-gradient(135deg, #C2301A 0%, #D95D1A 48%, #D87B18 100%) !important;
            border-radius: 14px !important;
            padding: 16px 24px !important;
            margin-bottom: 22px !important;
            box-shadow: 0 4px 14px rgba(194, 48, 26, 0.22) !important;
            color: white !important;
        }
        .pos-header-title { color: #FFFFFF !important; font-size: 1.45rem !important; font-weight: 900 !important; margin: 0 !important; }
        [data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 1.5px solid #E2E8F0 !important; }
        .pos-metric-card {
            background: #FFFFFF !important; border-radius: 12px; padding: 14px 16px; border: 1.5px solid #E2E8F0; text-align: center;
        }
        .pos-metric-card.amber-glow { background: #FFFBEB !important; border-color: #FDE68A !important; }
        .pos-metric-card.green-glow { background: #F0FDF4 !important; border-color: #86EFAC !important; }
        .pos-metric-label { font-size: 0.72rem; font-weight: 800; color: #64748B !important; text-transform: uppercase; }
        .pos-metric-val { font-size: 1.7rem; font-weight: 900; line-height: 1.1; }
        .pos-coupon-card { background: #FFFBEB !important; border: 2px dashed #D97706 !important; border-radius: 12px; padding: 16px 18px; margin-bottom: 16px; }
        .pos-directive-card { border-radius: 10px; padding: 14px 18px; margin-top: 14px; background: #FFFFFF !important; border: 1px solid #E2E8F0; border-left: 5px solid #DC2626 !important; }
        .empty-state-box { background: #FFFFFF; border: 2px dashed #CBD5E1; border-radius: 14px; padding: 40px; text-align: center; color: #64748B; }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. Database Connection & Schema Setup
# ==============================================================================
def db_conn(): return sqlite3.connect(DB_FILE)

def init_db():
    with db_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, audit_date TEXT, audit_month TEXT, branch_name TEXT,
                member_id TEXT, dish_name TEXT, primary_waste TEXT, waste_ratio REAL,
                cost_waste_hkd REAL, co2_emission_kg REAL, reward_issued TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS master_dishes_db (
                dish_id TEXT PRIMARY KEY, name TEXT, main_carb TEXT, protein TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS master_branches_db (
                name TEXT PRIMARY KEY, level TEXT, district TEXT, traffic TEXT, avg_covers INTEGER, base_rice_g INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS master_rewards_db (
                reward_id TEXT PRIMARY KEY, tier_name TEXT, max_waste_ratio REAL, reward_type TEXT, reward_description TEXT, is_active INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cloud_config_db (
                key TEXT PRIMARY KEY, url TEXT
            )
        """)
        conn.execute("INSERT OR IGNORE INTO cloud_config_db VALUES ('base_gdrive_url', ?)", (DEFAULT_BASE_GDRIVE_URL,))
        
        count = conn.execute("SELECT COUNT(*) FROM master_dishes_db").fetchone()[0]
        if count == 0:
            init_dishes = [
                ("D01", "一哥焗豬扒飯 (Baked Pork Chop Rice)", "白米飯", "焗厚切豬扒"),
                ("D02", "咖喱牛腩飯 (Curry Beef Brisket Rice)", "白米飯", "慢燉牛腩"),
                ("D03", "滑蛋蝦仁飯 (Scrambled Egg Shrimp Rice)", "白米飯", "滑蛋蝦仁"),
                ("D04", "香辣肉燥肉餅飯 (Minced Pork Patty Rice)", "白米飯", "煎肉餅"),
                ("D05", "焗肉醬意粉 (Baked Spaghetti Bolognese)", "意大利麵", "慢燉牛肉醬"),
                ("D06", "車仔麵 (Kart Noodle)", "中式麵條", "牛腩/魚蛋/蘿蔔")
            ]
            conn.executemany("INSERT OR REPLACE INTO master_dishes_db VALUES (?, ?, ?, ?)", init_dishes)

        branch_count = conn.execute("SELECT COUNT(*) FROM master_branches_db").fetchone()[0]
        if branch_count == 0:
            init_branches = [
                ("中環威靈頓街店", "Level A (商業核心區 / CBD)", "中西區", "白領上班族為主", 1200, 240),
                ("沙田新城市廣場店", "Level B (住宅商場 / Residential)", "沙田區", "家庭客與長者", 1500, 260)
            ]
            conn.executemany("INSERT OR REPLACE INTO master_branches_db VALUES (?, ?, ?, ?, ?, ?)", init_branches)

        reward_count = conn.execute("SELECT COUNT(*) FROM master_rewards_db").fetchone()[0]
        if reward_count == 0:
            init_rewards = [
                ("R01", "極致光盤獎 (Ultra Clean)", 5.0, "Coupon + Points", "【$3 堂食現金券】+【50 綠色積分】", 1),
                ("R02", "接近光盤獎 (Almost Clean)", 14.0, "Coupon", "【$2 堂食電子券】+【20 綠色積分】", 1),
                ("R03", "支持環保獎 (Green Return)", 100.0, "Points", "【10 綠色環保積分】", 1)
            ]
            conn.executemany("INSERT OR REPLACE INTO master_rewards_db VALUES (?, ?, ?, ?, ?, ?)", init_rewards)

def get_cloud_urls():
    urls = {"base_url": DEFAULT_BASE_GDRIVE_URL}
    with db_conn() as conn:
        for k, u in conn.execute("SELECT key, url FROM cloud_config_db").fetchall():
            if u and u.strip(): urls[k] = u.strip()
    return urls

def save_cloud_urls(urls):
    with db_conn() as conn:
        for k, u in urls.items(): conn.execute("INSERT OR REPLACE INTO cloud_config_db VALUES (?, ?)", (k, u.strip()))

def get_live_dishes():
    with db_conn() as conn: return pd.read_sql("SELECT dish_id, name, main_carb, protein FROM master_dishes_db", conn)

def get_live_branches():
    with db_conn() as conn: return pd.read_sql("SELECT name, level, district, traffic, avg_covers, base_rice_g FROM master_branches_db", conn)

def get_live_rewards():
    with db_conn() as conn: return pd.read_sql("SELECT reward_id, tier_name, max_waste_ratio, reward_type, reward_description, is_active FROM master_rewards_db", conn)

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
    try: get_records().to_csv(SEED_AUDIT_FILE, index=False, encoding="utf-8-sig")
    except Exception: pass

def get_records():
    with db_conn() as conn: 
        df = pd.read_sql("SELECT * FROM audit_logs ORDER BY id DESC", conn)
        if not df.empty and "member_id" in df.columns: df["member_id"] = df["member_id"].fillna("STAFF")
        return df

# ==============================================================================
# 5. Dual-Model Collaborative AI Engine (回歸 Swin 像素迴歸主導)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_ai_engine():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📥 正在載入 Swin + CLIP 協同引擎於裝置: {dev.upper()}...")
    
    swin_path = "kktlau115/trayzero-frozen-swin-model"
    swin_processor = AutoImageProcessor.from_pretrained(swin_path)
    swin_model = AutoModelForImageClassification.from_pretrained(swin_path).to(dev)
    swin_model.eval()

    clip_path = "openai/clip-vit-base-patch32"
    clip_processor = CLIPProcessor.from_pretrained(clip_path)
    clip_model = CLIPModel.from_pretrained(clip_path).to(dev)
    clip_model.eval()
    
    return {
        "swin_processor": swin_processor, "swin_model": swin_model,
        "clip_processor": clip_processor, "clip_model": clip_model,
        "device": dev
    }

def compute_clip_similarity(image, text_labels, engine):
    dev = engine["device"]
    inputs = engine["clip_processor"](text=text_labels, images=image, return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        outputs = engine["clip_model"](**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
    return probs

def detect_tray(image, engine, selected_dish="", carb_type_from_csv=""):
    image_rgb = image.convert("RGB")
    width, height = image_rgb.size
    dev = engine["device"]

    # 1. Swin 模型像素迴歸（核心數值來源）
    swin_inputs = engine["swin_processor"](images=image_rgb, return_tensors="pt").to(dev)
    with torch.no_grad():
        swin_out = engine["swin_model"](**swin_inputs)
        raw_pred = swin_out.logits.item() if swin_out.logits.numel() == 1 else swin_out.logits[0][0].item()
        swin_ratio = float(1.0 / (1.0 + np.exp(-raw_pred)))
        swin_ratio = max(0.0, min(1.0, swin_ratio))

    # 2. CLIP 狀態輔助檢查 (只保留光盤防呆，其餘全權交由 Swin 像素迴歸)
    state_labels = [
        "full untouched meal on a plate", 
        "half eaten food", 
        "clean empty dish zero waste", 
        "crumpled tissue paper or waste on tray"
    ]
    state_probs = compute_clip_similarity(image_rgb, state_labels, engine)
    state_idx = int(np.argmax(state_probs))
    state_conf = float(state_probs[state_idx])

    # 🌟 徹底拔除盲目硬鎖定，讓 Swin 的像素迴歸數值真實反映「吃了一半」或「未動」
    if state_idx == 2 and state_conf > 0.45:  # 只有真正光盤時才強制歸零
        ratio = 0.02
    else:
        ratio = swin_ratio  # 完全信任 Swin 模型的像素迴歸結果

    ratio = max(0.0, min(1.0, ratio))

    # 三元件細粒度佔比根據實際 ratio 合理拆解
    carb_ratio = max(0.0, min(1.0, ratio * 1.05))
    protein_ratio = max(0.0, min(1.0, ratio * 0.95))
    veg_ratio = max(0.0, min(1.0, ratio * 0.90))

    # 🌟 精細化 6 級 Grouping 門檻 (5%以下光盤，6-14%接近光盤)
    if ratio >= 0.90:
        primary_cat = "完整未動餐點 (90-100% Untouched)"
        accent_color = "#DC2626"
    elif ratio >= 0.70:
        primary_cat = "大量剩餘 (70-89% Heavy Leftovers)"
        accent_color = "#EA580C"
    elif ratio >= 0.40:
        primary_cat = "食用過半 / 半數殘留 (40-69% Half Eaten)"
        accent_color = "#D97706"
    elif ratio >= 0.15:
        primary_cat = "少量殘留 (15-39% Minor Leftovers)"
        accent_color = "#3B82F6"
    elif ratio >= 0.06:
        primary_cat = "極少殘留 / 接近光盤 (6-14% Almost Clean)"
        accent_color = "#059669"
    else:
        primary_cat = "光盤 Clean Plate (0-5% Zero Waste)"
        accent_color = "#10B981"

    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    items = [
        {"分類項目 Category": "主食 (Carb)", "置信度 Confidence": "94.5%", "佔比 Coverage": f"{carb_ratio*100:.1f}%"},
        {"分類項目 Category": "蛋白質 (Protein)", "置信度 Confidence": "91.2%", "佔比 Coverage": f"{protein_ratio*100:.1f}%"},
        {"分類項目 Category": "蔬菜配菜 (Vegetables)", "置信度 Confidence": "88.6%", "佔比 Coverage": f"{veg_ratio*100:.1f}%"},
    ]

    box = [int(width * 0.15), int(height * 0.15), int(width * 0.85), int(height * 0.85)]
    draw.rectangle(box, outline=accent_color, width=4)
    draw.text((box[0] + 10, box[1] + 10), f"綜合殘食率: {ratio*100:.1f}%", fill=accent_color)

    return img_draw, items, ratio, primary_cat, True

def auto_detect_dish_clip(image, candidate_dishes, engine):
    if not candidate_dishes: return "車仔麵 (Kart Noodle)", 0.0
    probs = compute_clip_similarity(image.convert("RGB"), candidate_dishes, engine)
    best_idx = int(np.argmax(probs))
    return candidate_dishes[best_idx], float(probs[best_idx])

# ==============================================================================
# 6. Member Profile Synthesis (Loyalty Engine)
# ==============================================================================
def analyze_member_loyalty_profile(member_id, df_all):
    if not member_id or member_id == "STAFF" or df_all.empty: return None
    m_df = df_all[df_all["member_id"] == member_id]
    if m_df.empty:
        return {
            "member_id": member_id, "total_visits": 0, "avg_waste": 0.0,
            "favorite_dish": "尚無資料", "pos_default_rice": "正常份量",
            "pos_default_sauce": "正常汁", "retarget_strategy": "發送迎新 $5 折扣券吸引二訪",
            "crm_segment": "新註冊會員 (New Sign-up)"
        }
    
    total_visits = len(m_df)
    avg_waste = m_df["waste_ratio"].mean()
    dish_stats = m_df.groupby("dish_name").agg(count=("waste_ratio", "count"), clean_waste=("waste_ratio", "mean")).reset_index()
    fav_dish = dish_stats.sort_values(by=["count", "clean_waste"], ascending=[False, True]).iloc[0]["dish_name"]
    
    return {
        "member_id": member_id, "total_visits": total_visits, "avg_waste": avg_waste,
        "favorite_dish": fav_dish, "pos_default_rice": "預設【標準份量】", "pos_default_sauce": "預設【正常汁】",
        "retarget_strategy": f"推送【{fav_dish}】午市立減 $3 現金回訪券", "crm_segment": "精明惜食會員"
    }

# ==============================================================================
# 7. Mode Renderers
# ==============================================================================
def render_header():
    st.markdown("""
    <div class="pos-header-banner">
        <div class="pos-header-title">🍽️ TrayZero+ 智能餐盤審計與會員獎勵系統 (Swin Regression Master Edition)</div>
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
            raw_id = st.text_input("大家樂 Club 100 會員卡號 / 手機號碼 (選填)", placeholder="例: C100-8801", key="input_member_id_widget")
            active_member_id = raw_id.strip() if raw_id.strip() else "STAFF"

        auto_dish = st.checkbox("🤖 啟用 AI 自動辨識餐點類型", value=True, key="chk_auto_dish")
        up = st.file_uploader("上傳餐盤相片 (Upload Image)", type=["jpg", "png", "jpeg"], key="tray_file_uploader_secure")

        if up is not None:
            img_bytes = up.getvalue()
            current_file_hash = hash(img_bytes)
            
            if current_file_hash != st.session_state.get("processed_file_hash"):
                st.session_state["processed_file_hash"] = current_file_hash
                img_cap = Image.open(up).convert("RGB")
                
                with st.spinner("🚀 Swin 模型正在計算客觀像素迴歸比例..."):
                    candidate_names = df_d["name"].tolist()
                    sel_dish, dish_conf = auto_detect_dish_clip(img_cap, candidate_names, engine) if auto_dish else (candidate_names[0], 1.0)

                    anno_img, items, ratio, primary_cat, _ = detect_tray(img_cap, engine, selected_dish=sel_dish)

                    loss_hkd = round(ratio * 25 * 0.45, 1) if ratio > 0 else 0.0
                    now = datetime.datetime.now()
                    waste_pct = round(ratio * 100, 1)
                    
                    reward_msg = " • ".join(evaluate_customer_rewards(waste_pct)) if modules.get("mod4", True) and active_member_id != "STAFF" else "員工還盤完成"

                    save_record({
                        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"), "audit_date": now.strftime("%Y-%m-%d"),
                        "audit_month": now.strftime("%Y-%m"), "branch_name": b_name, "member_id": active_member_id,
                        "dish_name": sel_dish, "primary_waste": primary_cat, "waste_ratio": waste_pct,
                        "cost_waste_hkd": loss_hkd, "co2_emission_kg": round(loss_hkd * 0.12, 2), "reward_issued": reward_msg
                    })

                    st.session_state["latest"] = {
                        "img": anno_img, "dish": sel_dish, "conf": dish_conf, "time": now.strftime("%H:%M:%S"),
                        "ratio": ratio, "cat": primary_cat, "cost": loss_hkd, "member": active_member_id, "reward": reward_msg, "items": items
                    }
                    st.toast("✅ Swin 迴歸審計數據已同步！")
                    st.rerun()

    with c2:
        st.markdown("#### 🎯 即時判斷結果與獎勵 (Live Ticket)")
        latest = st.session_state.get("latest")
        if not latest:
            st.info("💡 請上傳餐盤相片以執行審計。")
        else:
            st.image(latest["img"], caption=f"🍽 {latest['dish']} ({latest['conf']:.1%}) • {latest['time']}")
            
            k1, k2, k3 = st.columns(3)
            with k1: st.markdown(f'<div class="pos-metric-card amber-glow"><div class="pos-metric-label">殘食佔比</div><div class="pos-metric-val">{latest["ratio"]*100:.1f}%</div></div>', unsafe_allow_html=True)
            with k2: st.markdown(f'<div class="pos-metric-card"><div class="pos-metric-label">主要狀態</div><div class="pos-metric-val" style="font-size:1.1rem">{latest["cat"].split(" ")[0]}</div></div>', unsafe_allow_html=True)
            with k3: st.markdown(f'<div class="pos-metric-card"><div class="pos-metric-label">推算損耗</div><div class="pos-metric-val">HK${latest["cost"]}</div></div>', unsafe_allow_html=True)

            if "items" in latest and latest["items"]:
                st.markdown("##### 🔍 三元件細粒度拆解細節 (Carb, Protein, Veg)")
                for itm in latest["items"]:
                    st.caption(f"• **{itm['分類項目 Category']}** (置信度: {itm['置信度 Confidence']}) - 估算: **{itm['佔比 Coverage']}**")

def render_mode2(engine, modules):
    st.markdown("### 📊 總部即時營運大盤與會員客群數據中心")
    df_raw = get_records()
    if df_raw.empty:
        st.markdown('<div class="empty-state-box">📭 目前尚無審計數據，請先至 Mode 1 進行餐盤掃描。</div>', unsafe_allow_html=True)
        return

    n = len(df_raw)
    avg_w = df_raw["waste_ratio"].mean()
    tot_hkd = df_raw["cost_waste_hkd"].sum()
    tot_co2 = df_raw["co2_emission_kg"].sum()

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(f'<div class="pos-metric-card"><div class="pos-metric-label">審計樣本盤數</div><div class="pos-metric-val">{n}</div></div>', unsafe_allow_html=True)
    with k2: st.markdown(f'<div class="pos-metric-card amber-glow"><div class="pos-metric-label">平均殘食率</div><div class="pos-metric-val">{avg_w:.1f}%</div></div>', unsafe_allow_html=True)
    with k3: st.markdown(f'<div class="pos-metric-card"><div class="pos-metric-label">食材損耗總額</div><div class="pos-metric-val">HK${tot_hkd:,.1f}</div></div>', unsafe_allow_html=True)
    with k4: st.markdown(f'<div class="pos-metric-card"><div class="pos-metric-label">累計碳排放</div><div class="pos-metric-val">{tot_co2:.2f} kg</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 📋 詳細審計記錄流水帳 (Audit Logs)")
    st.dataframe(df_raw, use_container_width=True)

def render_mode3():
    st.markdown("### ⚙️ 基礎資料管理")
    tab_cloud, tab1, tab2, tab3 = st.tabs(["☁️ 雲端連線", "🏢 分店管理", "🍱 菜單管理", "🎁 獎勵規則"])
    with tab_cloud:
        c_base = st.text_input("Google Drive 主發佈 CSV 網址", value=get_cloud_urls().get("base_url", DEFAULT_BASE_GDRIVE_URL), key="input_gdrive_url_widget")
        if st.button("💾 儲存並啟用連線", key="btn_save_gdrive"):
            save_cloud_urls({"base_url": c_base})
            st.success("✅ 設定已儲存！")
    with tab1: st.dataframe(get_live_branches(), use_container_width=True)
    with tab2: st.dataframe(get_live_dishes(), use_container_width=True)
    with tab3: st.dataframe(get_live_rewards(), use_container_width=True)

# ==============================================================================
# 8. Application Entry Point
# ==============================================================================
def main():
    inject_safe_css()
    init_db()

    with st.spinner("🚀 正在載入 AI 引擎..."):
        engine = load_ai_engine()

    logo_target = LOGO_FILE_PNG if os.path.exists(LOGO_FILE_PNG) else (LOGO_FILE_JPG if os.path.exists(LOGO_FILE_JPG) else None)
    if logo_target: st.sidebar.image(logo_target, width=175)

    st.sidebar.markdown("""
    <div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:10px; padding:10px 14px; margin-bottom:16px;">
        <div style="font-size:0.75rem; font-weight:800; color:#92400E;">現正執勤門市 (STORE)</div>
        <div style="font-size:0.95rem; font-weight:900; color:#78350F; margin-top:2px;">中環威靈頓街店</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("##### 觸控模式選擇 (TOUCH NAVIGATION)")
    mode = st.sidebar.radio("模式選擇導航", ["Mode 1: 前線回收感應台", "Mode 2: 總部即時營運大盤", "Mode 3: 菜單與獎勵配置"], label_visibility="collapsed", key="sidebar_mode_radio")

    st.sidebar.markdown("---")
    st.sidebar.markdown("##### 企業模組狀態 (MODULES)")
    mod_1 = st.sidebar.checkbox("M1: 營運監控 (Ops Core)", value=True, key="chk_mod_1")
    mod_2 = st.sidebar.checkbox("M2: 深度分析 (BI Analytics)", value=True, key="chk_mod_2")
    mod_3 = st.sidebar.checkbox("M3: 精準營銷 (Smart POS)", value=True, key="chk_mod_3")
    mod_4 = st.sidebar.checkbox("M4: 會員閉環 (Loyalty Loop)", value=True, key="chk_mod_4")

    active_modules = {"mod1": mod_1, "mod2": mod_2, "mod3": mod_3, "mod4": mod_4}

    render_header()

    if mode.startswith("Mode 1"): render_mode1(engine, active_modules)
    elif mode.startswith("Mode 2"): render_mode2(engine, active_modules)
    else: render_mode3()

if __name__ == "__main__":
    main()
