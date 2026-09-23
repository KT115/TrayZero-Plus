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
    AutoModelForObjectDetection, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    pipeline
)

# ==============================================================================
# 0. Global File Paths & Persistence Binding
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BRANCH_FILE = os.path.join(BASE_DIR, "master_branches.csv")
DISH_FILE = os.path.join(BASE_DIR, "master_dishes.csv")
SEED_AUDIT_FILE = os.path.join(BASE_DIR, "seed_audit_logs.csv")
DB_FILE = os.path.join(BASE_DIR, "trayzero_audit.db")
LOGO_FILE_PNG = os.path.join(BASE_DIR, "CDC_810.png")
LOGO_FILE_JPG = os.path.join(BASE_DIR, "CDC_810.jpg")

FOOD_WHITELIST = {
    "bowl": "Carb", "cake": "Carb", "sandwich": "Meat", "pizza": "Meat", "hot dog": "Meat",
    "carrot": "Veg_Soup", "broccoli": "Veg_Soup", "apple": "Veg_Soup", "orange": "Veg_Soup",
    "donut": "Meat", "cup": "Veg_Soup", "bottle": "Veg_Soup", "dining table": "Tray"
}

# ==============================================================================
# 1. UI Styling & Typography
# ==============================================================================
def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Noto+Sans+TC:wght@400;500;700;900&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', 'Noto Sans TC', sans-serif;
        }

        .stApp {
            background-color: #F8FAFC !important;
            color: #0F172A;
        }

        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
        }

        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
            color: #334155 !important;
            font-weight: 500;
        }

        /* Sidebar Logo Centering */
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

        /* Minimalist Single-Line Header */
        .trayzero-header {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 18px 26px;
            margin-bottom: 22px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.03);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .trayzero-title {
            color: #0F172A !important;
            font-size: 1.25rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.01em;
            margin: 0 !important;
            white-space: nowrap !important;
        }

        /* Clean Micro-Elevated Cards */
        .clean-card {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 18px 20px;
            margin-bottom: 14px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 4px 12px -2px rgba(15, 23, 42, 0.03);
            transition: all 0.2s ease-in-out;
        }
        .clean-card:hover {
            border-color: #CBD5E1;
            box-shadow: 0 10px 25px -4px rgba(15, 23, 42, 0.06);
            transform: translateY(-2px);
        }

        .clean-label {
            font-size: 0.74rem;
            color: #64748B;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }

        .clean-val {
            font-size: 1.8rem;
            font-weight: 800;
            color: #0F172A;
            line-height: 1.1;
        }

        /* CRM Retargeting Cards */
        .crm-card {
            background: linear-gradient(135deg, #F0FDF4 0%, #FFFFFF 100%);
            border: 1.5px solid #86EFAC;
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 14px;
            box-shadow: 0 4px 12px rgba(34, 197, 94, 0.06);
        }

        .crm-title {
            font-size: 0.95rem;
            font-weight: 800;
            color: #166534;
            margin-bottom: 6px;
        }

        .member-live-badge {
            background: #EFF6FF;
            border: 1px solid #BFDBFE;
            border-radius: 12px;
            padding: 12px 16px;
            margin-bottom: 14px;
            font-size: 0.86rem;
            color: #1E40AF;
        }

        /* Directives */
        .directive-card {
            border-radius: 14px;
            padding: 16px 20px;
            margin-bottom: 12px;
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #CBD5E1;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.02);
        }
        .directive-chef { border-left-color: #EF4444; }
        .directive-pos { border-left-color: #10B981; }
        .directive-crm { border-left-color: #3B82F6; }

        .directive-title {
            font-weight: 700;
            font-size: 0.88rem;
            color: #0F172A;
            margin-bottom: 4px;
        }

        .directive-body {
            font-size: 0.84rem;
            color: #475569;
            line-height: 1.6;
        }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. Database Layer with Seed Persistence
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
        cols = [c[1] for c in conn.execute("PRAGMA table_info(audit_logs)").fetchall()]
        if "member_id" not in cols: 
            conn.execute("ALTER TABLE audit_logs ADD COLUMN member_id TEXT")
        if "reward_issued" not in cols: 
            conn.execute("ALTER TABLE audit_logs ADD COLUMN reward_issued TEXT")
        
        row_count = conn.execute("SELECT COUNT(*) FROM audit_logs").fetchone()[0]
        if row_count == 0 and os.path.exists(SEED_AUDIT_FILE):
            try:
                seed_df = pd.read_csv(SEED_AUDIT_FILE, encoding="utf-8-sig")
                seed_df.to_sql("audit_logs", conn, if_exists="append", index=False)
            except Exception:
                pass

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
    except Exception:
        pass

def get_records():
    with db_conn() as conn: 
        return pd.read_sql("SELECT * FROM audit_logs ORDER BY id DESC", conn)

def reset_db():
    with db_conn() as conn: 
        conn.execute("DELETE FROM audit_logs")
    if os.path.exists(SEED_AUDIT_FILE):
        try:
            os.remove(SEED_AUDIT_FILE)
        except Exception:
            pass

def load_master_data():
    if os.path.exists(BRANCH_FILE):
        df_b = pd.read_csv(BRANCH_FILE, encoding="utf-8-sig")
    else:
        df_b = pd.DataFrame(columns=["name", "district", "avg_covers", "base_rice_g"])

    if os.path.exists(DISH_FILE):
        df_d = pd.read_csv(DISH_FILE, encoding="utf-8-sig")
    else:
        df_d = pd.DataFrame(columns=["dish_id", "name", "main_carb", "protein"])

    return df_b, df_d

# ==============================================================================
# 3. AI Inference Engine
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_ai_engine():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m_path = os.path.join(BASE_DIR, "Fine-tuned_Model_files")
    if not (os.path.exists(m_path) and any(os.scandir(m_path))):
        m_path = "hustvl/yolos-tiny"
    
    proc = AutoImageProcessor.from_pretrained(m_path)
    det = AutoModelForObjectDetection.from_pretrained(m_path).to(dev)
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
        "device": dev
    }

def detect_tray(image, engine):
    inp = engine["proc"](images=image, return_tensors="pt").to(engine["device"])
    with torch.no_grad(): 
        out = engine["det"](**inp)
        
    sz = torch.tensor([image.size[::-1]]).to(engine["device"])
    res = engine["proc"].post_process_object_detection(out, threshold=0.20, target_sizes=sz)[0]
    
    img_draw = image.copy()
    total_area = image.size[0] * image.size[1]
    waste_area = 0
    draw = ImageDraw.Draw(img_draw)
    items, valid_food = [], False
    color_map = {"Carb": "#EF4444", "Meat": "#F59E0B", "Veg_Soup": "#10B981"}
    primary = "光盤 Clean Plate"

    for box, score, label_id in zip(res["boxes"].tolist(), res["scores"].tolist(), res["labels"].tolist()):
        lbl = engine["det"].config.id2label.get(label_id, "item")
        if lbl not in FOOD_WHITELIST: 
            continue
            
        cat = FOOD_WHITELIST[lbl]
        valid_food = True
        
        if cat == "Carb": 
            name, primary = "主食殘留 Carb Residual", "主食殘留 Carb Residual"
        elif cat == "Meat":
            name = "肉類殘留 Meat Residual"
            if "主食" not in primary: 
                primary = "肉類殘留 Protein Residual"
        elif cat == "Veg_Soup":
            name = f"配菜/醬汁 Sides ({lbl})"
            if primary == "光盤 Clean Plate": 
                primary = "配菜/醬汁 Sides & Sauce"
        else: 
            name = "餐盤基準 Tray Baseline"

        b = [max(0, box[0]), max(0, box[1]), min(image.size[0], box[2]), min(image.size[1], box[3])]
        area = (b[2] - b[0]) * (b[3] - b[1])
        if cat != "Tray": 
            waste_area += area

        c = color_map.get(cat, "#3B82F6")
        draw.rectangle(b, outline=c, width=3)
        draw.text((b[0] + 4, b[1] + 4), f"{name} {score:.0%}", fill=c)
        items.append({
            "分類項目 Category": name, 
            "置信度 Confidence": f"{score:.1%}", 
            "佔比 Coverage": f"{area/total_area:.1%}"
        })

    ratio = min(1.0, waste_area / (total_area * 0.65)) if (total_area > 0 and valid_food) else 0.0
    return img_draw, items, ratio, primary, valid_food

def auto_detect_dish_clip(image, candidate_dishes, engine):
    if not candidate_dishes:
        return "未定義餐點 Undefined Dish", 0.0
    clean_labels = [d.strip() for d in candidate_dishes]
    try:
        results = engine["clip"](image, candidate_labels=clean_labels)
        return results[0]["label"], results[0]["score"]
    except Exception:
        return candidate_dishes[0], 0.75

# ==============================================================================
# 4. 會員偏好演算與 CRM Retargeting 生成模組 (Core Loyalty Loop Engine)
# ==============================================================================
def analyze_member_loyalty_profile(member_id, df_all):
    """
    深入分析會員最近用餐過盤數據，推導其點餐偏好及大家樂 CRM Retargeting 行動方針
    """
    if not member_id or member_id == "GUEST" or df_all.empty:
        return None
    
    m_df = df_all[df_all["member_id"] == member_id]
    if m_df.empty:
        return {
            "member_id": member_id,
            "total_visits": 0,
            "avg_waste": 0.0,
            "favorite_dish": "尚無資料",
            "pos_default_rice": "正常飯量",
            "pos_default_sauce": "正常汁",
            "retarget_strategy": "發送通用迎新 $5 折扣券吸引二訪",
            "crm_segment": "新註冊會員 (New Sign-up)"
        }
    
    total_visits = len(m_df)
    avg_waste = m_df["waste_ratio"].mean()
    
    # 統計高頻點餐且吃得最乾淨的「最喜愛餐點」
    dish_stats = m_df.groupby("dish_name").agg(
        count=("waste_ratio", "count"),
        clean_waste=("waste_ratio", "mean")
    ).reset_index()
    fav_dish = dish_stats.sort_values(by=["count", "clean_waste"], ascending=[False, True]).iloc[0]["dish_name"]
    
    # 主食偏好推導 (若主食殘留頻繁 > 20% 則預設少飯)
    carb_wastes = m_df[m_df["primary_waste"].str.contains("主食", na=False)]
    if not carb_wastes.empty and carb_wastes["waste_ratio"].mean() > 20.0:
        pos_rice = "預設【少飯 (-30g / 立減 $2)】"
        crm_seg = "控醣輕食族 (Low-Carb Diners)"
    elif avg_waste < 10.0:
        pos_rice = "預設【正常飯量 (光盤常客)】"
        crm_seg = "高飽足飽腹族 (Standard/High-Calorie)"
    else:
        pos_rice = "預設【標準飯量】"
        crm_seg = "均衡飲食族 (Balanced Diners)"

    # 醬汁/配菜推導
    sauce_wastes = m_df[m_df["primary_waste"].str.contains("配菜|醬汁", na=False)]
    if not sauce_wastes.empty and sauce_wastes["waste_ratio"].mean() > 25.0:
        pos_sauce = "預設【少汁 / 醬汁另上】"
    else:
        pos_sauce = "預設【正常汁】"

    # 生成給大家樂 CRM 的 Retargeting 具體推廣策略
    if "控醣" in crm_seg:
        retarget_strategy = f"針對最愛餐點【{fav_dish}】推送「健康少飯換特飲優惠券」；推廣高蛋白低碳輕食套餐。"
    elif "高飽足" in crm_seg:
        retarget_strategy = f"向其大家樂 App 推送【{fav_dish}】加配小食（雞翼/紅豆冰）$8 組合券，拉高客單價。"
    else:
        retarget_strategy = f"推送【{fav_dish}】午市立減 $3 現金回訪券，鎖定工作日午市高頻復購。"

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
# 5. UI Views & Component Rendering
# ==============================================================================
def render_header(modules):
    active_badges = []
    if modules["mod1"]: active_badges.append("M1: 營運監控")
    if modules["mod2"]: active_badges.append("M2: 數據洞察")
    if modules["mod3"]: active_badges.append("M3: 精準營銷")
    if modules["mod4"]: active_badges.append("M4: 會員閉環")
    badge_str = " • ".join(active_badges)

    st.markdown(f"""
    <div class="trayzero-header">
        <div>
            <h2 class="trayzero-title">🍽️ TrayZero+ 大家樂智能餐盤審計與會員閉環平台</h2>
            <div style="font-size:0.75rem; color:#64748B; margin-top:3px; font-weight:600;">已授權模組 (Licensed Modules): {badge_str}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_mode1(df_b, df_d, engine, modules):
    if df_b.empty or df_d.empty:
        st.warning("⚠️ 門市或餐點清單為空！請確認 GitHub 倉庫根目錄已上傳 master_branches.csv 與 master_dishes.csv。\n(Store or menu database is empty. Please verify GitHub CSV files.)")
        return

    df_history = get_records()
    c1, c2 = st.columns([1.1, 0.9])
    
    with c1:
        st.markdown("#### 🏢 回收台設置與會員識別 (Station & Member Authentication)")
        b_name = st.selectbox("執勤門市 (Active Store Location)", df_b["name"].tolist())
        
        # ----------------------------------------------------------------------
        # 會員輸入與即時偏好分析 (Member Scan & Instant Preference)
        # ----------------------------------------------------------------------
        active_member_id = "GUEST"
        if modules["mod4"]:
            st.markdown("##### 📲 大家樂 Club 100 會員識別 (Member Scanner)")
            member_col1, member_col2 = st.columns([3, 1])
            with member_col1:
                raw_member_id = st.text_input(
                    "掃描或輸入會員卡號 (Scan/Enter Member ID)", 
                    value=st.session_state.get("last_input_member", ""),
                    placeholder="例: C100-8801 / 手機號碼",
                    help="支援條碼槍掃描讀入。留空或點選訪客將歸檔為 GUEST"
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
                else: 
                    st.info("🟢 監控中：當前餐盤已完成分析，等待更換餐盤...\n(Monitoring: Active tray already analyzed. Awaiting next tray...)")
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
            with st.spinner("🚀 AI 正在分析殘食並提取顧客偏好 (YOLOS + CLIP)..."):
                anno_img, items, ratio, primary_cat, is_food = detect_tray(img_cap, engine)

                if not is_food:
                    st.error("🚫 偵測失敗：未檢測到合法餐盤或食物物件！（已自動過濾人物/背景）\n(Detection Failed: No valid tray or food objects detected!)")
                    st.session_state["latest"] = None
                else:
                    candidate_names = df_d["name"].tolist()
                    if auto_dish:
                        sel_dish, dish_conf = auto_detect_dish_clip(img_cap, candidate_names, engine)
                    else:
                        sel_dish = candidate_names[0]
                        dish_conf = 1.0

                    loss_hkd = round(ratio * 25 * 0.45, 1)
                    now = datetime.datetime.now()
                    
                    # 獎勵及會員記錄標記
                    if modules["mod4"] and active_member_id != "GUEST":
                        if ratio < 0.15:
                            reward_msg = "🎉 達成光盤獎勵！已派發【$3 堂食優惠券 + 50 綠色積分】至大家樂 App"
                        else:
                            reward_msg = "已累積【10 綠色環保積分】至 Club 100 帳戶"
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
                        "waste_ratio": round(ratio * 100, 1),
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
                    st.toast(f"✅ 還盤數據已同步至大家樂 CRM！會員卡號【{active_member_id}】。")
                    st.rerun()

    with c2:
        st.markdown("#### 🎯 前線掃描結果與會員數據 (Audit Result & Member Loop)")
        latest = st.session_state.get("latest")
        if not latest:
            st.info("💡 尚未執行偵測或畫面非餐盤。請對準餐盤掃描。\n(No valid tray scan available.)")
        else:
            conf_str = f"({latest.get('conf', 1.0):.1%})" if 'conf' in latest else ""
            st.image(latest["img"], caption=f"🍽️ {latest['dish']} {conf_str} • {latest['time']}", use_container_width=True)
            
            if modules["mod4"] and latest.get("member") != "GUEST":
                # 即時展示個人偏好與 Retargeting 聯動卡片
                st.markdown(f"""
                <div class="crm-card">
                    <div class="crm-title">📲 大家樂 Loyalty Loop 反向偏好更新成功</div>
                    • <b>關聯會員 ID</b>: <code>{latest.get('member')}</code><br>
                    • <b>本次還盤結果</b>: {latest.get('reward')}<br>
                    • <b>自動反哺 POS 規則</b>: 下次點餐系統已預載顧客客製化偏好
                </div>
                """, unsafe_allow_html=True)

            k1, k2, k3 = st.columns(3)
            k1.markdown(f'<div class="clean-card"><div class="clean-label">殘食佔比 Waste Ratio</div><div class="clean-val" style="color:{"#EF4444" if latest["ratio"] > 0.3 else "#10B981"}">{latest["ratio"]:.1%}</div></div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="clean-card"><div class="clean-label">主要殘留 Primary Residual</div><div class="clean-val" style="font-size:1.05rem;margin-top:6px;">{latest["cat"].split(" ")[0]}</div></div>', unsafe_allow_html=True)
            k3.markdown(f'<div class="clean-card"><div class="clean-label">推算損耗 Loss (HK$)</div><div class="clean-val" style="color:#F59E0B">HK${latest["cost"]}</div></div>', unsafe_allow_html=True)

def render_mode2(df_b, df_d, engine, modules):
    st.markdown("### 📊 總部營運與會員客群 Retargeting 數據中心 (Executive HQ & Loyalty Engine)")
    df_raw = get_records()

    st.markdown("#### 🎛️ 多維度分析透視 (Analysis Dimensions)")
    c1, c2 = st.columns(2)
    with c1:
        b_filter = st.selectbox(
            "1. 門市維度過濾 (Store Location Dimension)", 
            ["🌐 全部分店 (Overall Branches)"] + df_b["name"].tolist() if not df_b.empty else ["🌐 全部分店 (Overall Branches)"]
        )
        sel_b = "ALL" if "全部" in b_filter else b_filter
    with c2:
        d_filter = st.selectbox(
            "2. 食物種類維度過濾 (Menu Item Dimension)", 
            ["🍱 全部餐點品項 (Overall Menu Items)"] + df_d["name"].tolist() if not df_d.empty else ["🍱 全部餐點品項 (Overall Menu Items)"]
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
    k1.markdown(f'<div class="clean-card"><div class="clean-label">審計樣本盤數 Audited Trays</div><div class="clean-val">{n} <span style="font-size:0.85rem;color:#94A3B8">TRAYS</span></div><div class="clean-sub">即時同步 Real-time Sync</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="clean-card"><div class="clean-label">平均殘食率 Waste Ratio</div><div class="clean-val" style="color:{"#EF4444" if avg_w > 25 else "#10B981"}">{avg_w:.1f}%</div><div class="clean-sub">基準目標 Target: &lt;15%</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="clean-card"><div class="clean-label">食材損耗總額 Total Loss</div><div class="clean-val" style="color:#F59E0B">HK${tot_hkd:,.1f}</div><div class="clean-sub">動態估算 Dynamic Valuation</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="clean-card"><div class="clean-label">累計碳排放 GHG Emissions</div><div class="clean-val" style="color:#3B82F6">{tot_co2:.2f} <span style="font-size:0.85rem;color:#94A3B8">kg</span></div><div class="clean-sub">Scope 3 ESG Metric</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # --------------------------------------------------------------------------
    # 核心亮點：Module 4 啟用時呈現「大家樂會員偏好與 Retargeting 數據庫」
    # --------------------------------------------------------------------------
    if modules["mod4"] and not df_raw.empty:
        st.markdown("#### 🎯 大家樂會員偏好與 Retargeting 數據中心 (Loyalty Insights & CRM Feed)")
        st.caption("透過 TrayZero+ 分析每位會員在各門市的真實殘食偏好，直接輸出給大家樂 Club 100 App 與自助點餐機做客製化預設與精準促銷。")

        unique_members = [m for m in df_raw["member_id"].dropna().unique().tolist() if m != "GUEST"]
        
        if unique_members:
            crm_records = []
            for mid in unique_members:
                prof = analyze_member_loyalty_profile(mid, df_raw)
                crm_records.append({
                    "會員卡號 (Member ID)": prof["member_id"],
                    "客群分類 (Segment)": prof["crm_segment"],
                    "最喜愛餐點 (Favorite Dish)": prof["favorite_dish"],
                    "平均殘食率 (Avg Waste)": f"{prof['avg_waste']:.1f}%",
                    "點餐機預設主食 (POS Default Rice)": prof["pos_default_rice"],
                    "點餐機預設醬汁 (POS Default Sauce)": prof["pos_default_sauce"],
                    "建議大家樂 CRM 推送優惠券策略 (Retargeting Strategy)": prof["retarget_strategy"]
                })
            
            df_crm = pd.DataFrame(crm_records)
            st.dataframe(df_crm, use_container_width=True)

            # 支援一鍵導出給大家樂 CRM 系統
            csv_crm = df_crm.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 匯出大家樂 Club 100 Retargeting 數據包 (Export CRM Feed CSV)",
                data=csv_crm,
                file_name=f"cdc_loyalty_retarget_feed_{datetime.date.today()}.csv",
                mime="text/csv",
                help="可直接導入大家樂 CRM 系統或 Kiosk 後台更新預設選項"
            )
        else:
            st.info("💡 目前資料庫中多為訪客（GUEST）數據，請在 Mode 1 輸入會員卡號以生成個人化 CRM 偏好畫像。")

        st.markdown("---")

    # 決策建議中心 (Advisory Directives)
    if n > 0:
        st.markdown("#### 🧭 大家樂總部營運與菜單工程建議 (Executive Advisory Hub)")
        scope = st.radio("覆盤時限 (Advisory Scope)", ["📅 日度營運覆盤建議 (Daily Operational Review)", "🗓️ 月度戰略採購建議 (Monthly Strategic Advisory)"], horizontal=True)
        scope_code = "DAILY" if "日度" in scope else "MONTHLY"
        date_col = "audit_date" if scope_code == "DAILY" else "audit_month"
        
        dates = df_filtered[date_col].dropna().unique().tolist()
        if not dates:
            dates = [datetime.date.today().strftime("%Y-%m-%d" if scope_code=="DAILY" else "%Y-%m")]
            
        s_date = st.selectbox(f"選擇審計{'日期' if scope_code=='DAILY' else '月份'} (Select Audit {'Date' if scope_code=='DAILY' else 'Month'})", dates)
        df_scope = df_filtered[df_filtered[date_col] == s_date]

        if not df_scope.empty:
            t_branch = sel_b if sel_b != "ALL" else df_scope.groupby("branch_name")["waste_ratio"].mean().idxmax()
            t_dish = sel_d if sel_d != "ALL" else df_scope.groupby("dish_name")["waste_ratio"].mean().idxmax()
            sub_avg_w = df_scope["waste_ratio"].mean()
            sub_loss = df_scope["cost_waste_hkd"].sum()

            c_adv1, c_adv2 = st.columns([1, 2])
            with c_adv1:
                st.markdown(f"""
                <div class="clean-card">
                    <div class="clean-label">當期指標摘要 (Scope Summary)</div>
                    <div style="font-size:0.88rem;color:#334155;line-height:1.8;margin-top:8px;">
                        • 審計盤數: <b>{len(df_scope)} 盤</b><br>
                        • 殘食率: <b style="color:#EF4444">{sub_avg_w:.1f}%</b><br>
                        • 目標門市: <b>{t_branch}</b><br>
                        • 目標餐點: <b>{t_dish}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c_adv2:
                # 廚房端指令
                st.markdown(f"""
                <div class="directive-card directive-chef">
                    <div class="directive-title">👨‍🍳 後廚出餐計量標準校準 Head Chef ({t_branch} • {t_dish})</div>
                    <div class="directive-body">【即時份量校準】平均殘食率達 {sub_avg_w:.1f}%。針對「{t_dish}」換裝標準平底飯勺（每份減量 30g 出餐），單期預估防損挽回 HK$ {max(150, round(sub_loss * 0.4)):,.0f}。</div>
                </div>
                """, unsafe_allow_html=True)

                # POS / Kiosk 端指令 (Module 3 / 4)
                if modules["mod3"] or modules["mod4"]:
                    st.markdown(f"""
                    <div class="directive-card directive-pos">
                        <div class="directive-title">🖥️ 點餐機 (Kiosk) 與大家樂 App 反向客製化連動</div>
                        <div class="directive-body">【智慧預設下發】系統已自動將高頻剩餘「{t_dish}」主食之會員，於點餐終端預設勾選「少飯（立減 $2）」或「少汁」，在點餐階段源頭減廢。</div>
                    </div>
                    """, unsafe_allow_html=True)

                # CRM Retargeting 端指令
                if modules["mod4"]:
                    st.markdown(f"""
                    <div class="directive-card directive-crm">
                        <div class="directive-title">🎯 大家樂 CRM 精準行銷與二次回購推廣 (Retargeting Action)</div>
                        <div class="directive-body">【會員偏好標籤同步】已提煉顧客吃得最乾淨的「最愛餐品」，向其 Club 100 App 發放專屬優惠券促成二次復購，提升顧客黏著度。</div>
                    </div>
                    """, unsafe_allow_html=True)

    # 深度圖表分析 (Module 2)
    if modules["mod2"] and not df_filtered.empty:
        st.markdown("---")
        st.markdown("#### 📈 Module 2: 深度商業智慧與數據洞察 (BI Deep Dive)")
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("##### 🏢 各門市平均殘食率 Store Waste Ratio (%)")
            st.bar_chart(df_filtered.groupby("branch_name")["waste_ratio"].mean(), color="#3B82F6")
        with g2:
            st.markdown("##### 🍱 各食物種類耗損 Waste Cost by Dish (HK$)")
            st.bar_chart(df_filtered.groupby("dish_name")["cost_waste_hkd"].sum(), color="#EF4444")

def render_mode3(df_b, df_d):
    st.markdown("### ⚙️ 基礎資料管理 (Master Data Management & Bulk Upload)")
    tab1, tab2 = st.tabs(["🏢 分店清單 (Branches)", "🍱 餐點品項管理 (Menu Items & AI Photo Registration)"])

    with tab1:
        st.markdown("#### 批次上傳分店清單 (Bulk Upload Branch Directory)")
        up_b = st.file_uploader("上傳分店 CSV (Upload Store CSV - Overwrites)", type=["csv"], key="up_b")
        if up_b:
            try:
                new_df_b = pd.read_csv(up_b, encoding="utf-8-sig")
                new_df_b.to_csv(BRANCH_FILE, index=False, encoding="utf-8-sig")
                st.success(f"🎉 成功更新 {len(new_df_b)} 間分店！(Successfully updated {len(new_df_b)} branches!)")
                st.rerun()
            except Exception as e: 
                st.error(f"Upload error: {e}")

        st.markdown("#### 線上手動編輯 (Live Branch Editor)")
        edit_b = st.data_editor(df_b, num_rows="dynamic", use_container_width=True, key="ed_b")
        if st.button("💾 儲存分店手動修改 (Save Branch Directory)", type="primary"):
            edit_b.to_csv(BRANCH_FILE, index=False, encoding="utf-8-sig")
            st.success("✅ 分店清單已成功儲存！(Branch directory saved successfully!)")
            st.rerun()

    with tab2:
        st.markdown("#### 📸 新增菜品照片上傳與 AI 辨識註冊 (Register New Dish via Photo Upload)")
        st.caption("在此上傳新菜品（如肉醬意粉）的參考照片，AI 將自動提取特徵向量並同步至前線辨識庫。")
        
        col_reg1, col_reg2 = st.columns([1, 1])
        with col_reg1:
            new_dish_id = st.text_input("品項編號 (Dish ID)", value=f"D{len(df_d)+1:02d}")
            new_dish_name = st.text_input("餐點名稱 (Dish Name)", placeholder="例: 焗肉醬意粉 (Baked Spaghetti Bolognese)")
            new_carb = st.selectbox("主要碳水主食 (Main Carbohydrate)", ["意大利麵/意粉 (Spaghetti)", "白米飯 (Steamed Rice)", "蛋炒飯 (Egg Fried Rice)", "中式麵條 (Noodles)", "無 (None)"])
            new_protein = st.text_input("主力蛋白質/主菜 (Protein Source)", placeholder="例: 慢燉牛肉醬 (Minced Beef Sauce)")
        
        with col_reg2:
            new_dish_photo = st.file_uploader("上傳菜品樣本照片 (Upload Dish Sample Photo for AI Feature Extraction)", type=["jpg", "png", "jpeg"], key="new_dish_photo_input")
            if new_dish_photo:
                photo_preview = Image.open(new_dish_photo)
                st.image(photo_preview, caption="菜品樣本預覽 (Sample Preview)", width=240)
        
        if st.button("🚀 建立新品項特徵並註冊至 AI (Train & Register Dish to AI)", type="primary"):
            if not new_dish_name:
                st.error("❌ 請輸入餐點名稱！(Please provide Dish Name)")
            else:
                new_row = pd.DataFrame([{
                    "dish_id": new_dish_id,
                    "name": new_dish_name,
                    "main_carb": new_carb.split(" ")[0],
                    "protein": new_protein if new_protein else "肉醬"
                }])
                df_updated = pd.concat([df_d, new_row], ignore_index=True).drop_duplicates(subset=["dish_id"], keep="last")
                df_updated.to_csv(DISH_FILE, index=False, encoding="utf-8-sig")
                st.success(f"🎉 成功建立新品項【{new_dish_name}】特徵！AI 即時辨識已生效。(New product registered successfully!)")
                st.rerun()

        st.markdown("---")
        st.markdown("#### 批次上傳餐點清單 (Bulk Upload Menu CSV Directory)")
        up_d = st.file_uploader("上傳餐點 CSV (Upload Menu CSV - Overwrites)", type=["csv"], key="up_d")
        if up_d:
            try:
                new_df_d = pd.read_csv(up_d, encoding="utf-8-sig")
                req_d = {"dish_id", "name", "main_carb", "protein"}
                if req_d.issubset(new_df_d.columns):
                    new_df_d.to_csv(DISH_FILE, index=False, encoding="utf-8-sig")
                    st.success(f"🎉 成功更新 {len(new_df_d)} 項餐點！(Successfully updated {len(new_df_d)} dishes!)")
                    st.rerun()
                else: 
                    st.error(f"Missing required columns: {req_d}")
            except Exception as e: 
                st.error(f"Upload error: {e}")

        st.markdown("#### 線上手動編輯 (Live Menu Editor)")
        edit_d = st.data_editor(df_d, num_rows="dynamic", use_container_width=True, key="ed_d")
        if st.button("💾 儲存餐點手動修改 (Save Menu Directory)", type="secondary"):
            edit_d.to_csv(DISH_FILE, index=False, encoding="utf-8-sig")
            st.success("✅ 餐點清單已成功儲存！(Menu directory saved successfully!)")
            st.rerun()

# ==============================================================================
# 6. Main Pipeline
# ==============================================================================
def main():
    st.set_page_config(
        page_title="TrayZero+ | 大家樂智能餐盤審計與會員閉環平台", 
        page_icon="🍽️", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_custom_css()
    init_db()
    df_b, df_d = load_master_data()

    with st.spinner("🚀 正在啟動雙核心 AI 引擎 (Initializing AI Engines)..."):
        engine = load_ai_engine()

    # Sidebar Logo
    logo_target = LOGO_FILE_PNG if os.path.exists(LOGO_FILE_PNG) else (LOGO_FILE_JPG if os.path.exists(LOGO_FILE_JPG) else None)
    if logo_target:
        col_l1, col_l2, col_l3 = st.sidebar.columns([0.15, 0.7, 0.15])
        with col_l2:
            st.image(logo_target, width=175)
    else:
        st.sidebar.warning("⚠️ 請上傳 CDC_810.png 至根目錄")

    st.sidebar.title("🧩 功能模組授權 (Modules)")
    mod_1 = st.sidebar.checkbox("M1: 營運監控 (Ops Core)", value=True, disabled=True, help="基礎核心模組，無法停用")
    mod_2 = st.sidebar.checkbox("M2: 深度分析 (BI Analytics)", value=True, help="啟用門市交叉報表、深度圖表與 CSV 導出")
    mod_3 = st.sidebar.checkbox("M3: 精準營銷 (Smart POS)", value=True, help="啟用點餐機少飯扣減與反向菜單推薦指令")
    mod_4 = st.sidebar.checkbox("M4: 會員閉環 (Loyalty Loop)", value=True, help="啟用 Club 100 卡號掃碼、個人化畫像與光盤獎勵券")

    active_modules = {
        "mod1": mod_1,
        "mod2": mod_2,
        "mod3": mod_3,
        "mod4": mod_4
    }

    render_header(active_modules)

    st.sidebar.markdown("---")
    st.sidebar.title("🎛️ 系統導航 (Navigation)")
    mode = st.sidebar.radio("工作模式 (Navigation)", [
        "Mode 1: 前線餐盤智能偵測與會員還盤 (Tray Station & Member Return)",
        "Mode 2: 總部即時營運與客群大盤 (Executive HQ & Analytics)",
        "Mode 3: 基礎資料設定 (Master Data Management)"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.caption("系統測試維護 (System Maintenance)")
    if st.sidebar.button("🗑️ 清空審計資料庫 (Reset Audit DB)", type="secondary"):
        reset_db()
        st.session_state["latest"] = None
        st.session_state["last_h"] = None
        st.session_state["active_upload_hash"] = None
        st.sidebar.success("✅ 資料庫已完全清空！(Database cleared!)")
        st.rerun()

    if mode.startswith("Mode 1"): 
        render_mode1(df_b, df_d, engine, active_modules)
    elif mode.startswith("Mode 2"): 
        render_mode2(df_b, df_d, engine, active_modules)
    else: 
        render_mode3(df_b, df_d)

if __name__ == "__main__":
    main()
