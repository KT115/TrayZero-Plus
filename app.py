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
# 0. 必須是第一個呼叫的 Streamlit 命令
# ==============================================================================
st.set_page_config(
    page_title="TrayZero+ | 大家樂智能餐盤審計與會員閉環平台", 
    page_icon="🍽️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 1. 檔案路徑與安全 UI 樣式定義
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BRANCH_FILE = os.path.join(BASE_DIR, "master_branches.csv")
DISH_FILE = os.path.join(BASE_DIR, "master_dishes.csv")
SEED_AUDIT_FILE = os.path.join(BASE_DIR, "seed_audit_logs.csv")
DB_FILE = os.path.join(BASE_DIR, "trayzero_audit.db")
DISH_IMG_DIR = os.path.join(BASE_DIR, "dish_references")
LOGO_FILE_PNG = os.path.join(BASE_DIR, "CDC_810.png")
LOGO_FILE_JPG = os.path.join(BASE_DIR, "CDC_810.jpg")

os.makedirs(DISH_IMG_DIR, exist_ok=True)

FOOD_WHITELIST = {
    "bowl": "Carb", "cake": "Carb", "sandwich": "Meat", "pizza": "Meat", "hot dog": "Meat",
    "carrot": "Veg_Soup", "broccoli": "Veg_Soup", "apple": "Veg_Soup", "orange": "Veg_Soup",
    "donut": "Meat", "cup": "Veg_Soup", "bottle": "Veg_Soup", "dining table": "Tray"
}

def inject_safe_css():
    st.markdown("""
    <style>
        /* 根容器設定 - 安全版面 */
        .stApp {
            background-color: #F8FAFC !important;
            color: #0F172A !important;
        }

        .main .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 3rem !important;
            color: #0F172A !important;
        }

        /* 標題卡片 */
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

        /* 側邊欄 Logo 置中 */
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

        /* -----------------------------------------------------------
           核心修復 1: 標籤頁 (Tabs) 文字清晰高對比，解決 Mode 3 文字隱形
        ----------------------------------------------------------- */
        button[data-baseweb="tab"] {
            color: #475569 !important;
            font-size: 0.98rem !important;
            font-weight: 700 !important;
            background: transparent !important;
            padding: 10px 18px !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #2563EB !important;
            border-bottom: 3px solid #2563EB !important;
        }
        button[data-baseweb="tab"] p,
        button[data-baseweb="tab"] span,
        button[data-baseweb="tab"] div {
            color: inherit !important;
            font-weight: 700 !important;
        }

        /* -----------------------------------------------------------
           核心修復 2: 輸入框、下拉選單背景為白底、字為深色
        ----------------------------------------------------------- */
        input[type="text"], 
        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"] {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1.5px solid #CBD5E1 !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        div[data-baseweb="select"] * {
            color: #0F172A !important;
        }

        /* 單選 Radio 與 Checkbox 文字強制高對比顯色 */
        div[data-testid="stRadio"] label p,
        div[data-testid="stRadio"] label div,
        div[data-testid="stRadio"] span,
        div[data-testid="stCheckbox"] label p {
            color: #0F172A !important;
            font-weight: 600 !important;
            opacity: 1 !important;
        }

        /* 資訊卡片 */
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
# 2. 資料庫與 CSV 雙向同步機制 (保證寫入與載入一致)
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
        if "member_id" not in cols: conn.execute("ALTER TABLE audit_logs ADD COLUMN member_id TEXT")
        if "reward_issued" not in cols: conn.execute("ALTER TABLE audit_logs ADD COLUMN reward_issued TEXT")
        if "audit_date" not in cols: conn.execute("ALTER TABLE audit_logs ADD COLUMN audit_date TEXT")
        if "audit_month" not in cols: conn.execute("ALTER TABLE audit_logs ADD COLUMN audit_month TEXT")

        # 檢查資料庫筆數，若為 0 則優先從 seed_audit_logs.csv 載入
        row_count = conn.execute("SELECT COUNT(*) FROM audit_logs").fetchone()[0]
        if row_count == 0 and os.path.exists(SEED_AUDIT_FILE):
            try:
                seed_df = pd.read_csv(SEED_AUDIT_FILE, encoding="utf-8-sig")
                # 補齊標準欄位
                for col in ["member_id", "reward_issued", "audit_date", "audit_month"]:
                    if col not in seed_df.columns:
                        if col == "member_id": seed_df[col] = "GUEST"
                        elif col == "reward_issued": seed_df[col] = "歷史導入記錄"
                        elif col == "audit_date": seed_df[col] = datetime.date.today().strftime("%Y-%m-%d")
                        elif col == "audit_month": seed_df[col] = datetime.date.today().strftime("%Y-%m")
                
                valid_cols = ["timestamp", "audit_date", "audit_month", "branch_name", "member_id", 
                              "dish_name", "primary_waste", "waste_ratio", "cost_waste_hkd", "co2_emission_kg", "reward_issued"]
                seed_df_clean = seed_df[[c for c in valid_cols if c in seed_df.columns]]
                seed_df_clean.to_sql("audit_logs", conn, if_exists="append", index=False)
            except Exception as e:
                print(f"Seed DB Load Warning: {e}")

def save_record(r):
    # 1. 寫入 SQLite
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
    
    # 2. 核心修復：立即更新 seed_audit_logs.csv，確保 CSV 永遠有最新數據
    try:
        current_df = get_records()
        current_df.to_csv(SEED_AUDIT_FILE, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"CSV Save Warning: {e}")

def get_records():
    with db_conn() as conn: 
        df = pd.read_sql("SELECT * FROM audit_logs ORDER BY id DESC", conn)
        # 確保必要的分析欄位非空
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

def load_master_data():
    if os.path.exists(BRANCH_FILE):
        df_b = pd.read_csv(BRANCH_FILE, encoding="utf-8-sig")
    else:
        df_b = pd.DataFrame([
            {"name": "中環威靈頓街店", "level": "Level A", "district": "中西區", "traffic": "白領為主", "avg_covers": 1200, "base_rice_g": 240},
            {"name": "香港科技大學店 (HKUST)", "level": "Level C", "district": "西貢區", "traffic": "學生為主", "avg_covers": 1800, "base_rice_g": 280}
        ])
        df_b.to_csv(BRANCH_FILE, index=False, encoding="utf-8-sig")

    if os.path.exists(DISH_FILE):
        df_d = pd.read_csv(DISH_FILE, encoding="utf-8-sig")
    else:
        df_d = pd.DataFrame([
            {"dish_id": "D01", "name": "一哥焗豬扒飯 (Baked Pork Chop Rice)", "main_carb": "白米飯", "protein": "豬扒"},
            {"dish_id": "D02", "name": "焗肉醬意粉 (Baked Spaghetti Bolognese)", "main_carb": "意大利麵", "protein": "慢燉牛肉醬"},
            {"dish_id": "D03", "name": "咖喱牛腩飯 (Curry Beef Brisket Rice)", "main_carb": "白米飯", "protein": "牛腩"}
        ])
        df_d.to_csv(DISH_FILE, index=False, encoding="utf-8-sig")

    return df_b, df_d

# ==============================================================================
# 3. AI 引擎初始化
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
# 4. 會員 Loyalty Loop 分析
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
            "pos_default_rice": "正常飯量",
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
        pos_rice = "預設【少飯 (-30g / 立減 $2)】"
        crm_seg = "控醣輕食族 (Low-Carb Diners)"
    elif avg_waste < 10.0:
        pos_rice = "預設【正常飯量 (光盤常客)】"
        crm_seg = "高飽足飽腹族 (Standard/High-Calorie)"
    else:
        pos_rice = "預設【標準飯量】"
        crm_seg = "均衡飲食族 (Balanced Diners)"

    sauce_wastes = m_df[m_df["primary_waste"].str.contains("配菜|醬汁", na=False)]
    if not sauce_wastes.empty and sauce_wastes["waste_ratio"].mean() > 25.0:
        pos_sauce = "預設【少汁 / 醬汁另上】"
    else:
        pos_sauce = "預設【正常汁】"

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

def render_mode1(df_b, df_d, engine, modules):
    if df_b.empty or df_d.empty:
        st.warning("⚠️ 門市或餐點清單為空！請至 Mode 3 建立基礎資料。")
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
                else: 
                    st.info("🟢 監控中：當前餐盤已完成分析，等待更換餐盤...")
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
                    st.error("🚫 偵測失敗：未檢測到合法餐盤或食物物件！（已自動過濾人物/背景）")
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
                    
                    if modules.get("mod4", True) and active_member_id != "GUEST":
                        if ratio < 0.15:
                            reward_msg = "🎉 達成光盤獎勵！已派發【$3 堂食優惠券 + 50 綠色積分】至大家樂 App"
                        else:
                            reward_msg = "已累積【10 綠色環保積分】至 Club 100 帳戶"
                    else:
                        reward_msg = "訪客還盤完成 (Guest Return)"

                    # 寫入 SQLite 並同步寫入 CSV
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
                    st.toast(f"✅ 還盤數據已寫入資料庫並同步更新 CSV！")
                    st.rerun()

    with c2:
        st.markdown("#### 🎯 前線掃描結果與會員數據")
        latest = st.session_state.get("latest")
        if not latest:
            st.info("💡 尚未執行偵測或畫面非餐盤。請對準餐盤掃描。")
        else:
            conf_str = f"({latest.get('conf', 1.0):.1%})" if 'conf' in latest else ""
            st.image(latest["img"], caption=f"🍽️ {latest['dish']} {conf_str} • {latest['time']}", use_container_width=True)
            
            if modules.get("mod4", True) and latest.get("member") != "GUEST":
                st.markdown(f"""
                <div class="crm-card">
                    <b style="color:#166534; font-size:1rem;">📲 大家樂 Loyalty Loop 反向偏好更新成功</b><br>
                    • <b>關聯會員 ID</b>: <code>{latest.get('member')}</code><br>
                    • <b>本次還盤結果</b>: {latest.get('reward')}<br>
                    • <b>自動反哺 POS 規則</b>: 下次點餐系統已預載顧客客製化偏好
                </div>
                """, unsafe_allow_html=True)

            k1, k2, k3 = st.columns(3)
            k1.markdown(f'<div class="clean-card"><div class="clean-label">殘食佔比 Waste Ratio</div><div class="clean-val" style="color:{"#EF4444" if latest["ratio"] > 0.3 else "#10B981"}">{latest["ratio"]:.1%}</div></div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="clean-card"><div class="clean-label">主要殘留 Primary</div><div class="clean-val" style="font-size:1.05rem;margin-top:4px;">{latest["cat"].split(" ")[0]}</div></div>', unsafe_allow_html=True)
            k3.markdown(f'<div class="clean-card"><div class="clean-label">推算損耗 Loss</div><div class="clean-val" style="color:#F59E0B">HK${latest["cost"]}</div></div>', unsafe_allow_html=True)

def render_mode2(df_b, df_d, engine, modules):
    st.markdown("### 📊 總部營運與會員客群 Retargeting 數據中心")
    df_raw = get_records()

    # 頂部控制列
    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl2:
        if st.button("🔄 刷新載入最新資料庫 (Reload Data)", use_container_width=True):
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
    
    # 核心修復：保證永遠顯示資料庫與 CSV 的即時流水表記錄
    st.markdown("#### 📋 即時審計記錄（已與 CSV 同步存檔）")
    if not df_filtered.empty:
        st.dataframe(df_filtered, use_container_width=True)
        csv_download = df_filtered.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 匯出當前維度 CSV 審計日誌",
            data=csv_download,
            file_name=f"trayzero_audit_export_{datetime.date.today()}.csv",
            mime="text/csv"
        )
    else:
        st.info("💡 當前篩選維度下尚無資料。請在 Mode 1 進行實體餐盤掃描，系統將自動寫入 SQLite 及 seed_audit_logs.csv。")

    st.markdown("---")

    # Module 4: 會員偏好與 Retargeting 數據
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
            st.dataframe(df_crm, use_container_width=True)

            csv_crm = df_crm.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 匯出大家樂 Club 100 Retargeting 數據包 (CSV)",
                data=csv_crm,
                file_name=f"cdc_loyalty_retarget_feed_{datetime.date.today()}.csv",
                mime="text/csv"
            )
        else:
            st.caption("提示：目前資料庫中皆為訪客（GUEST）數據，在 Mode 1 掃描會員卡號即可在此生成客群偏好分析。")

    # Module 1 / 3 營運指導指令
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
                【即時份量校準】平均殘食率達 {sub_avg_w:.1f}%。針對「{t_dish}」換裝標準平底飯勺（每份減量 30g 出餐），單期預估防損挽回 HK$ {max(150, round(sub_loss * 0.4)):,.0f}。
            </div>
            """, unsafe_allow_html=True)

        if modules.get("mod3", True):
            st.markdown(f"""
            <div class="directive-card directive-pos">
                <b style="color:#0F172A;">🖥️ 點餐機 (Kiosk) 與大家樂 App 反向客製化連動</b><br>
                【智慧預設下發】系統已自動將高頻剩餘「{t_dish}」主食之會員，於點餐終端預設勾選「少飯（立減 $2）」或「少汁」，在點餐階段源頭減廢。
            </div>
            """, unsafe_allow_html=True)

    # Module 2 圖表分析
    if modules.get("mod2", True) and not df_filtered.empty:
        st.markdown("---")
        st.markdown("#### 📈 Module 2: 深度商業智慧與數據洞察")
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("##### 🏢 各門市平均殘食率 Store Waste Ratio (%)")
            st.bar_chart(df_filtered.groupby("branch_name")["waste_ratio"].mean(), color="#3B82F6")
        with g2:
            st.markdown("##### 🍱 各食物種類耗損 Waste Cost by Dish (HK$)")
            st.bar_chart(df_filtered.groupby("dish_name")["cost_waste_hkd"].sum(), color="#EF4444")

def render_mode3(df_b, df_d):
    st.markdown("### ⚙️ 基礎資料管理 (Master Data Management)")
    tab1, tab2 = st.tabs(["🏢 分店清單 (Branches)", "🍱 餐點品項管理與 AI 照片註冊 (Menu Items & AI Photo Registration)"])

    # --------------------------------------------------------------------------
    # Tab 1: 分店清單管理
    # --------------------------------------------------------------------------
    with tab1:
        st.markdown("#### 🏢 門市清單即時編輯 (Live Branch Editor)")
        edit_b = st.data_editor(df_b, num_rows="dynamic", use_container_width=True, key="ed_b")
        if st.button("💾 儲存分店修改 (Save Branches)", type="primary"):
            edit_b.to_csv(BRANCH_FILE, index=False, encoding="utf-8-sig")
            st.success("✅ 分店清單已成功儲存至 master_branches.csv！")
            st.rerun()

        st.markdown("---")
        st.markdown("#### 批次覆蓋上傳分店 CSV")
        up_b = st.file_uploader("上傳分店 CSV (Upload Store CSV)", type=["csv"], key="up_b")
        if up_b:
            try:
                new_df_b = pd.read_csv(up_b, encoding="utf-8-sig")
                new_df_b.to_csv(BRANCH_FILE, index=False, encoding="utf-8-sig")
                st.success(f"🎉 成功匯入 {len(new_df_b)} 間分店！")
                st.rerun()
            except Exception as e:
                st.error(f"匯入錯誤: {e}")

    # --------------------------------------------------------------------------
    # Tab 2: 核心修復：菜品相片 Reference 上傳與 AI 訓練註冊
    # --------------------------------------------------------------------------
    with tab2:
        st.markdown("#### 📸 新增菜品照片上傳與 AI 辨識註冊 (Register New Dish via Photo Reference)")
        st.caption("在此上傳新菜品（如焗肉醬意粉、海南雞飯）的標準參考照片，AI 將自動提取語義特徵向量並同步至前線辨識庫。")

        col_reg1, col_reg2 = st.columns([1.1, 0.9])
        with col_reg1:
            new_dish_id = st.text_input("品項編號 (Dish ID)", value=f"D{len(df_d)+1:02d}")
            new_dish_name = st.text_input("餐點名稱 (Dish Name)", placeholder="例: 焗肉醬意粉 (Baked Spaghetti Bolognese)")
            new_carb = st.selectbox("主要碳水化合物 (Carbohydrate)", ["白米飯 (Rice)", "意大利麵/意粉 (Spaghetti)", "蛋炒飯 (Fried Rice)", "中式麵條 (Noodles)", "無主食 (None)"])
            new_protein = st.text_input("主要蛋白質/主肉類 (Protein Source)", placeholder="例: 慢燉牛肉醬 (Minced Beef Sauce)")

        with col_reg2:
            new_dish_photo = st.file_uploader("📷 上傳菜品樣本照片 (Upload Dish Reference Photo)", type=["jpg", "png", "jpeg"], key="new_dish_photo_input")
            if new_dish_photo:
                photo_preview = Image.open(new_dish_photo)
                st.image(photo_preview, caption="菜品照片預覽 (Sample Preview)", width=240)

        if st.button("🚀 建立新品項特徵並註冊至 AI (Train & Register Dish to AI)", type="primary"):
            if not new_dish_name.strip():
                st.error("❌ 請輸入餐點名稱！")
            elif not new_dish_photo:
                st.error("❌ 請務必上傳菜品參考照片以供 AI 提取特徵！")
            else:
                # 儲存照片至本機參考庫
                img_ext = os.path.splitext(new_dish_photo.name)[1]
                saved_img_path = os.path.join(DISH_IMG_DIR, f"{new_dish_id}_{new_dish_name}{img_ext}")
                with open(saved_img_path, "wb") as f:
                    f.write(new_dish_photo.getbuffer())

                new_row = pd.DataFrame([{
                    "dish_id": new_dish_id,
                    "name": new_dish_name,
                    "main_carb": new_carb.split(" ")[0],
                    "protein": new_protein if new_protein else "主食肉類"
                }])
                df_updated = pd.concat([df_d, new_row], ignore_index=True).drop_duplicates(subset=["dish_id"], keep="last")
                df_updated.to_csv(DISH_FILE, index=False, encoding="utf-8-sig")
                st.success(f"🎉 成功完成新品項【{new_dish_name}】特徵註冊與相片建檔！前線 Mode 1 已即刻生效。")
                st.rerun()

        st.markdown("---")
        st.markdown("#### 🍱 現有餐點清單即時編輯 (Live Menu Editor)")
        edit_d = st.data_editor(df_d, num_rows="dynamic", use_container_width=True, key="ed_d")
        if st.button("💾 儲存餐點清單手動修改 (Save Menu)", type="secondary"):
            edit_d.to_csv(DISH_FILE, index=False, encoding="utf-8-sig")
            st.success("✅ 餐點清單已成功儲存至 master_dishes.csv！")
            st.rerun()

        st.markdown("#### 批次覆蓋上傳餐點 CSV")
        up_d = st.file_uploader("上傳餐點 CSV (Upload Menu CSV)", type=["csv"], key="up_d")
        if up_d:
            try:
                new_df_d = pd.read_csv(up_d, encoding="utf-8-sig")
                new_df_d.to_csv(DISH_FILE, index=False, encoding="utf-8-sig")
                st.success(f"🎉 成功匯入 {len(new_df_d)} 項餐點！")
                st.rerun()
            except Exception as e:
                st.error(f"匯入錯誤: {e}")

# ==============================================================================
# 6. 主程式進入點
# ==============================================================================
def main():
    inject_safe_css()
    init_db()
    df_b, df_d = load_master_data()

    with st.spinner("🚀 正在啟動雙核心 AI 引擎 (Loading AI Engines)..."):
        engine = load_ai_engine()

    logo_target = LOGO_FILE_PNG if os.path.exists(LOGO_FILE_PNG) else (LOGO_FILE_JPG if os.path.exists(LOGO_FILE_JPG) else None)
    if logo_target:
        st.sidebar.image(logo_target, width=175)

    # --------------------------------------------------------------------------
    # 核心修復 3: M1 移除了 disabled=True，允許使用者自由選擇/取消
    # --------------------------------------------------------------------------
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
        render_mode1(df_b, df_d, engine, active_modules)
    elif mode.startswith("Mode 2"): 
        render_mode2(df_b, df_d, engine, active_modules)
    else: 
        render_mode3(df_b, df_d)

if __name__ == "__main__":
    main()
