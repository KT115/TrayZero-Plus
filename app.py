import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import pandas as pd
import datetime

# ----------------------------------------------------
# 0. 全局版面配置與樣式設定
# ----------------------------------------------------
st.set_page_config(
    page_title="TrayZero+ 智慧餐盤殘食審計系統",
    page_icon="🍽️",
    layout="wide"
)

# 初始化 Session State 用於儲存歷史審計記錄（支援儀表板數據展示）
if "audit_history" not in st.session_state:
    st.session_state["audit_history"] = []

# 初始化系統設定參數 (System Config)
if "menu_config" not in st.session_state:
    st.session_state["menu_config"] = {
        "大家樂招牌海南雞飯": {"default_weight_g": 450, "carbon_factor": 2.5},
        "大家樂原塊焗豬扒飯": {"default_weight_g": 500, "carbon_factor": 3.0},
        "港式燒味雙拼飯": {"default_weight_g": 420, "carbon_factor": 2.8},
        "粟米肉粒飯": {"default_weight_g": 400, "carbon_factor": 2.0}
    }

# ----------------------------------------------------
# 1. 載入 Hugging Face 微調模型 (使用快取加速)
# ----------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_trayzero_model():
    model_id = "kktlau115/trayzero-vit-regression"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.to(device)
    model.eval()
    
    return processor, model, device

with st.spinner("🔄 正在初始化 TrayZero+ AI 引擎..."):
    processor, model, device = load_trayzero_model()

# ----------------------------------------------------
# 2. 側邊欄：多模組導航 (Modules & Navigation)
# ----------------------------------------------------
st.sidebar.title("🍽️ TrayZero+ 導航選單")
st.sidebar.markdown("---")

app_module = st.sidebar.radio(
    "選擇系統模組：",
    ["📷 智慧審計系統 (Audit)", "📊 數據儀表板 (Dashboard)", "⚙️ 系統設定 (System Config)"]
)

st.sidebar.markdown("---")
st.sidebar.info("🤖 **AI 模型狀態**\n\n已連結: `kktlau115/trayzero-vit-regression` (ViT Regression)")

# ====================================================
# 模組一：智慧審計系統 (包含上傳與即時拍照模式)
# ====================================================
if app_module == "📷 智慧審計系統 (Audit)":
    st.title("📷 TrayZero+ 智慧餐盤殘食審計與回收站")
    st.markdown("透過 AI 視覺模型即時評估回收餐盤殘食率，自動換算重量、碳足跡並發放會員獎勵。")
    st.markdown("---")
    
    col_ctrl, col_display = st.columns([1, 1], gap="large")
    
    with col_ctrl:
        st.subheader("1️⃣ 餐點與輸入方式選擇")
        
        # 選擇餐點
        menu_db = st.session_state["menu_config"]
        selected_dish = st.selectbox("📝 選擇本次用餐餐點類型：", list(menu_db.keys()))
        
        # 選擇拍照/上傳方式
        input_method = st.radio("選擇影像輸入方式：", ["上傳圖片檔案 (File Upload)", "即時相機拍照 (Camera Capture)"])
        
        uploaded_image = None
        if input_method == "上傳圖片檔案 (File Upload)":
            uploaded_file = st.file_uploader("請上傳托盤回收照片", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                uploaded_image = Image.open(uploaded_file).convert("RGB")
        else:
            camera_file = st.camera_input("請對準回收台托盤拍照")
            if camera_file is not None:
                uploaded_image = Image.open(camera_file).convert("RGB")
                
    with col_display:
        st.subheader("2️⃣ AI 視覺審計分析結果")
        
        if uploaded_image is not None:
            st.image(uploaded_image, caption="待審計之回收餐盤影像", use_container_width=True)
            
            with st.spinner("🤖 AI 正在深度分析殘食比例..."):
                inputs = processor(images=uploaded_image, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    raw_pred = outputs.logits.item()
                    residue_ratio = max(0.0, min(1.0, raw_pred))
            
            # 數值計算
            dish_info = menu_db[selected_dish]
            waste_weight = dish_info["default_weight_g"] * residue_ratio
            carbon_wasted = (waste_weight / 1000.0) * dish_info["carbon_factor"]
            points_earned = int((1.0 - residue_ratio) * 50)
            
            # 呈現 Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("預測殘食比例", f"{residue_ratio * 100:.1f}%")
            m2.metric("剩餘食物重量", f"{waste_weight:.1f} g")
            m3.metric("發放會員積分", f"{points_earned} PTS")
            
            st.progress(residue_ratio, text=f"殘食佔比: {residue_ratio * 100:.1f}%")
            
            # 動態回饋
            if residue_ratio < 0.1:
                st.success("🌟 **完美光盤行動！** 感謝您減少食物浪費，積分已自動存入會員帳戶！")
            elif residue_ratio < 0.4:
                st.info("👍 **表現良好！** 大部分餐點都有食用完畢。")
            else:
                st.warning("⚠️ **檢測到較多剩食。** 系統已記錄本次數據，鼓勵下次適量點餐。")
                
            # 自動記錄至 Session 歷史供儀表板調用
            audit_record = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dish": selected_dish,
                "ratio": round(residue_ratio * 100, 1),
                "weight_g": round(waste_weight, 1),
                "points": points_earned
            }
            # 避免重複寫入同一張圖
            if not st.session_state["audit_history"] or st.session_state["audit_history"][-1]["timestamp"] != audit_record["timestamp"]:
                st.session_state["audit_history"].append(audit_record)
        else:
            st.info("👈 請先於左側上傳照片或使用相機拍照，AI 審計報告將在此處即時生成。")

# ====================================================
# 模組二：數據儀表板 (Dashboard)
# ====================================================
elif app_module == "📊 數據儀表板 (Dashboard)":
    st.title("📊 TrayZero+ 營運與廚餘審計儀表板")
    st.markdown("彙整回收台即時審計數據，追蹤每日廚餘總量與減碳成效。")
    st.markdown("---")
    
    history = st.session_state["audit_history"]
    
    if len(history) > 0:
        df_history = pd.DataFrame(history)
        
        col_d1, col_d2, col_d3 = st.columns(3)
        col_d1.metric("總審計餐盤數", f"{len(df_history)} 盤")
        col_d2.metric("平均殘食比例", f"{df_history['ratio'].mean():.1f}%")
        col_d3.metric("累計發放環保積分", f"{df_history['points'].sum()} PTS")
        
        st.markdown("### 📋 近期回收審計明細記錄")
        st.dataframe(df_history, use_container_width=True)
    else:
        st.info("📭 目前尚無審計歷史記錄。請先至 **【智慧審計系統】** 執行幾筆托盤掃描，儀表板將自動生成數據圖表！")

# ====================================================
# 模組三：系統設定 (System Config)
# ====================================================
elif app_module == "⚙️ 系統設定 (System Config)":
    st.title("⚙️ TrayZero+ 系統參數與菜單設定")
    st.markdown("在此調整後端資料庫的餐點標準總重量與碳足跡係數。")
    st.markdown("---")
    
    st.subheader("📝 現有菜單資料庫檢視與微調")
    menu_db = st.session_state["menu_config"]
    
    for dish_name, config in menu_db.items():
        with st.expander(f"🍽️ {dish_name}"):
            new_weight = st.number_input(f"預設總重量 (g) - {dish_name}", value=config["default_weight_g"], key=f"w_{dish_name}")
            new_factor = st.number_input(f"碳排放係數 (kg CO2/kg) - {dish_name}", value=config["carbon_factor"], key=f"f_{dish_name}")
            
            if st.button(f"儲存變更 ({dish_name})", key=f"btn_{dish_name}"):
                st.session_state["menu_config"][dish_name]["default_weight_g"] = new_weight
                st.session_state["menu_config"][dish_name]["carbon_factor"] = new_factor
                st.success(f"✅ {dish_name} 參數已成功更新！")
