import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ----------------------------------------------------
# 版面配置與樣式設定 (改為寬螢幕 wide)
# ----------------------------------------------------
st.set_page_config(
    page_title="TrayZero+ 智慧餐盤殘食審計系統",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ TrayZero+ 智慧餐盤殘食審計系統")
st.markdown("歡迎使用大家樂智慧回收台審計系統。請於左側選取餐點並上傳托盤照片，AI 將自動進行深度審計。")
st.markdown("---")

# ----------------------------------------------------
# 1. 載入 Hugging Face 微調好的專屬模型 (使用快取加速)
# ----------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_trayzero_model():
    model_id = "kktlau115/trayzero-vit-regression"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"📥 正在載入模型: {model_id} (運行於 {device.upper()})...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.to(device)
    model.eval()
    
    return processor, model, device

# ----------------------------------------------------
# 2. 側邊欄控制區 (Sidebar Controls)
# ----------------------------------------------------
with st.sidebar:
    st.header("⚙️ 審計控制台")
    
    with st.spinner("🔄 載入 AI 引擎中..."):
        processor, model, device = load_trayzero_model()
    st.success("✅ AI 模型就緒")
    
    st.markdown("---")
    
    # 菜單與預設重量資料庫
    menu_database = {
        "大家樂招牌海南雞飯": {"default_weight_g": 450, "carbon_factor": 2.5},
        "大家樂原塊焗豬扒飯": {"default_weight_g": 500, "carbon_factor": 3.0},
        "港式燒味雙拼飯": {"default_weight_g": 420, "carbon_factor": 2.8},
        "粟米肉粒飯": {"default_weight_g": 400, "carbon_factor": 2.0}
    }

    selected_dish = st.selectbox(
        "📝 選擇餐點類型：", 
        list(menu_database.keys())
    )
    
    st.markdown("---")
    uploaded_file = st.file_uploader("📷 上傳回收托盤照片", type=["jpg", "jpeg", "png"])

# ----------------------------------------------------
# 3. 主畫面：影像展示與審計分析報告
# ----------------------------------------------------
if uploaded_file is not None:
    # 採用雙欄左右對稱排版 (左邊看圖，右邊看數據)
    col_left, col_right = st.columns([1, 1], gap="large")
    
    with col_left:
        st.subheader("📷 回收餐盤影像")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="已上傳的托盤回收照片", use_container_width=True)
        
    with col_right:
        st.subheader("🤖 AI 即時推論分析")
        
        with st.spinner("AI 正在分析托盤畫面殘食比例..."):
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                raw_prediction = outputs.logits.item()
                residue_ratio = max(0.0, min(1.0, raw_prediction))
        
        # 計算後端數據
        dish_info = menu_database[selected_dish]
        total_weight = dish_info["default_weight_g"]
        waste_weight_g = total_weight * residue_ratio
        carbon_wasted = (waste_weight_g / 1000.0) * dish_info["carbon_factor"]
        points_earned = int((1.0 - residue_ratio) * 50)
        
        # 數據指標卡片
        m1, m2, m3 = st.columns(3)
        m1.metric("預測殘食比例", f"{residue_ratio * 100:.1f}%")
        m2.metric("估算剩餘重量", f"{waste_weight_g:.1f} g")
        m3.metric("會員環保積分", f"{points_earned} PTS")
        
        st.progress(residue_ratio, text=f"殘食佔比: {residue_ratio * 100:.1f}%")
        
        # 動態反饋訊息
        if residue_ratio < 0.1:
            st.success("🌟 **完美光盤行動！** 感謝您減少食物浪費，積分已自動存入！")
        elif residue_ratio < 0.4:
            st.info("👍 **表現良好！** 大部分餐點都有食用完畢。")
        else:
            st.warning("⚠️ **檢測到較多剩食。** 鼓勵下次適量點餐，減少廚餘碳足跡。")
else:
    st.info("👈 請先從左側側邊欄上傳餐盤回收照片，系統將自動為您生成審計報告。")
