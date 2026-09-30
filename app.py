import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ----------------------------------------------------
版面配置與樣式設定
# ----------------------------------------------------
st.set_page_config(
    page_title="TrayZero+ 智慧餐盤殘食審計系統",
    page_icon="🍽️",
    layout="centered"
)

st.title("🍽️ TrayZero+ 智慧餐盤殘食審計系統")
st.markdown("上傳大家的樂回收台餐盤照片，AI 系統將自動進行殘食率迴歸預測、重量換算與會員積分發放！")
st.markdown("---")

# ----------------------------------------------------
1. 載入 Hugging Face 微調好的專屬模型 (使用快取加速)
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

with st.spinner("🔄 正在初始化 TrayZero+ AI 引擎與載入雲端模型..."):
    processor, model, device = load_trayzero_model()

st.success("✅ AI 模型載入完畢，系統就緒！")

# ----------------------------------------------------
2. 後端商業邏輯資料庫：菜單與預設總重量 (對應第 2、3 項)
# ----------------------------------------------------
menu_database = {
    "大家樂招牌海南雞飯": {"default_weight_g": 450, "carbon_factor": 2.5},
    "大家樂原塊焗豬扒飯": {"default_weight_g": 500, "carbon_factor": 3.0},
    "港式燒味雙拼飯": {"default_weight_g": 420, "carbon_factor": 2.8},
    "粟米肉粒飯": {"default_weight_g": 400, "carbon_factor": 2.0}
}

selected_dish = st.selectbox(
    "📝 請選擇本次用餐的餐點類型（用於後端計算實際重量與碳足跡）：", 
    list(menu_database.keys())
)

# ----------------------------------------------------
3. 影像上傳與即時推論介面
# ----------------------------------------------------
uploaded_file = st.file_uploader("📷 請上傳回收台托盤照片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 讀取並顯示圖片
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="已上傳的餐盤回收照片", use_container_width=True)
    
    # 執行 AI 模型推論
    with st.spinner("🤖 AI 正在分析托盤畫面並預測殘食比例..."):
        # 影像前處理
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # 進行模型預測 (Regression)
        with torch.no_grad():
            outputs = model(**inputs)
            raw_prediction = outputs.logits.item()
            
            # 將數值嚴格限制在 0.0 到 1.0 之間
            residue_ratio = max(0.0, min(1.0, raw_prediction))

    # ----------------------------------------------------
    4. 計算後端數據（重量、碳足跡、積分獎勵）
    # ----------------------------------------------------
    dish_info = menu_database[selected_dish]
    total_weight = dish_info["default_weight_g"]
    
    # 計算實際剩餘殘食重量 (g)
    waste_weight_g = total_weight * residue_ratio
    
    # 計算碳排放浪費量 (kg CO2)
    carbon_wasted = (waste_weight_g / 1000.0) * dish_info["carbon_factor"]
    
    # 會員環保積分：殘食率愈低，積分愈高 (光盤最高 50 分)
    points_earned = int((1.0 - residue_ratio) * 50)

    # ----------------------------------------------------
    5. 呈現視覺化審計報告面板
    # ----------------------------------------------------
    st.markdown("---")
    st.subheader("📊 TrayZero+ 智慧審計分析報告")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("AI 預測殘食比例", f"{residue_ratio * 100:.1f}%")
    col2.metric("估算剩餘殘食重量", f"{waste_weight_g:.1f} g")
    col3.metric("發放會員環保積分", f"{points_earned} PTS")
    
    # 視覺化進度條
    st.progress(residue_ratio, text=f"殘食佔比進度條: {residue_ratio * 100:.1f}%")
    
    # 根據殘食比例給予動態反饋
    if residue_ratio < 0.1:
        st.success("🌟 **完美光盤行動！** 感謝您減少食物浪費，環保積分已自動存入您的會員帳戶！")
    elif residue_ratio < 0.4:
        st.info("👍 **表現良好！** 大部分餐點都有食用完畢，繼續保持。")
    else:
        st.warning("⚠️ **檢測到較多剩食。** 系統已記錄本次數據，鼓勵下次適量點餐，減少廚餘碳足跡。")
