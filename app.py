def detect_tray(image, engine, selected_dish="", carb_type_from_csv=""):
    image_rgb = image.convert("RGB")
    width, height = image_rgb.size
    dev = engine["device"]

    # 1. SWIN 模型計算基礎數值迴歸
    swin_inputs = engine["swin_processor"](images=image_rgb, return_tensors="pt").to(dev)
    with torch.no_grad():
        swin_out = engine["swin_model"](**swin_inputs)
        raw_pred = swin_out.logits.item() if swin_out.logits.numel() == 1 else swin_out.logits[0][0].item()
        swin_ratio = float(1.0 / (1.0 + np.exp(-raw_pred)))
        swin_ratio = max(0.0, min(1.0, swin_ratio))

    # 2. CLIP 獨立評估主食與蛋白質的殘留狀態
    carb_state_labels = ["clean empty bowl with no rice", "half eaten rice", "full untouched rice and sauce"]
    protein_state_labels = ["no meat left", "half eaten meat or chicken", "full untouched meat or pork chop"]

    carb_probs = compute_clip_similarity(image_rgb, carb_state_labels, engine)
    protein_probs = compute_clip_similarity(image_rgb, protein_state_labels, engine)

    # 根據機率分佈換算獨立百分比 (0% ~ 100%)
    # 索引 0 代表空/少，索引 1 代表一半，索引 2 代表完整
    carb_ratio = float(carb_probs[1] * 0.5 + carb_probs[2] * 0.95)
    protein_ratio = float(protein_probs[1] * 0.5 + protein_probs[2] * 0.95)

    # 確保數值合理並與 swin 綜合融合
    carb_ratio = max(0.0, min(1.0, (carb_ratio + swin_ratio) / 2))
    protein_ratio = max(0.0, min(1.0, (protein_ratio + swin_ratio) / 2))

    # 總體殘食佔比為兩者平均或加權
    ratio = float((carb_ratio * 0.5) + (protein_ratio * 0.5))
    ratio = max(0.0, min(1.0, ratio))

    if ratio < 0.1:
        primary_cat = "光盤 Clean Plate"
        accent_color = "#10B981"
    elif ratio > 0.85:
        primary_cat = "完整未動餐點 (未食用浪費)"
        accent_color = "#DC2626"
    else:
        primary_cat = "主食與主菜殘留"
        accent_color = "#D97706"

    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # 🌟 讓每一項元件顯示各自獨立計算出的百分比
    items = [
        {"分類項目 Category": "主食 (Carb)", "置信度 Confidence": f"{carb_probs.max():.1%}", "佔比 Coverage": f"{carb_ratio*100:.1f}%"},
        {"分類項目 Category": "蛋白質 (Protein)", "置信度 Confidence": f"{protein_probs.max():.1%}", "佔比 Coverage": f"{protein_ratio*100:.1f}%"},
    ]

    box = [int(width * 0.15), int(height * 0.15), int(width * 0.85), int(height * 0.85)]
    draw.rectangle(box, outline=accent_color, width=4)
    draw.text((box[0] + 10, box[1] + 10), f"綜合殘食率: {ratio*100:.1f}%", fill=accent_color)

    return img_draw, items, ratio, primary_cat, True
