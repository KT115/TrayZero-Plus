# 1. 安裝開源 AI 橡皮擦模型（完全免費、無任何 API）
!pip install -q simple-lama-inpainting pillow numpy

import torch
from PIL import Image, ImageDraw, ImageFilter
from simple_lama_inpainting import SimpleLama

# 2. 載入 LaMa 擦除模型至 Colab GPU
print("正在載入開源 LaMa 物理擦除模型...")
simple_lama = SimpleLama()

# 3. 讀取您拍的一哥焗飯
img = Image.open("ref_baked.jpg").convert("RGB")
w, h = img.size

# 4. 建立「湯匙挖掉一半」的遮罩（Mask）
# 白色區域 = 被吃掉的部分（將被物理擦除露出瓷盤底）
mask = Image.new("L", (w, h), 0)
draw = ImageDraw.Draw(mask)

# 精準鎖定焗盤內部右側與中央（挖掉約 50% 食物）
cx, cy = int(w * 0.52), int(h * 0.65)
rx, ry = int(w * 0.22), int(h * 0.12)
draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)

# 輕微邊緣羽化，讓食物斷面自然
mask = mask.filter(ImageFilter.GaussianBlur(radius=3))

print("🚀 正在執行真實物理吃掉（純減法，絕無加料）...")
# LaMa 會直接把選定區域擦除，並露出下方真實白瓷盤底
result = simple_lama(img, mask)
result.save("test_lama_pure_eaten.png")
print("✅ 測試完成！請在 Colab 左側點開 test_lama_pure_eaten.png 檢視效果。")
