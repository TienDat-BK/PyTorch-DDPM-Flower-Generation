import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os
import sys

# Import từ project
sys.path.append(os.path.abspath(os.getcwd()))
from src.DiffusionImage import DiffusionDataset
from src.config import config

def download_sample_images(num_images=5, size=252):
    images = []
    print(f"Đang tải {num_images} ảnh từ Picsum (size {size}x{size})...")
    for i in range(num_images):
        try:
            # Sử dụng loremflickr để lấy ảnh mèo (cat)
            url = f"https://loremflickr.com/{size}/{size}/cat?lock={i}"
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            # Chuyển thành tensor (C, H, W) và scale về [0, 1]
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            images.append(img_tensor)
        except Exception as e:
            print(f"Lỗi khi tải ảnh {i}: {e}")
            images.append(torch.rand(3, size, size))
    return torch.stack(images)

def denormalize(x):
    """Chuyển từ [-1, 1] về [0, 1] để plot"""
    return (x + 1.0) / 2.0

def main():
    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    
    # 2. Lấy 5 tấm ảnh sạch ban đầu
    size = 252
    x0_raw = download_sample_images(num_images=5, size=size) # (5, 3, 252, 252)
    
    # Scale về [-1, 1] như yêu cầu
    x0_scaled = (x0_raw * 2.0 - 1.0).to(device)
    
    # 3. Sử dụng class DiffusionDataset của bạn
    print("Khởi tạo DiffusionDataset...")
    dataset = DiffusionDataset(x0_scaled)
    
    # 4. Lấy noise và plot check
    print("Đang tạo noise và vẽ biểu đồ...")
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    titles = ["Gốc (x_0)", "Bị nhòe (x_t)"]
    
    for i in range(5):
        # __getitem__ trả về (xt, t, noise)
        xt, t, noise = dataset[i]
        
        # Chuyển về numpy để hiển thị
        img_0 = denormalize(x0_scaled[i]).cpu().permute(1, 2, 0).numpy()
        img_t = denormalize(xt).cpu().permute(1, 2, 0).numpy()
        
        # Plot Original
        axes[0, i].imshow(np.clip(img_0, 0, 1))
        axes[0, i].set_title(f"Ảnh {i+1}")
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_ylabel(titles[0], size='large')
        
        # Plot Noised
        axes[1, i].imshow(np.clip(img_t, 0, 1))
        axes[1, i].set_title(f"x_t (t={t})")
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_ylabel(titles[1], size='large')

    plt.suptitle("Kiểm tra DiffusionDataset: Ảnh gốc vs Ảnh đã thêm nhiễu", fontsize=16)
    
    # Lưu kết quả
    save_path = "check_diffusion_image.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Đã lưu kết quả tại: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()
