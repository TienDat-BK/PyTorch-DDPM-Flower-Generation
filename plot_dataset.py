import torch
import matplotlib.pyplot as plt
from src.DiffusionImage import GetDatasetImage
import subprocess
import sys


def check_gdown():
    try:
        import gdown
    except ImportError:
        print("Đang cài đặt 'gdown' để hỗ trợ tải dataset từ Google Drive...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])

def main():
    check_gdown()
    print("Đang lấy 10 ảnh từ dataset (chó)...")
    # Lấy 10 ảnh từ dataset
    # Hàm này trả về tensor (10, 3, 64, 64) với giá trị trong khoảng [-1, 1]
    images = GetDatasetImage(num_images=1000)
    # lấy random 10 tấm ảnh
    images = images[torch.randperm(images.shape[0])[:10]]
    print(f"Kích thước tensor: {images.shape}")
    print(f"Khoảng giá trị: [{images.min().item():.2f}, {images.max().item():.2f}]")

    if images.sum() == 0:
        print("CẢNH BÁO: Tensor rỗng (toàn số 0). Có thể do lỗi tải dataset.")

    # Hiển thị 10 ảnh
    fig, axes = plt.subplots(1, 10, figsize=(20, 3))
    
    # Chuyển từ [-1, 1] về [0, 255] chuẩn RGB (Dùng float 0-1 cho imshow ổn định hơn)
    images_display = (images + 1.0) / 2.0
    images_display = images_display.clamp(0, 1)
    
    for i in range(10):
        img = images_display[i]
        # Chuyển từ (C, H, W) sang (H, W, C) cho matplotlib
        img_np = img.permute(1, 2, 0).numpy()
        
        axes[i].imshow(img_np) # 64x64 không cần interpolation='nearest' quá mức
        axes[i].axis('off')
        axes[i].set_title(f"Face {i+1}")
        
    plt.tight_layout()
    save_path = "dataset_10_portraits.png"
    plt.savefig(save_path)
    print(f"Đã lưu ảnh dataset vào: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
