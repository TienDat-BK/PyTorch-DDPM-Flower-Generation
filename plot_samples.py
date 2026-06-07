import torch
import matplotlib.pyplot as plt
from src.config import config
from models.unet import UNet
from train import GenerateImage
import glob
import os

file_name = "model_450_flower_pro.pth"

def main():
    print("Khởi tạo mô hình...")
    # 1. Khởi tạo mô hình
    model = UNet(image_size=config.image_size, input_channels=config.input_channels).to(config.device)
    
    # 2. (Tuỳ chọn) Load weights nếu có file checkpoint
    if  file_name == "":
        checkpoints = glob.glob("model_*.pth")
        if checkpoints:
            # Tìm checkpoint mới nhất
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
            print(f"Tìm thấy file weights: {latest_checkpoint}. Đang tải...")
        try:
            model.load_state_dict(torch.load(latest_checkpoint, map_location=config.device))
        except Exception as e:
            print(f"Lỗi khi tải weights: {e}")
    else:
        print("Không thấy file .pth nào. Sẽ dùng mô hình với trọng số khởi tạo ngẫu nhiên (chưa train).")

    # chỉ đích danh file model cần load
    model_path = "model_450_flower_pro.pth"
    print(f"Đang tải weights từ: {model_path}...")
    try:
        model.load_state_dict(torch.load(model_path, map_location=config.device))
    except Exception as e:
        print(f"Lỗi khi tải weights: {e}")

    # 3. Sinh 10 ảnh mẫu
    print("Đang tạo 10 ảnh từ hàm GenerateImage...")
    samples = GenerateImage(model, num_samples=20)
    
    print(f"Kích thước tensor trả về: {samples.shape}, kiểu dữ liệu: {samples.dtype}")
    
    # 4. Hiển thị 10 ảnh
    samples = samples.cpu()
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    axes = axes.flatten()
    for i in range(20):
        img = samples[i]
        # Torch image là (C, H, W). Matplotlib cần (H, W, C)
        img_np = img.permute(1, 2, 0).numpy()
        
        axes[i].imshow(img_np)
        axes[i].axis('off')
        axes[i].set_title(f"Ảnh {i+1}")
        
    plt.tight_layout()
    
    save_path = "plot_20_samples.png"
    plt.savefig(save_path)
    print(f"Đã lưu ảnh tổng hợp vào: {save_path}")
    
    # plt.show() # Uncomment dòng này nếu bạn muốn GUI tự động bật lên

if __name__ == "__main__":
    main()
