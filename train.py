import torch
from src.config import *
from src.DiffusionImage import *
from models.unet import *

@torch.no_grad()
def GenerateImage(model, num_samples=1):
    model.eval()
    device = config.device
    
    # 1. Bắt đầu từ nhiễu trắng x_T ~ N(0, I)
    x = torch.randn((num_samples, config.input_channels, config.image_size, config.image_size)).to(device)
    
    # 2. Lặp ngược từ T-1 về 0
    for t in reversed(range(config.num_timesteps)):
        # Tạo batch t
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        
        # Dự đoán nhiễu bằng model
        predicted_noise = model(x, t_batch)
        
        # Lấy các hệ số alpha, beta tại bước t
        alpha = config.alphas[t].to(device)
        alpha_hat = config.alphas_hat[t].to(device)
        beta = config.betas[t].to(device)
        
        # Tính toán x_{t-1} theo công thức DDPM
        if t > 0:
            z = torch.randn_like(x)
        else:
            z = 0 # Bước cuối không thêm nhiễu
            
        # sigma_t thường được chọn là sqrt(beta_t)
        sigma_t = torch.sqrt(beta)
        
        x = (1 / torch.sqrt(alpha)) * (
            x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise
        ) + sigma_t * z
        
    # 3. Đưa ảnh từ [-1, 1] về [0, 255] và định dạng uint8
    x = (x + 1.0) / 2.0
    x = torch.clamp(x, 0, 1)
    x = (x * 255).to(torch.uint8)
    
    model.train()
    # TRả về tensor dạng (num_samples, 3, 32, 32) với giá trị [0, 255]
    return x

import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os

def save_samples(samples, epoch):
    # Tạo thư mục lưu trữ nếu chưa có
    os.makedirs("output", exist_ok=True)
    
    samples = samples.cpu() # Đảm bảo ở CPU
    
    for i in range(samples.shape[0]):
        # Lấy từng ảnh trong batch (C, H, W)
        img = samples[i]
        # Lưu ảnh với định dạng {epoch}_{chỉ số}.png
        filename = f"output/{epoch}_{i}.png"
        
        # Dùng vutils.save_image để lưu. 
        # BẮT BUỘC: hàm `save_image` của thư viện torchvision bị lỗi nếu truyền trực tiếp tensor uint8.
        # Ta cần phải chia 255.0 quay lại dạng float [0, 1] chỉ riêng cho bước lưu file này.
        vutils.save_image(img.float() / 255.0, filename)
    print(f"--- Đã lưu {samples.shape[0]} ảnh mẫu vào thư mục 'output' ---")

class DPM_pinelineTrain:
    def __init__(self):
        # Sửa tên class từ Unet -> UNet cho đúng với định nghĩa trong unet.py
        #load model đã train trước đó
        self.model = UNet(image_size=config.image_size, input_channels=config.input_channels)
        self.model.load_state_dict(torch.load("model_450_flower_pro.pth"))
        self.model.to(config.device)
        
        # Đưa optimizer ra ngoài vòng lặp train, nó sẽ theo model trong suốt vòng đời của class
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        # scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=500, 
            eta_min=0.00002
        )

        self.X_0 = GetDatasetImage(5000)
        self.all_step_losses = []
        
    def train(self, epochs, generate_every_n_epochs=30):
        # Tạo sẵn folder để lưu plots
        os.makedirs("loss_plots", exist_ok=True)

        for epoch in range(epochs):
            # Chuẩn bị dataset và dataloader
            dataset = DiffusionDataset(self.X_0)
            dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
            
            # ----------------------------FIT----------------------------
            print("Epoch", epoch+1, "training...")
                
            # Gọi hàm train_step thay vì fit
            step_losses = self.model.train_step(dataloader, self.optimizer, device=config.device)
            self.scheduler.step()
            if step_losses:
                self.all_step_losses.extend(step_losses)

            # Vẽ biểu đồ loss mỗi 30 epoch
            if (epoch + 1) % 30 == 0 and len(self.all_step_losses) > 0:
                plt.figure(figsize=(10, 5))
                plt.plot(self.all_step_losses, label="Step Loss", alpha=0.6)
                
                # Tính moving average mượt hơn (ví dụ trung bình mỗi 50 steps)
                if len(self.all_step_losses) >= 50:
                    import numpy as np
                    window = 50
                    moving_avg = np.convolve(self.all_step_losses, np.ones(window)/window, mode='valid')
                    plt.plot(range(window-1, len(self.all_step_losses)), moving_avg, color='red', label="Moving Average (50 steps)")

                plt.xlabel("Training Steps")
                plt.ylabel("Loss")
                plt.title(f"Training Loss curve up to Epoch {epoch+1}")
                plt.legend()
                plt.grid(True)
                plot_filename = f"loss_plots/loss_epoch_{epoch+1}.png"
                plt.savefig(plot_filename)
                plt.close()
                print(f"--- Đã lưu biểu đồ loss tại '{plot_filename}' ---")

            if epoch % generate_every_n_epochs == 0:
                print(f"Đang tiến hành tạo ảnh mẫu tại epoch {epoch}...")
                samples = GenerateImage(self.model, num_samples=4)
                save_samples(samples, epoch)
            
            # mỗi 50 epoch thì lưu model
            if epoch % 50 == 0:
                torch.save(self.model.state_dict(), f"model_{epoch}.pth")
        

def main():
    pipeline = DPM_pinelineTrain()
    # Huấn luyện mô hình
    pipeline.train(epochs=config.num_epochs)
    
    # Thử tạo ảnh sau khi train
    print("Đang tạo ảnh mẫu...")
    samples = GenerateImage(pipeline.model, num_samples=4)
    # Ở đây bạn có thể dùng matplotlib để show samples nếu muốn
    print("Đã tạo xong ảnh mẫu (tensor).")

if __name__ == "__main__":
    main()
