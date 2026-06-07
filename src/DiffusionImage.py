import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import numpy as np
from src.config import *

class DiffusionDataset(Dataset):
    def __init__(self, x0_data):
        """
        x0_data: Tensor chứa toàn bộ ảnh sạch (đã đẩy lên VRAM như bạn muốn)
        """
        self.x0 = x0_data

    def __len__(self):
        return len(self.x0)

    def __getitem__(self, idx):
        img = self.x0[idx]
        
        # 1. Chọn t ngẫu nhiên cho từng ảnh một
        t = torch.randint(0, config.num_timesteps, (1,)).item()
        
        # 2. Sinh nhiễu e ngẫu nhiên
        noise = torch.randn_like(img)
        
        # 3. Tạo xt bằng công thức nhảy vọt
        # Lấy giá trị căn alpha_bar tại bước t
        sqrt_alphas_hat = config.sqrt_alphas_hat[t]
        sqrt_one_minus_alphas_hat = config.sqrt_one_minus_alphas_hat[t]
        
        xt = sqrt_alphas_hat * img + sqrt_one_minus_alphas_hat * noise
        
        # Trả về bộ 3: Ảnh nhòe, Bước thời gian, và Nhiễu gốc (Ground Truth)
        return xt, t, noise

def GetDatasetImage(num_images=1000):
    """
    Lấy bộ dataset ảnh thú cưng từ OxfordIIITPet (Chỉ lấy Chó).
    - num_images: số lượng ảnh muốn lấy
    - Thực hiện Horizon Flip nếu số lượng ảnh gốc không đủ.
    - Trả về: Tensor (N, 3, 64, 64) đã scale về [-1, 1]
    """
    import torchvision.transforms as transforms
    import torchvision.transforms.functional as TF
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Sử dụng config.data_path cho chuyên nghiệp
    # split='trainval' chứa 3680 ảnh (cả Chó và Mèo)
    # target_types='binary-category' để phân biệt Chó (1) và Mèo (0)
    dataset = datasets.OxfordIIITPet(root=config.data_path, split='trainval', download=True, transform=transform, target_types='binary-category')
    images = []
    
    print(f"Bắt đầu lọc ảnh loài Chó từ OxfordIIITPet...")
    # Lặp để lấy đủ số lượng num_images (chỉ lấy ảnh Chó)
    for i in range(len(dataset)):
        img, species = dataset[i]
        if species == 1: # 1 là Dog, 0 là Cat
            images.append(img)
            
        if len(images) >= num_images:
            break
            
    num_found = len(images)
    print(f"Đã tìm thấy {num_found} ảnh chó gốc.")
        
    # 2. Kiểm tra nếu vẫn thiếu ảnh so với yêu cầu
    if len(images) < num_images:
        needed = num_images - len(images)
        # Số lượng có thể lật tối đa là số ảnh gốc chúng ta vừa lấy
        to_flip = min(needed, len(images))
        
        print(f"Dataset không đủ {num_images} ảnh chó (chỉ có {num_found}). "
              f"Đang thực hiện lật ngang (Horizontal Flip) thêm {to_flip} ảnh...")
        
        for i in range(to_flip):
            img_flipped = TF.hflip(images[i])
            images.append(img_flipped)
        
    if not images:
        return torch.zeros((num_images, 3, 64, 64))
        
    pet_tensor = torch.stack(images)
    print(f"Hoàn tất! Tổng cộng đã chuẩn bị: {pet_tensor.shape[0]} ảnh.")
    
    return pet_tensor