"""
compute_fid.py
==============
Chấm điểm FID (Fréchet Inception Distance) cho mô hình DPM.
Sử dụng torchmetrics.image.fid.FrechetInceptionDistance — clean & chuẩn.

- Load model từ model_450.pth
- Load toàn bộ ảnh thật từ Flowers102 (split='train')
- Sinh đúng số lượng ảnh fake bằng số ảnh thật
- Tính FID bằng torchmetrics

Chạy bằng lệnh:
    py compute_fid.py
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

# ─── Thiết lập sys.path để import được các module nội bộ ─────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import config
from models.unet import UNet
from train import GenerateImage

# ─── Hằng số ─────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "model_450_flower_pro.pth"
BATCH_SIZE_GEN  = 32   # Số ảnh sinh mỗi lần; giảm xuống nếu bị OOM
BATCH_SIZE_REAL = 64   # Batch size khi feed ảnh thật vào FID metric


def main():
    print("=" * 60)
    print("  Chấm điểm FID cho mô hình DPM  (dùng torchmetrics)")
    print("=" * 60)
    device = config.device
    print(f"  Device   : {device}")
    print(f"  Image sz : {config.image_size}x{config.image_size}")
    print(f"  Timesteps: {config.num_timesteps}")
    print(f"  Model    : {CHECKPOINT_PATH}")

    # ── Bước 1: Khởi tạo FID metric ──────────────────────────────────────────
    # feature=2048  → dùng pool3 của InceptionV3 (chuẩn FID gốc)
    # normalize=True → nhận ảnh float [0,1] thay vì uint8 [0,255]
    print("\n[1/4] Khởi tạo FrechetInceptionDistance (feature=2048)...")
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # ── Bước 2: Load & feed ảnh thật vào FID metric ──────────────────────────
    print("\n[2/4] Load ảnh thật từ Flowers102 (split='train')...")
    transform_real = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),   # → float32 [0, 1]
    ])

    real_dataset = datasets.Flowers102(
        root=config.data_path,
        split='train',
        download=True,
        transform=transform_real,
    )

    num_real   = len(real_dataset)
    real_loader = DataLoader(real_dataset, batch_size=BATCH_SIZE_REAL,
                             shuffle=False, num_workers=0)

    print(f"  → Dataset có {num_real} ảnh. Đang đưa vào FID metric...")
    for imgs, _ in tqdm(real_loader, desc="  Real images"):
        imgs = imgs.to(device)
        fid_metric.update(imgs, real=True)

    print(f"  ✓ Đã feed xong {num_real} ảnh thật.")

    # ── Bước 3: Load model & sinh ảnh fake ───────────────────────────────────
    print(f"\n[3/4] Load model từ '{CHECKPOINT_PATH}' và sinh {num_real} ảnh fake...")

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy '{CHECKPOINT_PATH}'.\n"
            "Hãy chắc chắn chạy script từ thư mục MyDPM."
        )

    model = UNet(image_size=config.image_size, input_channels=config.input_channels)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("  ✓ Load model thành công!")

    generated = 0
    pbar = tqdm(total=num_real, desc="  Fake images")

    while generated < num_real:
        batch = min(BATCH_SIZE_GEN, num_real - generated)

        # GenerateImage trả về (B, 3, H, W) uint8 [0, 255]
        imgs_uint8 = GenerateImage(model, num_samples=batch)

        # Chuyển về float [0, 1] để match normalize=True của FID
        imgs_float = imgs_uint8.float() / 255.0
        imgs_float = imgs_float.to(device)

        fid_metric.update(imgs_float, real=False)
        generated += batch
        pbar.update(batch)

    pbar.close()
    print(f"  ✓ Đã sinh và feed xong {generated} ảnh fake.")

    # ── Bước 4: Tính FID ─────────────────────────────────────────────────────
    print("\n[4/4] Đang tính FID score...")
    fid_score = fid_metric.compute().item()

    print("\n" + "=" * 60)
    print(f"  ✅ FID Score = {fid_score:.4f}")
    print("=" * 60)
    print("  (FID càng thấp thì mô hình sinh ảnh càng tốt)")
    print("  Tham khảo:")
    print("    FID <  10  : Chất lượng rất cao (SOTA)")
    print("    FID 10–50  : Chất lượng tốt")
    print("    FID 50–100 : Chất lượng trung bình")
    print("    FID > 100  : Cần cải thiện thêm")

    return fid_score


if __name__ == "__main__":
    main()
