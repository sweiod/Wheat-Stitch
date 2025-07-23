

import os
import sys
import glob
import cv2
import torch
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F  # <--- 新增

# —— 添加模块搜索路径 ——
ROOT = os.path.dirname(__file__)
sys.path.append(os.path.join(ROOT, "Codes"))
sys.path.append(os.path.join(ROOT, "Composition", "Codes"))

# —— Warp 阶段依赖 ——
import Codes.utils.torch_DLT as torch_DLT
import Codes.utils.torch_homo_transform as torch_homo_transform
import Codes.utils.torch_tps_transform as torch_tps_transform
from Codes import grid_res
from Codes.Siamese_SE import SiameseResNet50
from Codes.network_SE import Network as WarpNet
from Codes.test_output import build_output_model as warp_build

# —— Composition 阶段依赖 ——
from Composition.Codes.network import Network as CompNet, build_model as comp_build

# —— 预处理变换 ——
resize_512 = T.Resize((512,512))
grid_h, grid_w = grid_res.GRID_H, grid_res.GRID_W

# —— 配置 ——
DATASET_DIR     = r"D:\study\Image-stitching\UDIS-Wheat\stitch\input"
OUT_DIR         = r"D:\study\Image-stitching\UDIS-Wheat\stitch\output"
WARP_CKPT_PATH  = r"D:\study\Image-stitching\UDIS-Wheat\UDIS2-W\model\epoch100_model_SE.pth"
COMP_CKPT_PATH  = r"D:\study\Image-stitching\UDIS-Wheat\UDIS2-W\Composition\model\epoch050_model.pth"
GPU_ID          = "0"

def load_image(path, device):
    """读取图片并归一化到[-1,1], 返回 tensor([1,3,H,W])"""
    img = cv2.imread(path).astype(np.float32)
    img = (img / 127.5) - 1.0
    img = img.transpose(2,0,1)  # HWC -> CHW
    return torch.from_numpy(img).unsqueeze(0).to(device)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # —— 加载 Warp 网络 ——
    warp_net = WarpNet().to(device).eval()
    ck1 = torch.load(WARP_CKPT_PATH, map_location=device)
    warp_net.load_state_dict(ck1['model'])
    print(f"[Warp] Loaded {WARP_CKPT_PATH}")

    # —— 加载 Composition 网络 ——
    comp_net = CompNet().to(device).eval()
    ck2 = torch.load(COMP_CKPT_PATH, map_location=device)
    comp_net.load_state_dict(ck2['model'])
    print(f"[Comp] Loaded {COMP_CKPT_PATH}")

    # —— 收集所有待拼接图像，按文件名排序 ——
    img_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "*.jpg")))
    if len(img_paths) < 2:
        print("至少需要两张图片进行拼接。")
        return

    # —— 初始化画布为第一张图 ——
    canvas = load_image(img_paths[0], device)

    # —— 依次与后续每张图进行链式拼接 ——
    for idx, next_path in enumerate(img_paths[1:], start=1):
        print(f"Stitching: {os.path.basename(img_paths[idx-1])} + {os.path.basename(next_path)}")
        next_img = load_image(next_path, device)

        # —— 关键：保证 next_img 大小 == canvas 大小 ——
        _,_,Hc,Wc = canvas.size()
        _,_,Hn,Wn = next_img.size()
        if (Hn, Wn) != (Hc, Wc):
            # 用双线性插值调整到一致
            next_img = F.interpolate(next_img, size=(Hc, Wc), mode='bilinear', align_corners=False)

        with torch.no_grad():
            # Warp 阶段
            out_w = warp_build(warp_net, canvas, next_img)
            warp1 = out_w['final_warp1']
            mask1 = out_w['final_warp1_mask']
            warp2 = out_w['final_warp2']
            mask2 = out_w['final_warp2_mask']

            # Composition 阶段
            out_c = comp_build(comp_net, warp1, warp2, mask1, mask2)
            canvas = out_c['stitched_image']  # 更新画布

        # 可选：保存中间结果
        inter = canvas[0].cpu().numpy()
        inter = ((inter + 1.0) * 127.5).clip(0,255).astype(np.uint8).transpose(1,2,0)
        save_mid = os.path.join(OUT_DIR, f"stitched_{idx:03d}.jpg")
        cv2.imwrite(save_mid, inter)
        print(f"  --> saved intermediate: {save_mid}")

    # —— 保存最终拼接结果 ——
    final = canvas[0].cpu().numpy()
    final = ((final + 1.0) * 127.5).clip(0,255).astype(np.uint8).transpose(1,2,0)
    save_final = os.path.join(OUT_DIR, "stitched_all.jpg")
    cv2.imwrite(save_final, final)
    print(f"[Result] Final stitched image saved to: {save_final}")

if __name__ == "__main__":
    main()
