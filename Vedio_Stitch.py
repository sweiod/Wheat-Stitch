import os
import sys
import cv2
import torch
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

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
VIDEO_PATH         = r"D:\study\Image-stitching\UDIS-Wheat\stitch\input\vedio-2.mp4"
FRAME_INTERVAL_S   = 1   # 每隔 1 秒抽一帧
OUT_DIR            = r"D:\study\Image-stitching\UDIS-Wheat\stitch\output\1"
WARP_CKPT_PATH     = r"D:\study\Image-stitching\UDIS-Wheat\UDIS2-W\model\epoch100_model_SE.pth"
COMP_CKPT_PATH     = r"D:\study\Image-stitching\UDIS-Wheat\UDIS2-W\Composition\model\epoch050_model.pth"
GPU_ID             = "0"

def frame_to_tensor(frame, device):
    """把一帧 BGR 图像转为归一化到[-1,1]的 Tensor([1,3,H,W])"""
    img = frame.astype(np.float32)
    img = (img / 127.5) - 1.0
    img = img.transpose(2,0,1)  # HWC -> CHW
    return torch.from_numpy(img).unsqueeze(0).to(device)

def extract_keyframes(video_path, interval_s, device):
    """从视频按秒数间隔抽帧，返回 Tensor 列表"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = int(fps * interval_s)
    keyframes, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            keyframes.append(frame_to_tensor(frame, device))
        idx += 1
    cap.release()
    return keyframes

def tensor_to_bgr_uint8(tensor):
    """把[-1,1] Tensor([1,3,H,W]) 转回 BGR uint8 numpy([H,W,3])"""
    arr = tensor[0].cpu().numpy().transpose(1,2,0)  # HWC
    img = ((arr + 1.0) * 127.5).clip(0,255).astype(np.uint8)
    return img

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # —— 加载网络 ——
    warp_net = WarpNet().to(device).eval()
    ck1 = torch.load(WARP_CKPT_PATH, map_location=device)
    warp_net.load_state_dict(ck1['model'])
    print(f"[Warp] Loaded {WARP_CKPT_PATH}")

    comp_net = CompNet().to(device).eval()
    ck2 = torch.load(COMP_CKPT_PATH, map_location=device)
    comp_net.load_state_dict(ck2['model'])
    print(f"[Comp] Loaded {COMP_CKPT_PATH}")

    # —— 抽取关键帧 ——
    print("Extracting keyframes …")
    frames = extract_keyframes(VIDEO_PATH, FRAME_INTERVAL_S, device)
    if len(frames) < 2:
        print("关键帧不足以拼接。")
        return
    print(f"Extracted {len(frames)} keyframes.")

    # —— 初始化画布为第一帧 ——
    canvas = frames[0]

    # —— 链式拼接所有关键帧 ——
    for idx, next_img in enumerate(frames[1:], start=1):
        print(f"Stitching frame {idx} / {len(frames)-1}")

        # 保证尺寸一致
        _,_,Hc,Wc = canvas.size()
        _,_,Hn,Wn = next_img.size()
        if (Hn, Wn) != (Hc, Wc):
            next_img = F.interpolate(next_img, size=(Hc, Wc),
                                     mode='bilinear', align_corners=False)

        # 深度拼接
        with torch.no_grad():
            out_w = warp_build(warp_net, canvas, next_img)
            warp1, mask1 = out_w['final_warp1'], out_w['final_warp1_mask']
            warp2, mask2 = out_w['final_warp2'], out_w['final_warp2_mask']
            out_c = comp_build(comp_net, warp1, warp2, mask1, mask2)
            canvas = out_c['stitched_image']

        # 转回 numpy
        inter_np = tensor_to_bgr_uint8(canvas)      # 当前拼接结果
        curr_np  = tensor_to_bgr_uint8(next_img)    # 原始下一帧

        # —— 保证同尺寸 ——
        h1, w1 = curr_np.shape[:2]
        h2, w2 = inter_np.shape[:2]
        if (h1, w1) != (h2, w2):
            inter_np = cv2.resize(inter_np, (w1, h1), interpolation=cv2.INTER_AREA)

        # 计算 PSNR
        psnr_val = compare_psnr(curr_np, inter_np, data_range=255)

        # 动态选择 win_size：不超过 7，且不超过最小边，保证为奇数
        min_dim = min(h1, w1)
        win_size = min(7, min_dim)
        if win_size % 2 == 0:
            win_size -= 1
        # 如果图像过小，直接跳过 SSIM 计算
        if win_size < 3:
            ssim_val = float('nan')
            print(f"[Metrics] Frame {idx:03d} — PSNR: {psnr_val:.2f} dB, SSIM: N/A (图像过小，win_size={win_size})")
        else:
            ssim_val = compare_ssim(
                curr_np, inter_np,
                data_range=255,
                channel_axis=2,
                win_size=win_size
            )
            print(f"[Metrics] Frame {idx:03d} — PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f} (win_size={win_size})")

        # 保存中间结果
        save_mid = os.path.join(OUT_DIR, f"inter_{idx:03d}.jpg")
        cv2.imwrite(save_mid, inter_np)

    # —— 保存最终全景图 ——
    pano = tensor_to_bgr_uint8(canvas)
    save_path = os.path.join(OUT_DIR, "panorama.jpg")
    cv2.imwrite(save_path, pano)
    print(f"[Result] Panorama saved to: {save_path}")

if __name__ == "__main__":
    main()
