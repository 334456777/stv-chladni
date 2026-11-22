"""生成 stv 的图标：深色圆角底上的金色克拉尼沙纹。

方板模态 φ = cos(mπu)·cos(nπv) + cos(nπu)·cos(mπv)：32px 起用 (4,6)，更小的尺寸细节会糊，改用 (2,4)。
用法（需要 numpy、Pillow）：python scripts/make_icon.py assets/stv.ico assets/icon.png
"""
import sys
import numpy as np
from PIL import Image

SAND = np.array([224, 167, 94]) / 255.0   # gold 预设的沙色 #E0A75E
BG = np.array([16, 15, 14]) / 255.0

def render(size):
    m, n = (4, 6) if size >= 32 else (2, 4)
    ss = 8 if size <= 64 else 4
    s = size * ss
    t = (np.arange(s) + 0.5) / s
    x, y = np.meshgrid(t, t)
    # 圆角方形底
    r = 0.225
    dx = np.maximum(np.abs(x - 0.5) - (0.5 - r), 0)
    dy = np.maximum(np.abs(y - 0.5) - (0.5 - r), 0)
    inside = (np.hypot(dx, dy) <= r).astype(float)
    # 板面内缩一点，沙纹不贴边
    inset = 0.085
    u = (x - inset) / (1 - 2 * inset)
    v = (y - inset) / (1 - 2 * inset)
    pi = np.pi
    f = np.cos(m*pi*u)*np.cos(n*pi*v) + np.cos(n*pi*u)*np.cos(m*pi*v)
    fu = -m*pi*np.sin(m*pi*u)*np.cos(n*pi*v) - n*pi*np.sin(n*pi*u)*np.cos(m*pi*v)
    fv = -n*pi*np.cos(m*pi*u)*np.sin(n*pi*v) - m*pi*np.cos(n*pi*u)*np.sin(m*pi*v)
    d = np.abs(f) / np.maximum(np.hypot(fu, fv), 1e-6) * (1 - 2 * inset)   # 到节线的距离（占图标宽度）
    half = max(0.020, 0.75 / size)                                          # 线的半宽，小图标加粗
    line = np.clip(1.0 - d / half, 0, 1) ** 0.8
    plate = ((u > 0) & (u < 1) & (v > 0) & (v < 1)).astype(float)
    if size >= 64:
        rng = np.random.default_rng(7)
        grain = rng.random((s, s))
        line = line * (0.72 + 0.28 * grain)                                 # 沙粒质感
        line = np.maximum(line, (grain > 0.9965) * 0.55 * plate)             # 零星散落的沙粒
    a = line * plate * inside
    rgb = BG * (1 - a[..., None]) + SAND * a[..., None]
    rgba = np.dstack([rgb, inside])
    img = Image.fromarray((rgba * 255 + 0.5).astype(np.uint8), "RGBA")
    return img.resize((size, size), Image.LANCZOS)

if __name__ == "__main__":
    ico, png = sys.argv[1], sys.argv[2]
    sizes = [16, 20, 24, 32, 40, 48, 64, 128, 256]
    imgs = [render(s) for s in sizes]
    imgs[-1].save(png, optimize=True)
    imgs[-1].save(ico, format="ICO", sizes=[(s, s) for s in sizes], append_images=imgs[:-1])
