import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np


def depth_to_erp_panorama(depth_cubemap, erp_height=64, erp_width=128):
    """
    将立方体贴图深度图转换为ERP（等距圆柱投影）全景深度图
    
    Args:
        depth_cubemap: (B, 6, H, W) 立方体贴图深度图，6个面分别为：
                      [front, right, left, back, up, down]
        erp_height: ERP图像的高度
        erp_width: ERP图像的宽度
    
    Returns:
        erp_depth: (B, 1, erp_height, erp_width) ERP格式的全景深度图
    """
    B, _, H, W = depth_cubemap.shape
    
    # 创建ERP坐标网格
    # theta: 经度角 (0 to 2π)
    # phi: 纬度角 (-π/2 to π/2)
    theta = torch.linspace(0, 2 * math.pi, erp_width, device=depth_cubemap.device)
    phi = torch.linspace(-math.pi/2, math.pi/2, erp_height, device=depth_cubemap.device)
    
    # 创建网格
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='xy')
    
    # 将球面坐标转换为3D笛卡尔坐标
    x = torch.cos(phi_grid) * torch.cos(theta_grid)
    y = torch.cos(phi_grid) * torch.sin(theta_grid)
    z = torch.sin(phi_grid)
    
    # 扩展维度以匹配batch size
    x = x.unsqueeze(0).expand(B, -1, -1)  # (B, erp_height, erp_width)
    y = y.unsqueeze(0).expand(B, -1, -1)
    z = z.unsqueeze(0).expand(B, -1, -1)
    
    # 初始化ERP深度图
    erp_depth = torch.zeros(B, erp_height, erp_width, device=depth_cubemap.device)
    
    # 定义立方体贴图面的方向向量
    # [front(+X), right(-Y), left(+Y), back(-X), up(+Z), down(-Z)]
    face_dirs = torch.tensor([
        [-1, 0, 0],   # front (I)
        [0, -1, 0],  # right (yaw_p)
        [0, 1, 0],   # left (yaw_n)
        [1, 0, 0],  # back (yaw_180)
        [0, 0, -1],   # up (pitch_up)
        [0, 0, 1],  # down (pitch_down)
    ], device=depth_cubemap.device, dtype=torch.float32)
    
    # 计算所有面的点积，找到每个像素对应的最大面
    max_dot_product = torch.zeros(B, erp_height, erp_width, device=depth_cubemap.device)
    best_face = torch.zeros(B, erp_height, erp_width, dtype=torch.long, device=depth_cubemap.device)
    
    for face_idx in range(6):
        face_dir = face_dirs[face_idx]
        dot_product = (x * face_dir[0] + y * face_dir[1] + z * face_dir[2])
        
        if face_idx == 0:
            max_dot_product = dot_product.clone()
            best_face.fill_(0)
        else:
            mask = dot_product > max_dot_product
            max_dot_product = torch.where(mask, dot_product, max_dot_product)
            best_face = torch.where(mask, torch.full_like(best_face, face_idx), best_face)
    
    # 现在为每个面处理对应的像素
    for face_idx in range(6):
        face_mask = (best_face == face_idx)
        
        if not face_mask.any():
            continue  # 如果没有像素属于这个面，跳过
        
        # 将3D坐标投影到当前立方体贴图面
       # 侧面保持你现在的（关键是 v 用 -z/denom）
        if face_idx == 0:      # +X (front)
            u = ( y / x + 1) / 2 * W
            v = (-z / x + 1) / 2 * H
        elif face_idx == 1:    # +Y (right)
            u = (-x / y + 1) / 2 * W
            v = (-z / y + 1) / 2 * H
        elif face_idx == 2:    # -Y (left)
            u = ( x / -y + 1) / 2 * W
            v = (-z / -y + 1) / 2 * H
        elif face_idx == 3:    # -X (back)
            u = (-y / -x + 1) / 2 * W
            v = (-z / -x + 1) / 2 * H

        # 顶/底：在你原来基础上**旋转 90°**来对齐侧面
        elif face_idx == 4:    # +Z (up) —— 旋转 +90°
            # 原先若是 u = x/z, v =  y/z；现在换成：
            u = (  y / z + 1) / 2 * W   # u 用 +y/z
            v = (  x / z + 1) / 2 * H   # v 用 -x/z 以统一“向上=减小 v̄”
        else:                  # -Z (down) —— 旋转 -90°
            # 原先若是 u = x/-z, v = y/-z；现在换成：
            u = ( y / -z + 1) / 2 * W   # 等价于 y/(-z)
            v = (  -x / -z + 1) / 2 * H   # 等价于 -x/(-z)
        
        # 确保坐标在有效范围内
        u = torch.clamp(u, 0, W - 1)
        v = torch.clamp(v, 0, H - 1)
        
        # 使用双线性插值从立方体贴图中采样深度值
        u0 = torch.floor(u).long()
        u1 = torch.clamp(u0 + 1, 0, W - 1)
        v0 = torch.floor(v).long()
        v1 = torch.clamp(v0 + 1, 0, H - 1)
        
        # 计算插值权重
        wu = u - u0.float()
        wv = v - v0.float()
        
        # 双线性插值
        # 使用gather操作来避免高级索引的形状问题
        B_idx = torch.arange(B, device=depth_cubemap.device).view(B, 1, 1)
        F_idx = torch.full((B, erp_height, erp_width), face_idx, device=depth_cubemap.device)
        
        depth_00 = depth_cubemap[B_idx, F_idx, v0, u0]
        depth_01 = depth_cubemap[B_idx, F_idx, v0, u1]
        depth_10 = depth_cubemap[B_idx, F_idx, v1, u0]
        depth_11 = depth_cubemap[B_idx, F_idx, v1, u1]
        
        depth_interp = (depth_00 * (1 - wu) * (1 - wv) +
                       depth_01 * wu * (1 - wv) +
                       depth_10 * (1 - wu) * wv +
                       depth_11 * wu * wv)
        
        # 只更新当前面负责的像素
        erp_depth = torch.where(face_mask, depth_interp, erp_depth)
    
    return erp_depth.unsqueeze(1)  # (B, 1, H, W)


def process_erp_depth(erp_depth, depth_min=0.3, depth_max=24.0, noise_std=0.02, pool_size=4):
    """
    对ERP全景深度图进行数值裁剪和下采样处理
    
    Args:
        erp_depth: (B, 1, H, W) ERP格式的全景深度图
        depth_min: 深度最小值
        depth_max: 深度最大值
        noise_std: 噪声标准差
        pool_size: 下采样池化大小
    
    Returns:
        processed_depth: (B, 1, H//pool_size, W//pool_size) 处理后的深度图
    """
    # 数值裁剪和归一化
    processed_depth = 3.0 / erp_depth.clamp(depth_min, depth_max) - 0.6
    
    # 添加噪声
    # processed_depth = processed_depth + torch.randn_like(processed_depth) * noise_std
    
    # 下采样
    processed_depth = F.max_pool2d(processed_depth, pool_size, pool_size)
    
    return processed_depth


def cubemap_to_erp_pipeline(depth_cubemap, erp_height=64, erp_width=128, 
                           depth_min=0.3, depth_max=24.0, noise_std=0.02, pool_size=1):
    """
    完整的立方体贴图到ERP处理流水线
    
    Args:
        depth_cubemap: (B, 6, H, W) 立方体贴图深度图
        erp_height: ERP图像高度
        erp_width: ERP图像宽度
        depth_min: 深度最小值
        depth_max: 深度最大值
        noise_std: 噪声标准差
        pool_size: 下采样池化大小
    
    Returns:
        processed_erp: (B, 1, erp_height//pool_size, erp_width//pool_size) 处理后的ERP深度图
    """
    # 转换为ERP格式
    erp_depth = depth_to_erp_panorama(depth_cubemap, erp_height, erp_width)
    
    # 数值裁剪和下采样
    processed_erp = process_erp_depth(erp_depth, depth_min, depth_max, noise_std, pool_size)
    
    return processed_erp


def visualize_cubemap_and_erp(depth_cubemap, erp_depth, save_path=None, batch_idx=0):
    """
    可视化立方体贴图和ERP深度图的一致性
    
    Args:
        depth_cubemap: (B, 6, H, W) 立方体贴图深度图
        erp_depth: (B, 1, H, W) ERP深度图
        save_path: 保存路径，如果为None则显示图像
        batch_idx: 要可视化的批次索引
    """
    # 提取指定批次的数据
    cubemap = depth_cubemap[batch_idx].cpu().numpy()  # (6, H, W)
    erp = erp_depth[batch_idx, 0].cpu().numpy()  # (H, W)
    
    # 确保ERP是2D数组
    if erp.ndim > 2:
        erp = erp.squeeze()
    
    # 创建子图
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'立方体贴图深度图 vs ERP深度图 (Batch {batch_idx})', fontsize=16)
    
    # 立方体贴图的6个面
    face_names = ['Front (+X)', 'Right (-Y)', 'Left (+Y)', 'Back (-X)', 'Up (+Z)', 'Down (-Z)']
    
    for i in range(6):
        row = i // 3
        col = i % 3
        
        im = axes[row, col].imshow(cubemap[i], cmap='viridis', aspect='equal')
        axes[row, col].set_title(f'{face_names[i]}\nRange: [{cubemap[i].min():.2f}, {cubemap[i].max():.2f}]')
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    
    # ERP深度图
    im_erp = axes[2, 1].imshow(erp, cmap='viridis', aspect='equal')
    axes[2, 1].set_title(f'ERP深度图\nRange: [{erp.min():.2f}, {erp.max():.2f}]')
    axes[2, 1].axis('off')
    plt.colorbar(im_erp, ax=axes[2, 1], fraction=0.046, pad=0.04)
    
    # 隐藏最后一个子图
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化图像已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_depth_consistency(depth_cubemap, erp_depth, batch_idx=0):
    """
    分析立方体贴图和ERP深度图的一致性
    
    Args:
        depth_cubemap: (B, 6, H, W) 立方体贴图深度图
        erp_depth: (B, 1, H, W) ERP深度图
        batch_idx: 要分析的批次索引
    """
    print(f"\n=== 深度一致性分析 (Batch {batch_idx}) ===")
    
    # 提取数据
    cubemap = depth_cubemap[batch_idx]  # (6, H, W)
    erp = erp_depth[batch_idx, 0]  # (H, W)
    
    # 确保ERP是2D张量
    if erp.ndim > 2:
        erp = erp.squeeze()
    
    # 立方体贴图统计
    print("立方体贴图深度统计:")
    face_names = ['Front (+X)', 'Right (-Y)', 'Left (+Y)', 'Back (-X)', 'Up (+Z)', 'Down (-Z)']
    for i, name in enumerate(face_names):
        face_depth = cubemap[i]
        print(f"  {name}: 范围 [{face_depth.min():.3f}, {face_depth.max():.3f}], "
              f"均值 {face_depth.mean():.3f}, 标准差 {face_depth.std():.3f}")
    
    # ERP统计
    print(f"\nERP深度统计:")
    print(f"  范围: [{erp.min():.3f}, {erp.max():.3f}]")
    print(f"  均值: {erp.mean():.3f}")
    print(f"  标准差: {erp.std():.3f}")
    
    # 整体一致性分析
    cubemap_all = cubemap.flatten()
    erp_all = erp.flatten()
    
    print(f"\n整体一致性:")
    print(f"  立方体贴图总范围: [{cubemap_all.min():.3f}, {cubemap_all.max():.3f}]")
    print(f"  ERP总范围: [{erp_all.min():.3f}, {erp_all.max():.3f}]")
    print(f"  范围重叠度: {min(cubemap_all.max(), erp_all.max()) - max(cubemap_all.min(), erp_all.min()):.3f}")


def test_erp_conversion_with_visualization():
    """
    测试ERP转换并可视化结果
    """
    print("开始ERP转换测试和可视化...")
    
    # 创建测试数据
    B, C, H, W = 1, 6, 48, 64  # 改为批次大小1进行测试
    # 创建固定深度值便于分析对比
    depth_cubemap = torch.zeros(B, C, H, W)
    for b in range(B):
        for c in range(C):
            # 为每个面设置固定的深度值
            if c == 0:  # front
                depth_cubemap[b, c] = 5.0
            elif c == 1:  # right
                depth_cubemap[b, c] = 7.0
            elif c == 2:  # left
                depth_cubemap[b, c] = 9.0
            elif c == 3:  # back
                depth_cubemap[b, c] = 11.0
            elif c == 4:  # up
                depth_cubemap[b, c] = 13.0
            else:  # down
                depth_cubemap[b, c] = 15.0
    
    print(f"输入立方体贴图形状: {depth_cubemap.shape}")
    
    # 转换为ERP
    erp_depth = depth_to_erp_panorama(depth_cubemap, erp_height=64, erp_width=128)
    print(f"ERP深度图形状: {erp_depth.shape}")
    
    # 分析一致性
    analyze_depth_consistency(depth_cubemap, erp_depth, batch_idx=0)
    
    # 可视化
    visualize_cubemap_and_erp(depth_cubemap, erp_depth, save_path='cubemap_erp_comparison.png', batch_idx=0)
    
    # 测试处理后的结果
    processed_erp = process_erp_depth(erp_depth, depth_min=0.3, depth_max=24.0, noise_std=0.02, pool_size=4)
    print(f"处理后ERP形状: {processed_erp.shape}")
    
    # 可视化处理后的结果
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 原始ERP
    im1 = axes[0].imshow(erp_depth[0, 0].cpu().numpy(), cmap='viridis', aspect='equal')
    axes[0].set_title('原始ERP深度图')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 处理后的ERP
    im2 = axes[1].imshow(processed_erp[0, 0].cpu().numpy(), cmap='viridis', aspect='equal')
    axes[1].set_title('处理后ERP深度图')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('erp_processing_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ ERP转换测试和可视化完成!")
    print("生成的图像文件:")
    print("  - cubemap_erp_comparison.png: 立方体贴图与ERP对比")
    print("  - erp_processing_comparison.png: ERP处理前后对比")





def simple_cubemap_to_erp_rgb(cubemap: torch.Tensor, erp_height=224, erp_width=448):
    """
    将立方体贴图（RGB）拼接为 equirectangular（ERP）图。
    支持 cubemap 形状:
      - (6, H, W, 3)  通道在最后
      - (6, 3, H, W)  通道在前
    返回: (1, 3, erp_height, erp_width)  (NCHW)
    
    立方体面次序（必须与输入一致）:
      [front(+X), right(+Y), left(-Y), back(-X), up(+Z), down(-Z)]
    """
    import math

    assert cubemap.dim() == 4, f"expect 4D tensor, got {cubemap.shape}"

    # 统一成 face_tensor: (6, H, W, 3)
    if cubemap.shape[-1] == 3:          # (6, H, W, 3)
        face_tensor = cubemap
        Hc, Wc = face_tensor.shape[1], face_tensor.shape[2]
    elif cubemap.shape[1] == 3:         # (6, 3, H, W) -> (6, H, W, 3)
        face_tensor = cubemap.permute(0, 2, 3, 1).contiguous()
        Hc, Wc = face_tensor.shape[1], face_tensor.shape[2]
    else:
        raise ValueError(f"unsupported cubemap shape: {cubemap.shape}; expected (6,H,W,3) or (6,3,H,W)")

    device = face_tensor.device
    dtype  = face_tensor.dtype

    # ---- ERP 网格（theta: [0,2π), phi: [-π/2, π/2]）----
    theta = torch.linspace(0, 2*math.pi, steps=erp_width, device=device)         # [0, 2π)
    phi   = torch.linspace(-math.pi/2, math.pi/2, steps=erp_height, device=device)  # [-π/2, π/2]
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='xy')

    # 球->笛卡尔（右手坐标）
    x = torch.cos(phi_grid) * torch.cos(theta_grid)
    y = torch.cos(phi_grid) * torch.sin(theta_grid)
    z = torch.sin(phi_grid)

    face_dirs = torch.tensor([
        [-1, 0, 0],   # front (I)
        [0, -1, 0],  # right (yaw_p)
        [0, 1, 0],   # left (yaw_n)
        [1, 0, 0],  # back (yaw_180)
        [0, 0, -1],   # up (pitch_up)
        [0, 0, 1],  # down (pitch_down)
    ], device=device, dtype=torch.float32)

    # # ---- 面朝向（与上面的 6 面顺序严格对应）----
    # face_dirs = torch.tensor([
    #     [ 1,  0,  0],   # front  +X
    #     [ 0,  1,  0],   # right  +Y
    #     [ 0, -1,  0],   # left   -Y
    #     [-1,  0,  0],   # back   -X
    #     [ 0,  0,  1],   # up     +Z
    #     [ 0,  0, -1],   # down   -Z
    # ], device=device, dtype=torch.float32)

    # 找每个 ERP 像素对应的面（最大点乘）
    # dot_maps: (6, H_erp, W_erp)
    dirs = torch.stack([x, y, z], dim=0).to(face_dirs.dtype)  # (3, H, W)
    dot_maps = (face_dirs @ dirs.view(3, -1)).view(6, erp_height, erp_width)
    best_face = torch.argmax(dot_maps, dim=0)  # (H_erp, W_erp)

    # 初始化 ERP RGB（H,W,3）
    erp_rgb = torch.zeros((erp_height, erp_width, 3), device=device, dtype=dtype)

    # 方便：常量
    eps = 1e-12
    Wc_f = float(Wc)
    Hc_f = float(Hc)

    # 按面映射最近邻采样
    for face_idx in range(6):
        face_mask = (best_face == face_idx)  # (H_erp, W_erp)
        if not face_mask.any():
            continue

        # 避免除零：对每个面的分母加 eps
        if face_idx == 0:      # front (+X)
            denom = x.sign() * torch.clamp(x.abs(), min=eps)
            u = ( y / denom + 1.0) * 0.5 * Wc_f
            v = ( -z / denom + 1.0) * 0.5 * Hc_f
        elif face_idx == 1:    # right (+Y)
            denom = y.sign() * torch.clamp(y.abs(), min=eps)
            u = (-x / denom + 1.0) * 0.5 * Wc_f
            v = ( -z / denom + 1.0) * 0.5 * Hc_f
        elif face_idx == 2:    # left (-Y)
            denom = (-y).sign() * torch.clamp((-y).abs(), min=eps)
            u = ( x / denom + 1.0) * 0.5 * Wc_f
            v = ( -z / denom + 1.0) * 0.5 * Hc_f
        elif face_idx == 3:    # back (-X)
            denom = (-x).sign() * torch.clamp((-x).abs(), min=eps)
            u = (-y / denom + 1.0) * 0.5 * Wc_f
            v = ( -z / denom + 1.0) * 0.5 * Hc_f
        elif face_idx == 4:  # up (+Z)
            denom = z.sign() * torch.clamp(z.abs(), min=eps)
            u = (  y / denom + 1.0) * 0.5 * Wc_f   # 注意用 x
            v = (  x / denom + 1.0) * 0.5 * Hc_f   # 注意用 y
        elif face_idx == 5:  # down (-Z)
            denom = (-z).sign() * torch.clamp((z).abs(), min=eps)
            u = (  y / denom + 1.0) * 0.5 * Wc_f
            v = (   -x / denom + 1.0) * 0.5 * Hc_f
        # clamp + 最近邻
        u = torch.clamp(u, 0, Wc_f - 1)
        v = torch.clamp(v, 0, Hc_f - 1)
        u_int = torch.round(u).long()
        v_int = torch.round(v).long()

        # （与你原实现一致）翻转面内 v 行索引
        # v_int = (Hc - 1) - v_int

        # 从该面取 (H_erp, W_erp, 3) 的颜色图
        # face_tensor[face_idx]: (Hc, Wc, 3)
        sampled = face_tensor[face_idx][v_int, u_int]  # broadcasting index => (H_erp,W_erp,3)

        # 只在本面的 mask 处覆盖
        erp_rgb = torch.where(face_mask.unsqueeze(-1), sampled, erp_rgb)

    # 返回 NCHW
    return erp_rgb.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)



if __name__ == "__main__":
    test_erp_conversion_with_visualization()
