import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from .utils import hash_state_dict_keys
# from .ucpe_attention import UcpeSelfAttention
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def convert_rt_to_viewmats(rt_matrix: torch.Tensor) -> torch.Tensor:
    """
    将B x F x 12的RT矩阵转换为B x F x 4 x 4的viewmats格式
    
    Args:
        rt_matrix: 形状为(B, F, 12)的RT矩阵，其中前9个元素是旋转矩阵的行向量排列，
                  后3个元素是平移向量
    
    Returns:
        viewmats: 形状为(B, F, 4, 4)的齐次变换矩阵
    """
    B, F, _ = rt_matrix.shape
    
    # 提取旋转部分 (B, F, 9) 和平移部分 (B, F, 3)
    rotation_flat = rt_matrix[..., :9]  # 前9个元素是旋转矩阵的行向量
    translation = rt_matrix[..., 9:]    # 后3个元素是平移向量
    
    # 重塑旋转部分为3x3矩阵 (B, F, 3, 3)
    rotation = rotation_flat.reshape(B, F, 3, 3)
    
    # 创建4x4齐次变换矩阵
    viewmats = torch.eye(4, device=rt_matrix.device, dtype=rt_matrix.dtype)
    viewmats = viewmats.reshape(1, 1, 4, 4).repeat(B, F, 1, 1)
    
    # 填充旋转部分 (3x3)
    viewmats[..., :3, :3] = rotation
    
    # 填充平移部分
    viewmats[..., :3, 3] = translation
    
    # 最后一行保持 [0, 0, 0, 1]
    viewmats[..., 3, :] = torch.tensor([0, 0, 0, 1], 
                                      device=rt_matrix.device, 
                                      dtype=rt_matrix.dtype)
    
    return viewmats

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)

########################
def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    f_freqs_cis_reverse = precompute_freqs_cis_reverse(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis, f_freqs_cis_reverse


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def precompute_freqs_cis_reverse(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute with reverse phase direction
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), -freqs)  # 关键修改：添加负号
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis
########################

# def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
#     # 3d rope precompute
#     f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
#     h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
#     w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
#     return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)

def rope_apply_cam(x, freqs, num_heads, phase=None):
    if freqs is None and phase is None:
        return x
    
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    
    if phase is not None:
        phase = rearrange(phase, 'b s (n d) -> b s n d', n=num_heads)
        frame_freqs_dim = x.shape[-1] // 2 - phase.shape[-1]
        phase = torch.nn.functional.pad(phase, (frame_freqs_dim, 0), value=0.0)
        phase_shift = torch.exp(1j * phase)
        freqs = freqs.unsqueeze(0) * phase_shift if freqs is not None else phase_shift

    x_out = torch.view_as_complex(x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2).to(x.dtype)

    return x_out

def gather_axis_freqs(freqs_cis: torch.Tensor, freqs_cis_reverse: torch.Tensor, positions: torch.Tensor):
    pos = positions.to(freqs_cis.device)
    pos_abs = pos.abs().long().clamp(max=freqs_cis.shape[0] - 1)
    
    # 分别从正序列和负序列中 gather
    base_pos = freqs_cis[pos_abs]      # 正序列的值
    base_neg = freqs_cis_reverse[pos_abs]  # 负序列的值
    
    # 创建掩码：正位置用正序列，负位置用负序列
    neg_mask = (pos < 0).unsqueeze(-1)
    return torch.where(neg_mask, base_neg, base_pos)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SparseFrameAttentionModule(nn.Module):
    def __init__(self, num_heads, num_frames=40, frame_hw=30 * 52, top_k=40, frame_chunk_size=None):
        """
        Args:
            num_heads: 注意力头数
            num_frames: 帧数
            frame_hw: 每帧的空间尺寸 (H * W)
            top_k: 每帧选择的top-k个相似帧
            frame_chunk_size: 如果设置，将帧分批处理以节省显存。None表示不分组。
        """
        super().__init__()
        self.num_heads = num_heads 
        self.num_frames = num_frames
        self.frame_hw = frame_hw
        self.top_k = top_k
        self.total_selected = top_k + 1
        self.frame_chunk_size = frame_chunk_size  # 分批处理帧，用于显存优化
    
    def forward(self, q, k, v, similarity):
        B, seq_len, dim = q.shape
        F, HW, H = self.num_frames, self.frame_hw, self.num_heads
        head_dim = dim // H
        
        # 步骤1: 帧划分（保持原样）

        q_frames = rearrange(q, 'b (f hw) d -> b f hw d', f=F, hw=HW)
        k_frames = rearrange(k, 'b (f hw) d -> b f hw d', f=F, hw=HW)
        v_frames = rearrange(v, 'b (f hw) d -> b f hw d', f=F, hw=HW)
        
        # 步骤3: Top-k 选择（排除自身）
        mask = torch.eye(F, device=similarity.device, dtype=similarity.dtype) * (-1000000)  # [F, F]
        similarity_masked = similarity + mask.unsqueeze(0)
        topk_values, topk_indices = torch.topk(similarity_masked, k=self.top_k, dim=-1)  # [B, F, top_k]
        
        # 步骤4: 准备索引（包含自身帧）
        self_indices = torch.arange(F, device=q.device, dtype=torch.long).unsqueeze(0).unsqueeze(-1).expand(B, F, 1)
        selected_indices = torch.cat([self_indices, topk_indices], dim=-1)  # [B, F, total_selected]
        
        # 步骤5: 预分配输出张量，避免使用列表+stack
        output = torch.zeros(B, F, HW, dim, device=q.device, dtype=q.dtype)
        
        # 步骤6: 预先重组 Q 为多头格式（只需一次）
        q_frames_mh = rearrange(q_frames, 'b f hw (h d) -> b f h hw d', h=H)  # [B, F, H, HW, head_dim]
        
        frame_indices = range(F)
        self._process_frame_chunk(output, q_frames_mh, k_frames, v_frames,
                                    selected_indices, frame_indices, B, H, HW, dim)
        
        # 步骤8: 合并所有帧的输出
        x = rearrange(output, 'b f hw d -> b (f hw) d')  # [B, (F * HW), dim]
        
        # 注意：虽然 topk 的索引选择本身不可微，但梯度可以通过相似度矩阵反向传播
        # 梯度路径：loss -> output -> attention -> k_candidates/v_candidates -> k_frames/v_frames -> similarity -> q_frame_repr/k_frame_repr -> q/k
        # 这样模型可以学习通过调整 Q 和 K 来改变相似度，从而影响帧的选择
        
        return x
    
    def _process_frame_chunk(self, output, q_frames_mh, k_frames, v_frames, 
                            selected_indices, frame_indices, B, H, HW, dim):
        """处理一批帧的注意力计算"""
        for frame_idx in frame_indices:
            # 当前帧的 Q（多头格式）
            q_frame = q_frames_mh[:, frame_idx]  # [B, H, HW, head_dim]
            
            # 按需提取当前帧的候选KV（避免预创建完整的 selected_k/selected_v）
            # selected_indices[:, frame_idx]: [B, total_selected]
            frame_selected_indices = selected_indices[:, frame_idx]  # [B, total_selected]
            
            # 使用高级索引高效提取候选KV（只针对当前帧，显存友好）
            batch_indices = torch.arange(B, device=k_frames.device).unsqueeze(1).expand(-1, self.total_selected)  # [B, total_selected]
            k_candidates = k_frames[batch_indices, frame_selected_indices]  # [B, total_selected, HW, dim]
            v_candidates = v_frames[batch_indices, frame_selected_indices]  # [B, total_selected, HW, dim]
            
            # 重组为多头格式（按需，不存储完整张量）
            k_candidates_mh = rearrange(k_candidates, 'b k hw (h d) -> b k h hw d', h=H)  # [B, total_selected, H, HW, head_dim]
            v_candidates_mh = rearrange(v_candidates, 'b k hw (h d) -> b k h hw d', h=H)  # [B, total_selected, H, HW, head_dim]
            
            # 将候选帧的KV拼接为长序列
            k_flat = rearrange(k_candidates_mh, 'b k h hw d -> b (k hw) h d')  # [B, total_selected*HW, H, head_dim]
            v_flat = rearrange(v_candidates_mh, 'b k h hw d -> b (k hw) h d')  # [B, total_selected*HW, H, head_dim]
            
            # 准备注意力计算的输入格式
            q_flat_for_attn = rearrange(q_frame, 'b h hw d -> b hw (h d)')  # [B, HW, H * head_dim]
            k_flat_for_attn = rearrange(k_flat, 'b s h d -> b s (h d)')  # [B, total_selected*HW, H * head_dim]
            v_flat_for_attn = rearrange(v_flat, 'b s h d -> b s (h d)')  # [B, total_selected*HW, H * head_dim]
            
            # 计算注意力
            attn_out = flash_attention(
                q=q_flat_for_attn, k=k_flat_for_attn, v=v_flat_for_attn,
                num_heads=self.num_heads
            )  # [B, HW, H * head_dim] = [B, HW, dim]
            
            # 直接写入预分配的输出张量
            output[:, frame_idx] = attn_out
            
            # 清理当前循环的中间变量（显存优化）
            del k_candidates, v_candidates, k_candidates_mh, v_candidates_mh
            del k_flat, v_flat, q_flat_for_attn, k_flat_for_attn, v_flat_for_attn

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, 
                attention_type: str = "standard",  # "standard" or "sparse_frame"
                sparse_frame_args: dict = None, change_sparse: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attention_type = attention_type
        
        # 共享的线性层
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        if change_sparse:
            self.attention_type = "sparse_frame"
        else:
            self.attention_type = "standard"
        # 根据类型选择注意力模块
        if self.attention_type == "standard":
            self.attn = AttentionModule(num_heads)
        elif self.attention_type == "sparse_frame":
            sparse_args = sparse_frame_args or {}
            self.attn = SparseFrameAttentionModule(
                num_heads=num_heads,
                num_frames=sparse_args.get('num_frames', 40),
                frame_hw=sparse_args.get('frame_hw', 30 * 52),
                top_k=sparse_args.get('top_k', 9),
                frame_chunk_size=sparse_args.get('frame_chunk_size', None)  # 显存优化：分批处理帧
            )
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")
        
        self._q_last = None
        self._k_last = None
        self.cache_qk = False

    def enable_cache(self):
        self.cache_qk = True

    def disable_cache(self):
        self.cache_qk = False
        self._q_last = None
        self._k_last = None

    def forward(self, x, freqs, cam_emb=None):

        # if self.change_sparse:
        #     q_frames = rearrange(q, 'b (f h w) d -> b f h w d', f=42, h=30, w=52)
        #     k_frames = rearrange(k, 'b (f h w) d -> b f h w d', f=42, h=30, w=52)
        #     pool_size = 4

        #     # 重新排列维度以进行池化 [b, f, h, w, d] -> [b*f*d, 1, h, w]
        #     b, f, h, w, d = q_frames.shape
        #     q_reshaped = q_frames.permute(0, 1, 4, 2, 3).reshape(b * f * d, 1, h, w)
        #     k_reshaped = k_frames.permute(0, 1, 4, 2, 3).reshape(b * f * d, 1, h, w)

        #     # 4x4平均池化
        #     q_pooled = F.avg_pool2d(q_reshaped, kernel_size=pool_size, stride=pool_size)
        #     k_pooled = F.avg_pool2d(k_reshaped, kernel_size=pool_size, stride=pool_size)

        #     # 恢复形状 [b, f, d, h/pool_size, w/pool_size] -> [b, f, h/pool_size, w/pool_size, d]
        #     new_h, new_w = h // pool_size, w // pool_size
        #     q_pooled = q_pooled.reshape(b, f, d, new_h, new_w).permute(0, 1, 3, 4, 2)
        #     k_pooled = k_pooled.reshape(b, f, d, new_h, new_w).permute(0, 1, 3, 4, 2)

        #     # Flatten为 [b, f * new_h * new_w, d]
        #     # print("yyyyyy\n")
        #     q_frame_repr = q_pooled.reshape(b, f, d * new_h * new_w)
        #     k_frame_repr = k_pooled.reshape(b, f, d * new_h * new_w)
        if cam_emb is None or self.use_UCPE:
        
            if self.attention_type == "sparse_frame":
                q = self.norm_q(self.q(x))
                k = self.norm_k(self.k(x))
                v = self.v(x)
                q_frames = rearrange(q, 'b (f h w) d -> b f h w d', f=40, h=30, w=52)
                k_frames = rearrange(k, 'b (f h w) d -> b f h w d', f=40, h=30, w=52)
                q_frame_repr = q_frames.mean(dim=3).mean(dim=2)  # [B, F, dim]
                k_frame_repr = k_frames.mean(dim=3).mean(dim=2)  # [B, F, dim]
                similarity = torch.matmul(q_frame_repr, k_frame_repr.transpose(-2, -1))  # [B, F, F]
                q = rope_apply(q, freqs, self.num_heads)
                k = rope_apply(k, freqs, self.num_heads)
                x = self.attn(q, k, v, similarity)
                
            # x = self.attn(q, k, v, similarity)
            # x = self.o(x)
            else:
                q = self.norm_q(self.q(x))
                k = self.norm_k(self.k(x))
                v = self.v(x)
                q = rope_apply(q, freqs, self.num_heads)
                k = rope_apply(k, freqs, self.num_heads)
                x = self.attn(q, k, v)

            x = self.o(x)

        else:
            rope_phase_qk = self.rope_phase_qk(cam_emb)
            rope_phase_vo = self.rope_phase_vo(cam_emb)
            q = self.norm_q(self.q(x))
            k = self.norm_k(self.k(x))
            v = self.v(x)
            q = rope_apply_cam(q, freqs, self.num_heads, rope_phase_qk)
            k = rope_apply_cam(k, freqs, self.num_heads, rope_phase_qk)
            v = rope_apply_cam(v, None, self.num_heads, -rope_phase_vo)
            x = self.attn(q, k, v)
            x = self.o(x)
            x = rope_apply_cam(x, None, self.num_heads, rope_phase_vo)

        return x




class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        # self.moc = MoCBlock(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, context, cam_emb_tgt, cam_emb_con, cam_emb, t_mod, freqs): 
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        if cam_emb is None:
            if self.add_ref:
                input_x = rearrange(input_x, 'b (f h w) c -> b c f h w ', f=41, h=30, w=52).contiguous()
            else:
                input_x = rearrange(input_x, 'b (f h w) c -> b c f h w ', f=40, h=30, w=52).contiguous()
                
            # cam_emb_tgt = cam_emb_tgt.to(torch.bfloat16)
            # cam_emb_con = cam_emb_con.to(torch.bfloat16)
            cam_emb_tgt = self.cam_encoder_tgt(cam_emb_tgt)
            # cam_emb_tgt = cam_emb_tgt.repeat(1, 1, 1)
            cam_emb_tgt = cam_emb_tgt.unsqueeze(2).unsqueeze(3).repeat(1, 1, 30, 52, 1).permute(0, 4, 1, 2, 3)
            if self.add_ref:
                input_x[:, :, 21:, ...] = input_x[:, :, 21:, ...] + cam_emb_tgt
            else:
                input_x[:, :, 20:, ...] = input_x[:, :, 20:, ...] + cam_emb_tgt

            cam_emb_con = self.cam_encoder_con(cam_emb_con)
            # cam_emb_con = cam_emb_con.repeat(1, 1, 1)       
            cam_emb_con = cam_emb_con.unsqueeze(2).unsqueeze(3).repeat(1, 1, 30, 52, 1).permute(0, 4, 1, 2, 3)
            if self.add_ref:
                input_x[:, :, 1:21, ...] = input_x[:, :, 1:21, ...] + cam_emb_con
            else:
                input_x[:, :, :20, ...] = input_x[:, :, :20, ...] + cam_emb_con

            input_x = rearrange(input_x, 'b c f h w -> b (f h w) c').contiguous()

            # input_x = input_x + cam_emb_tgt
            x = x + gate_msa * self.projector(self.self_attn(input_x, freqs))
        else:
            if not self.use_UCPE:
                x = x + gate_msa * self.projector(self.self_attn(input_x, freqs, cam_emb))
            else:
                residual = self.self_attn(input_x, freqs, cam_emb)
                input_x = rearrange(input_x, 'b (f h w) c -> b c f h w ', f=40, h=30, w=52).contiguous()
                cam_emb_tgt = self.cam_encoder_tgt(cam_emb_tgt)
                cam_emb_tgt = cam_emb_tgt.unsqueeze(2).unsqueeze(3).repeat(1, 1, 30, 52, 1).permute(0, 4, 1, 2, 3) 
                input_x[:, :, 20:, ...] = input_x[:, :, 20:, ...] + cam_emb_tgt
                cam_emb_con = self.cam_encoder_con(cam_emb_con)
                cam_emb_con = cam_emb_con.unsqueeze(2).unsqueeze(3).repeat(1, 1, 30, 52, 1).permute(0, 4, 1, 2, 3)
                input_x[:, :, :20, ...] = input_x[:, :, :20, ...] + cam_emb_con
                input_x = rearrange(input_x, 'b c f h w -> b (f h w) c').contiguous()
                residual += self.cam_self_attn(input_x, cam_emb)
                x = x + gate_msa * residual
        
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(input_x)
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int], 
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim)  # clip_feature_dim = 1280

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.camera_adapter is not None and control_camera_latents_input is not None:
            control_camera_latents_input = control_camera_latents_input.to(x.device)
            x_camera = self.camera_adapter(control_camera_latents_input)
            x = x + x_camera
            grid_size = x.shape[2:]
            x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
            cam_emb = rearrange(x_camera, 'b c f h w -> b (f h w) c').contiguous()
            return x, grid_size, cam_emb  # x, grid_size: (f, h, w)

        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                cam_emb_tgt: torch.Tensor,
                cam_emb_con: torch.Tensor,
                context: torch.Tensor,
                cond_plucker_embedding: Optional[torch.Tensor] = None,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        cam_emb = None
        if cond_plucker_embedding is not None and self.use_PRoPE:
            x, (f, h, w), cam_emb = self.patchify(x, cond_plucker_embedding)
            # f = f // 2
            freqs = torch.cat([
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
            # freqs = freqs.repeat(2, 1, 1)

        else:
            x, (f, h, w) = self.patchify(x)
    
            freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        #######

        if self.use_UCPE:
            cam_emb = convert_rt_to_viewmats(cond_plucker_embedding)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, cam_emb_tgt, cam_emb_con, cam_emb, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, cam_emb_tgt, cam_emb_con, cam_emb, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, cam_emb_tgt, cam_emb_con, cam_emb, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
    
    
class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config
    
    def from_civitai(self, state_dict):
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        else:
            config = {}
        return state_dict, config