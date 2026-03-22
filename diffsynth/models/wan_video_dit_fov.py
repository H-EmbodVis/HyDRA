import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List
from einops import rearrange
from .utils import hash_state_dict_keys
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


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


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
    def __init__(self, num_heads, num_frames=42, frame_hw=30 * 52, top_k=5, frame_chunk_size=None, chunk_size=5, target_frame_count=21):
        """
        Args:
            num_heads: 注意力头数
            num_frames: 帧数
            frame_hw: 每帧的空间尺寸 (H * W)
            top_k: 从参考帧中选择的top-k数量（默认5）
            frame_chunk_size: 如果设置，将帧分批处理以节省显存。None表示不分组。
            chunk_size: 每个chunk包含的帧数，用于保持连续性（默认5）
            target_frame_count: 待去噪帧的数量（默认21，对应index 0-20）
        """
        super().__init__()
        self.num_heads = num_heads 
        self.num_frames = num_frames
        self.frame_hw = frame_hw
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.target_frame_count = target_frame_count
        self.reference_start = target_frame_count
        self.total_selected = chunk_size + top_k
        self.frame_chunk_size = frame_chunk_size  # 分批处理帧，用于显存优化
    
    def forward(self, q, k, v, camera_indices):
        B, seq_len, dim = q.shape
        F, HW, H = self.num_frames, self.frame_hw, self.num_heads
        head_dim = dim // H
        
        # 步骤1: 帧划分（保持原样）
        q_frames = rearrange(q, 'b (f hw) d -> b f hw d', f=F, hw=HW)
        k_frames = rearrange(k, 'b (f hw) d -> b f hw d', f=F, hw=HW)
        v_frames = rearrange(v, 'b (f hw) d -> b f hw d', f=F, hw=HW)
        
        # 步骤2: 计算帧级相似度
        q_frame_repr = q_frames.mean(dim=2)  # [B, F, dim]
        k_frame_repr = k_frames.mean(dim=2)  # [B, F, dim]
        similarity = torch.matmul(q_frame_repr, k_frame_repr.transpose(-2, -1))  # [B, F, F]
        
        # 步骤3: 选取候选帧
        selected_indices = torch.zeros(B, F, self.total_selected, device=q.device, dtype=torch.long)
        target_end = min(self.target_frame_count, F)
        reference_start = min(self.reference_start, F)
        reference_count = max(F - reference_start, 0)
        
        def build_chunk_indices(frame_idx: int):
            if target_end <= 0:
                return torch.zeros(self.chunk_size, device=q.device, dtype=torch.long)
            clamped_idx = max(0, min(frame_idx, max(target_end - 1, 0)))
            half_chunk = self.chunk_size // 2
            chunk_start = max(0, clamped_idx - half_chunk)
            chunk_end = min(target_end, chunk_start + self.chunk_size)
            if chunk_end - chunk_start < self.chunk_size:
                chunk_start = max(0, chunk_end - self.chunk_size)
            chunk_indices = torch.arange(chunk_start, chunk_end, device=q.device, dtype=torch.long)
            if chunk_indices.numel() == 0:
                chunk_indices = torch.zeros(1, device=q.device, dtype=torch.long)
            if frame_idx < target_end and not torch.any(chunk_indices == frame_idx):
                chunk_indices = chunk_indices.clone()
                chunk_indices[0] = frame_idx
            if chunk_indices.numel() < self.chunk_size:
                pad_value = chunk_indices[-1]
                padding = torch.full((self.chunk_size - chunk_indices.numel(),), pad_value, device=q.device, dtype=torch.long)
                chunk_indices = torch.cat([chunk_indices, padding], dim=0)
            elif chunk_indices.numel() > self.chunk_size:
                chunk_indices = chunk_indices[:self.chunk_size]
            return chunk_indices
        
        for frame_idx in range(F):
            chunk_indices = build_chunk_indices(frame_idx)
            chunk_indices_expanded = chunk_indices.unsqueeze(0).expand(B, -1)  # [B, chunk_size]
            
            topk_indices = torch.full((B, self.top_k), chunk_indices[-1], device=q.device, dtype=torch.long) if self.top_k > 0 else torch.empty(B, 0, device=q.device, dtype=torch.long)
            if self.top_k > 0 and reference_count > 0:
                ref_similarity = similarity[:, frame_idx, reference_start:]
                ref_len = ref_similarity.shape[-1]
                if ref_len > 0:
                    k_top = min(self.top_k, ref_len)
                    topk_rel = torch.topk(ref_similarity, k=k_top, dim=-1).indices
                    topk_abs = topk_rel + reference_start
                    if k_top < self.top_k:
                        pad_value = topk_abs[:, -1:].expand(B, self.top_k - k_top)
                        topk_abs = torch.cat([topk_abs, pad_value], dim=-1)
                    topk_indices = topk_abs

            if frame_idx < 21:
                fov_indices = torch.tensor(camera_indices[frame_idx]).unsqueeze(0).expand(B, -1).to(chunk_indices_expanded.device)
            # else:   
                # fov_indices = torch.zeros(B, 5, device=q.device, dtype=torch.long)
                combined_indices = torch.cat([chunk_indices_expanded, fov_indices], dim=-1) if self.top_k > 0 else chunk_indices_expanded
            else:
                combined_indices = torch.cat([chunk_indices_expanded, topk_indices], dim=-1) if self.top_k > 0 else chunk_indices_expanded

            selected_indices[:, frame_idx, :] = combined_indices
        
        # 步骤4: 预分配输出张量，避免使用列表+stack
        output = torch.zeros(B, F, HW, dim, device=q.device, dtype=q.dtype)
        
        # 步骤5: 预先重组 Q 为多头格式（只需一次）
        q_frames_mh = rearrange(q_frames, 'b f hw (h d) -> b f h hw d', h=H)  # [B, F, H, HW, head_dim]
        
        frame_indices = range(F)
        self._process_frame_chunk(output, q_frames_mh, k_frames, v_frames,
                                    selected_indices, frame_indices, B, H, HW, dim)
        
        # 步骤6: 合并所有帧的输出
        x = rearrange(output, 'b f hw d -> b (f hw) d')  # [B, (F * HW), dim]
        
        # 注意：虽然top-k的索引选择本身不可微，但梯度可以通过相似度矩阵反向传播
        # 梯度路径：loss -> output -> attention -> k_candidates/v_candidates -> k_frames/v_frames -> similarity -> q_frame_repr/k_frame_repr -> q/k
        
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
                 attention_type: str = "sparse_frame",  # "standard" or "sparse_frame"
                 sparse_frame_args: dict = None):
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
        self.attention_type = attention_type
        # 根据类型选择注意力模块
        if attention_type == "standard":
            self.attn = AttentionModule(num_heads)
        elif attention_type == "sparse_frame":
            sparse_args = sparse_frame_args or {}
            self.attn = SparseFrameAttentionModule(
                num_heads=num_heads,
                num_frames=sparse_args.get('num_frames', 42),
                frame_hw=sparse_args.get('frame_hw', 30 * 52),
                top_k=sparse_args.get('top_k', 5),  # 从参考帧中选择top-k，默认5
                frame_chunk_size=sparse_args.get('frame_chunk_size', None),  # 显存优化：分批处理帧
                chunk_size=sparse_args.get('chunk_size', 5),  # chunk大小，默认5
                target_frame_count=sparse_args.get('target_frame_count', 21),  # 待去噪帧数量
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

    def forward(self, x, freqs, camera_indices=None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        
        if self.cache_qk:
            self._q_last = q.detach()
            self._k_last = k.detach()
        else:
            self._q_last = None
            self._k_last = None
        if self.attention_type == "standard":
            x = self.attn(q, k, v)
        else:
            x = self.attn(q, k, v, camera_indices)
        x = self.o(x)
        return x

    @torch.no_grad()
    def save_attention(self, batch_idx=0, head_idx=None, max_tokens: int = 2048):
        """
        返回帧-帧注意力矩阵（CPU numpy, float32）。
        - 若 head_idx 为 None：对所有 head 的注意力做平均（先做帧级聚合再逐 head softmax，并累加平均），低显存顺序处理。
        - 若 head_idx 为 int：仅返回该 head 的注意力。
        """
        q = self._q_last
        k = self._k_last
        assert q is not None and k is not None, 'q/k未缓存,请在forward前enable_cache并在forward后立刻调用!'
        b, s, nd = q.shape

        F_FIXED = 42
        H_FIXED = 30
        W_FIXED = 52
        HW = H_FIXED * W_FIXED

        def head_slice(x, h):
            start = h * self.head_dim
            end = start + self.head_dim
            return x[:, :, start:end]  # [b, s, d]

        def frames_from_qk(q_hd, k_hd):
            if s == F_FIXED * HW:
                q_5d = rearrange(q_hd, 'b (f hw) d -> b f hw d', f=F_FIXED, hw=HW)
                k_5d = rearrange(k_hd, 'b (f hw) d -> b f hw d', f=F_FIXED, hw=HW)
                q_f = q_5d.mean(dim=3).mean(dim=2)  # [b, f]
                k_f = k_5d.mean(dim=3).mean(dim=2)  # [b, f]
                del q_5d, k_5d
                return q_f, k_f
            # 回退：等间隔下采样到<=max_tokens，再在d上平均并池化到<=F_FIXED帧
            if s > max_tokens:
                stride = s // max_tokens + (1 if s % max_tokens != 0 else 0)
                q_sub = q_hd[:, ::stride]
                k_sub = k_hd[:, ::stride]
            else:
                q_sub = q_hd
                k_sub = k_hd
            q_token = q_sub.mean(dim=2)  # [b, s']
            k_token = k_sub.mean(dim=2)  # [b, s']
            s_prime = q_token.shape[1]
            if s_prime > F_FIXED:
                pool_stride = s_prime // F_FIXED
                pool_ks = pool_stride
                s_trim = (s_prime // pool_stride) * pool_stride
                q_tok_trim = q_token[:, :s_trim]
                k_tok_trim = k_token[:, :s_trim]
                q_f = q_tok_trim.unfold(dimension=1, size=pool_ks, step=pool_stride).mean(dim=-1)
                k_f = k_tok_trim.unfold(dimension=1, size=pool_ks, step=pool_stride).mean(dim=-1)
                del q_tok_trim, k_tok_trim
            else:
                q_f = q_token
                k_f = k_token
            del q_sub, k_sub, q_token, k_token
            return q_f, k_f

        if head_idx is None:
            # 对所有head做平均；顺序处理以节省显存
            attn_sum = None
            for h in range(self.num_heads):
                q_hd = head_slice(q, h)
                k_hd = head_slice(k, h)
                q_f, k_f = frames_from_qk(q_hd, k_hd)  # [b, f], [b, f]
                attn_score = q_f[:, :, None] * k_f[:, None, :]
                attn_weight = torch.softmax(attn_score, dim=-1).to(torch.float32)  # [b, f, f]
                attn_sum = attn_weight if attn_sum is None else (attn_sum + attn_weight)
                del q_hd, k_hd, q_f, k_f, attn_score, attn_weight
            attn_avg = (attn_sum / self.num_heads)
            attn_map = attn_avg[batch_idx].cpu().numpy()
            del attn_sum, attn_avg
        else:
            # 单head分支（兼容）
            q_hd = head_slice(q, head_idx)
            k_hd = head_slice(k, head_idx)
            q_f, k_f = frames_from_qk(q_hd, k_hd)
            attn_score = q_f[:, :, None] * k_f[:, None, :]
            attn_weight = torch.softmax(attn_score, dim=-1).to(torch.float32)
            attn_map = attn_weight[batch_idx].cpu().numpy()
            del q_hd, k_hd, q_f, k_f, attn_score, attn_weight

        # 清理缓存与显存
        self._q_last = None
        self._k_last = None
        self.cache_qk = False
        torch.cuda.empty_cache()
        return attn_map


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

    def forward(self, x, context, cam_emb_tgt, cam_emb_con, camera_indices, t_mod, freqs): 
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)

        input_x = rearrange(input_x, 'b (f h w) c -> b c f h w ', f=42, h=30, w=52).contiguous()

        cam_emb_tgt = self.cam_encoder_tgt(cam_emb_tgt)
        # cam_emb_tgt = cam_emb_tgt.repeat(1, 1, 1)
        cam_emb_tgt = cam_emb_tgt.unsqueeze(2).unsqueeze(3).repeat(1, 1, 30, 52, 1).permute(0, 4, 1, 2, 3)

        input_x[:, :, :21, ...] = input_x[:, :, :21, ...] + cam_emb_tgt

        cam_emb_con = self.cam_encoder_con(cam_emb_con)
        # cam_emb_con = cam_emb_con.repeat(1, 1, 1)       
        cam_emb_con = cam_emb_con.unsqueeze(2).unsqueeze(3).repeat(1, 1, 30, 52, 1).permute(0, 4, 1, 2, 3)

        input_x[:, :, 21:, ...] = input_x[:, :, 21:, ...] + cam_emb_con

        input_x = rearrange(input_x, 'b c f h w -> b (f h w) c').contiguous()

        # input_x = input_x + cam_emb_tgt
        if camera_indices == None:
            x = x + gate_msa * self.projector(self.self_attn(input_x, freqs))
        else:
            x = x + gate_msa * self.projector(self.self_attn(input_x, freqs, camera_indices))
        
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

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
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
                camera_indices: List,
                context: torch.Tensor,
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
        
        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
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
                            x, context, cam_emb_tgt, cam_emb_con, camera_indices, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, cam_emb_tgt, cam_emb_con, camera_indices, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, cam_emb_tgt, cam_emb_con, camera_indices, t_mod, freqs)

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
