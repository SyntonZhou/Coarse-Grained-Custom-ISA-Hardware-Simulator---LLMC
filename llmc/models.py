"""
models.py -- model frontend (requirements #4 multi-model, #2 task structure).

Builds an ir.Graph from a ModelConfig.  The MLP is expanded into tiled SwiGLU
exactly the way the reference firmware splits the 5504-wide intermediate into
[2048, 2048, 1408] column tiles.

Attention is now fully lowered to the hardware instruction sequence:
  QKV proj -> [bias] -> RoPE(4 ops: cos-mul, sin-mul, swap, add) -> Rearrange(K)
  -> QK MatMul -> Mask -> Softmax -> Transpose(V) -> Rearrange(V)
  -> AV MatMul -> Concat -> Output proj.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .ir import Graph, OpType, DType, Kind
from .isa import round_up_64


# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    name: str
    hidden: int
    n_heads: int
    head_dim: int
    intermediate: int
    n_layers: int
    vocab: int
    n_kv_heads: Optional[int] = None
    norm: str = "rmsnorm"          # rmsnorm | layernorm
    act: str = "silu"             # silu | gelu
    rope: bool = True
    mlp_tiles: Optional[List[int]] = None   # explicit intermediate column split
    qkv_bias: bool = True
    rms_e: int = 0
    rms_r: int = 0
    max_token: int = 1024

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.mlp_tiles is None:
            self.mlp_tiles = self._auto_tiles(self.intermediate)

    @staticmethod
    def _auto_tiles(inter: int, cap: int = 2048) -> List[int]:
        tiles, left = [], inter
        while left > 0:
            t = min(cap, left)
            tiles.append(t)
            left -= t
        return tiles


# Presets ---------------------------------------------------------------------
QWEN = ModelConfig(
    name="qwen", hidden=2048, n_heads=16, head_dim=128, intermediate=5504,
    n_layers=24, vocab=151851, n_kv_heads=16, norm="rmsnorm", act="silu",
    rope=True, mlp_tiles=[2048, 2048, 1408], qkv_bias=True, max_token=1024,
)

LLAMA = ModelConfig(                 # Llama-3-8B shape (GQA, no QKV bias)
    name="llama", hidden=4096, n_heads=32, head_dim=128, intermediate=14336,
    n_layers=32, vocab=128256, n_kv_heads=8, norm="rmsnorm", act="silu",
    rope=True, qkv_bias=False,
)

LLAMA2_7B = ModelConfig(
    name="llama2_7b", hidden=4096, n_heads=32, head_dim=128, intermediate=11008,
    n_layers=32, vocab=32000, n_kv_heads=32, norm="rmsnorm", act="silu",
    rope=True, qkv_bias=False, max_token=4096,
)

PRESETS = {"qwen": QWEN, "llama": LLAMA, "llama2_7b": LLAMA2_7B}


# ---------------------------------------------------------------------------
class ModelBuilder:
    def __init__(self, cfg: ModelConfig, token_num: int = 16):
        self.cfg = cfg
        self.T = token_num          # 当前实际 token 数（如 16）
        self.max_T = cfg.max_token  # 最大预留（1024）
        self.g = Graph(f"{cfg.name}_T{token_num}")

    # -- weight helpers ----------------------------------------------------
    def _w(self, name, shape, **kwargs):
        return self.g.tensor(name, shape, DType.INT8, Kind.WEIGHT, **kwargs)

    def _w_fp16(self, name, shape, **kwargs):
        return self.g.tensor(name, shape, DType.FP16, Kind.WEIGHT, **kwargs)

    def _scale(self, name, shape, **kwargs):
        return self.g.tensor(name, shape, DType.FP16, Kind.SCALE, **kwargs)

    def _act(self, name, shape, dtype=DType.FP16, **kwargs):
        return self.g.tensor(name, shape, dtype, Kind.ACT, **kwargs)

    # -- MLP (tiled SwiGLU) -----------------------------------------------
    def build_mlp(self, x: str, layer: int, token_num: Optional[int] = None) -> str:
        g, cfg = self.g, self.cfg
        T = token_num if token_num is not None else self.T
        H, tiles = cfg.hidden, cfg.mlp_tiles
        p = f"L{layer}.mlp"
        x_i8 = self._act(f"{p}.x_i8", (T, H), DType.INT8)
        x_scale = self._scale(f"{p}.x_scale", (T,))
        g.op(OpType.FP16_INT8, [x], [x_i8], name=f"{p}.quant_in",
             rows=T, cols=H, addr_out_scale=x_scale,
             put_scale="scaleA", next_type=0)
        # 以下与原有 build_mlp 完全一致，只是所有 self.T 替换为 T
        gate_parts, up_parts = [], []
        for ti, tw in enumerate(tiles):
            wg = self._w(f"{p}.gate_w{ti}", (H, tw))
            sg = self._scale(f"{p}.gate_s{ti}", (tw,))
            wu = self._w(f"{p}.up_w{ti}", (H, tw))
            su = self._scale(f"{p}.up_s{ti}", (tw,))
            gate = self._act(f"{p}.gate{ti}", (T, tw))
            up = self._act(f"{p}.up{ti}", (T, tw))
            g.op(OpType.VEC_MATMUL, [x_i8, wg, sg], [gate], name=f"{p}.gate_mm{ti}",
                 a_row=T, b_row=H, b_col=tw, get_scale="scaleA", put_scale="none", next_type=1)
            g.op(OpType.VEC_MATMUL, [x_i8, wu, su], [up], name=f"{p}.up_mm{ti}",
                 a_row=T, b_row=H, b_col=tw, get_scale="scaleA", put_scale="none", next_type=1)
            gate_parts.append(gate)
            up_parts.append(up)

        h_parts = []
        for ti, tw in enumerate(tiles):
            act = self._act(f"{p}.act{ti}", (T, tw))
            g.op(OpType.SILU, [gate_parts[ti]], [act], name=f"{p}.silu{ti}",
                 elem=T * tw, dim=tw, put_scale="none", next_type=1)
            h = self._act(f"{p}.h{ti}", (T, tw))
            g.op(OpType.VU_MUL, [act, up_parts[ti]], [h], name=f"{p}.mul{ti}",
                 elem1=T * tw, dim=tw, elem2=T * tw, put_scale="none", next_type=1)
            h_i8 = self._act(f"{p}.h_i8_{ti}", (T, tw), DType.INT8)
            h_scale = self._scale(f"{p}.h_scale{ti}", (T,))
            g.op(OpType.FP16_INT8, [h], [h_i8], name=f"{p}.quant_h{ti}",
                 rows=T, cols=tw, addr_out_scale=h_scale,
                 put_scale="scaleA", next_type=0)
            h_parts.append(h_i8)

        partials = []
        for ti, tw in enumerate(tiles):
            wd = self._w(f"{p}.down_w{ti}", (tw, H))
            sd = self._scale(f"{p}.down_s{ti}", (H,))
            part = self._act(f"{p}.down{ti}", (T, H))
            g.op(OpType.VEC_MATMUL, [h_parts[ti], wd, sd], [part], name=f"{p}.down_mm{ti}",
                 a_row=T, b_row=tw, b_col=H, get_scale="scaleA", put_scale="none", next_type=1)
            partials.append(part)

        acc = partials[0]
        for ti in range(1, len(partials)):
            nxt = self._act(f"{p}.acc{ti}", (T, H))
            g.op(OpType.VU_ADD, [acc, partials[ti]], [nxt], name=f"{p}.add{ti}",
                 elem1=T * H, dim=H, elem2=T * H, put_scale="none", next_type=1)
            acc = nxt
        return acc
    
    def build_decode_attention(self, x: str, layer: int, token_idx: int = 0) -> str:
        g, cfg = self.g, self.cfg
        T = 1
        H, hd, n_h = cfg.hidden, cfg.head_dim, cfg.n_heads
        p = f"L{layer}.attn.t{token_idx}"        # 中间激活用带 token_idx 的前缀
        p_cache = f"L{layer}.attn"               # KV-cache / RoPE 表共享前缀
        row_idx = self.T + token_idx
        seq_len = row_idx + 1
        R = round_up_64(seq_len)

        # KV-cache: 共享，名字不带 token_idx
        k_cache = self.g.tensor(f"{p_cache}.k_cache", (cfg.max_token, hd), DType.FP16, Kind.KVCACHE)
        v_cache = self.g.tensor(f"{p_cache}.v_cache", (cfg.max_token, hd), DType.FP16, Kind.KVCACHE)

        # RoPE 表: 共享
        rope_cos = self._w_fp16(f"{p_cache}.rope_cos", (cfg.max_token, hd)) if cfg.rope else None
        rope_sin = self._w_fp16(f"{p_cache}.rope_sin", (cfg.max_token, hd)) if cfg.rope else None

        out_heads = []
        for h in range(n_h):
            # QKV proj
            wq = self._w(f"{p_cache}.q_w{h}", (H, hd)); sq = self._scale(f"{p_cache}.q_s{h}", (hd,))
            wk = self._w(f"{p_cache}.k_w{h}", (H, hd)); sk = self._scale(f"{p_cache}.k_s{h}", (hd,))
            wv = self._w(f"{p_cache}.v_w{h}", (H, hd)); sv = self._scale(f"{p_cache}.v_s{h}", (hd,))
            q = self._act(f"{p}.q{h}", (T, hd))
            # 在 build_decode_attention 中，alias tensor 增加 offset_expr
            k_new = self._act(f"{p}.k_new{h}", (T, hd),
                              alias_base=f"{p_cache}.k_cache",
                              alias_offset=row_idx * hd * 2,
                              alias_offset_expr=f"({self.T} + tok) * {hd * 2}")
            v_new = self._act(f"{p}.v_new{h}", (T, hd),
                              alias_base=f"{p_cache}.v_cache", alias_offset=row_idx * hd * 2,
                              alias_offset_expr=f"({self.T} + tok) * {hd * 2}")

            g.op(OpType.VEC_MATMUL, [x, wq, sq], [q], name=f"{p}.q_mm{h}",
                 a_row=T, b_row=H, b_col=hd, get_scale="scaleA", put_scale="none", next_type=1)
            g.op(OpType.VEC_MATMUL, [x, wk, sk], [k_new], name=f"{p}.k_mm{h}",
                 a_row=T, b_row=H, b_col=hd, get_scale="scaleA", put_scale="none", next_type=1)
            g.op(OpType.VEC_MATMUL, [x, wv, sv], [v_new], name=f"{p}.v_mm{h}",
                 a_row=T, b_row=H, b_col=hd, get_scale="scaleA", put_scale="none", next_type=1)

            # Bias
            if cfg.qkv_bias:
                q_b = self._w_fp16(f"{p_cache}.q_b{h}", (T, hd))
                k_b = self._w_fp16(f"{p_cache}.k_b{h}", (T, hd))
                v_b = self._w_fp16(f"{p_cache}.v_b{h}", (T, hd))
                q_ba = self._act(f"{p}.q_ba{h}", (T, hd))
                k_ba = self._act(f"{p}.k_ba{h}", (T, hd))
                v_ba = self._act(f"{p}.v_ba{h}", (T, hd))
                g.op(OpType.VU_ADD, [q, q_b], [q_ba], name=f"{p}.q_bias{h}",
                     elem1=T*hd, dim=hd, elem2=T*hd, put_scale="none", next_type=1)
                g.op(OpType.VU_ADD, [k_new, k_b], [k_ba], name=f"{p}.k_bias{h}",
                     elem1=T*hd, dim=hd, elem2=T*hd, put_scale="none", next_type=1)
                g.op(OpType.VU_ADD, [v_new, v_b], [v_ba], name=f"{p}.v_bias{h}",
                     elem1=T*hd, dim=hd, elem2=T*hd, put_scale="none", next_type=1)
                q, k_new, v_new = q_ba, k_ba, v_ba

            if cfg.rope:
                # RoPE Q: 只取当前位置的表（alias）
                rope_off = row_idx * hd * 2
                rope_cos_t = self.g.tensor(f"{p}.rope_cos_t{h}", (T, hd), DType.FP16, Kind.WEIGHT,
                                           alias_base=f"{p_cache}.rope_cos", alias_offset=rope_off,
                                           alias_offset_expr=f"({self.T} + tok) * {hd * 2}")
                rope_sin_t = self.g.tensor(f"{p}.rope_sin_t{h}", (T, hd), DType.FP16, Kind.WEIGHT,
                                           alias_base=f"{p_cache}.rope_sin", alias_offset=rope_off,
                                           alias_offset_expr=f"({self.T} + tok) * {hd * 2}")

                q_cos = self._act(f"{p}.q_cos{h}", (T, hd))
                g.op(OpType.VU_MUL, [q, rope_cos_t], [q_cos], name=f"{p}.rope_q_cos{h}",
                     elem1=T*hd, dim=hd, elem2=T*hd, put_scale="scaleA", next_type=1)
                q_sin = self._act(f"{p}.q_sin{h}", (T, hd))
                g.op(OpType.VU_MUL, [q, rope_sin_t], [q_sin], name=f"{p}.rope_q_sin{h}",
                     elem1=T*hd, dim=hd, elem2=T*hd, put_scale="scaleA", next_type=1)
                q_swapped = self._act(f"{p}.q_swapped{h}", (T, hd))
                g.op(OpType.SWAP, [q_sin], [q_swapped], name=f"{p}.rope_q_swap{h}",
                     elem=T*hd, token_num=T, dim=hd, put_scale="none", next_type=1)
                q_rope = self._act(f"{p}.qr{h}", (T, hd))
                g.op(OpType.VU_ADD, [q_swapped, q_cos], [q_rope], name=f"{p}.rope_q_add{h}",
                     elem1=T*hd, dim=hd, elem2=T*hd, put_scale="scaleA", next_type=0)

                # RoPE K: 整个 KV-cache 作为输入（alias）
                k_rope_in = self.g.tensor(f"{p}.k_rope_in{h}", (seq_len, hd), DType.FP16, Kind.ACT,
                                          alias_base=f"{p_cache}.k_cache", alias_offset=0,
                                          alias_offset_expr=f"0")  # alias 整行，offset_expr 由 generate_dma 计算
                k_cos = self._act(f"{p}.k_cos{h}", (seq_len, hd))
                g.op(OpType.VU_MUL, [k_rope_in, rope_cos], [k_cos], name=f"{p}.rope_k_cos{h}",
                     elem1=seq_len*hd, dim=hd, elem2=seq_len*hd, put_scale="scaleA", next_type=1)
                k_sin = self._act(f"{p}.k_sin{h}", (seq_len, hd))
                g.op(OpType.VU_MUL, [k_rope_in, rope_sin], [k_sin], name=f"{p}.rope_k_sin{h}",
                     elem1=seq_len*hd, dim=hd, elem2=seq_len*hd, put_scale="scaleA", next_type=1)
                k_swapped = self._act(f"{p}.k_swapped{h}", (seq_len, hd))
                g.op(OpType.SWAP, [k_sin], [k_swapped], name=f"{p}.rope_k_swap{h}",
                     elem=seq_len*hd, token_num=seq_len, dim=hd, put_scale="none", next_type=1)
                k_scale = self._scale(f"{p}.k_scale{h}", (seq_len,))
                k_rope = self._act(f"{p}.kr{h}", (seq_len, hd))
                g.op(OpType.VU_ADD, [k_swapped, k_cos], [k_rope], name=f"{p}.rope_k_add{h}",
                     elem1=seq_len*hd, dim=hd, elem2=seq_len*hd, put_scale="hbm", next_type=0,
                     addr_out_scale=k_scale)

                k_rearr = self._act(f"{p}.kr_rearr{h}", (R, hd))
                g.op(OpType.REARRANGE, [k_rope], [k_rearr], name=f"{p}.k_rearr{h}",
                     elem_in=seq_len*hd, elem_out=R*hd, token_num=seq_len, dim=hd, heads=1,
                     put_scale="none", next_type=0)

                qk = self._act(f"{p}.qk{h}", (T, R))
                g.op(OpType.VEC_MATMUL, [q_rope, k_rearr, k_scale], [qk], name=f"{p}.qk_mm{h}",
                     a_row=T, b_row=hd, b_col=R, get_scale="scaleAR", put_scale="none", next_type=1)
            else:
                k_rope_in = self.g.tensor(f"{p}.k_rope_in{h}", (seq_len, hd), DType.FP16, Kind.ACT,
                                          alias_base=f"{p_cache}.k_cache", alias_offset=0,
                                          alias_offset_expr=f"0")  # alias 整行，offset_expr 由 generate_dma 计算
                qk = self._act(f"{p}.qk{h}", (T, R))
                g.op(OpType.VEC_MATMUL, [q, k_rope_in, sk], [qk], name=f"{p}.qk_mm{h}",
                     a_row=T, b_row=hd, b_col=R, get_scale="scaleA", put_scale="none", next_type=1)

            # Softmax (Decode 无 mask)
            sm = self._act(f"{p}.sm{h}", (T, R), DType.INT8)
            g.op(OpType.SOFTMAX, [qk, qk], [sm], name=f"{p}.softmax{h}",
                 elem1=T*R, dim=R, elem2=T*R, vld_len=seq_len, put_scale="scaleA", next_type=0)

            if cfg.rope:
                v_scale = self._scale(f"{p}.v_scale{h}", (seq_len,))
                v_trans = self._act(f"{p}.v_trans{h}", (R, hd))
                v_rope_in = self.g.tensor(f"{p}.v_rope_in{h}", (seq_len, hd), DType.FP16, Kind.ACT,
                                          alias_base=f"{p_cache}.v_cache", alias_offset=0,
                                          alias_offset_expr=f"0")  # alias 整行，offset_expr 由 generate_dma 计算
                g.op(OpType.TRANSPOSE, [v_rope_in], [v_trans], name=f"{p}.v_trans{h}",
                     elem_in=seq_len*hd, elem_out=R*hd, token_num=seq_len, dim=hd, heads=1,
                     addr_out_scale=v_scale, put_scale="hbm", next_type=0)
                v_rearr = self._act(f"{p}.v_rearr{h}", (R, hd))
                g.op(OpType.REARRANGE, [v_trans], [v_rearr], name=f"{p}.v_rearr{h}",
                     elem_in=R*hd, elem_out=R*hd, token_num=R, dim=hd, heads=1,
                     put_scale="none", next_type=0)

                oh = self._act(f"{p}.o{h}", (T, hd))
                g.op(OpType.VEC_MATMUL, [sm, v_rearr, v_scale], [oh], name=f"{p}.av_mm{h}",
                     a_row=T, b_row=R, b_col=hd, get_scale="scaleAR", put_scale="none", next_type=1)
            else:
                v_rope_in = self.g.tensor(f"{p}.v_rope_in{h}", (seq_len, hd), DType.FP16, Kind.ACT,
                                          alias_base=f"{p_cache}.v_cache", alias_offset=0,
                                          alias_offset_expr=f"0")
                oh = self._act(f"{p}.o{h}", (T, hd))
                g.op(OpType.VEC_MATMUL, [sm, v_rope_in, sv], [oh], name=f"{p}.av_mm{h}",
                     a_row=T, b_row=R, b_col=hd, get_scale="scaleA", put_scale="none", next_type=1)
            out_heads.append(oh)

        concat = self._act(f"{p}.concat", (T, H))
        g.op(OpType.CONCAT, out_heads, [concat], name=f"{p}.concat",
             token_num=T, dim=H, heads=n_h, put_scale="none", next_type=0)

        wo = self._w(f"{p_cache}.o_w", (H, H)); so = self._scale(f"{p_cache}.o_s", (H,))
        out = self._act(f"{p}.out", (T, H))
        g.op(OpType.VEC_MATMUL, [concat, wo, so], [out], name=f"{p}.o_mm",
             a_row=T, b_row=H, b_col=H, get_scale="scaleA", put_scale="none", next_type=1)
        return out

    def build_decode_layer(self, x: str, layer: int, token_idx: int = 0) -> str:
        g, cfg, H = self.g, self.cfg, self.cfg.hidden
        p = f"L{layer}.t{token_idx}"          # 带 token_idx
        norm_op = OpType.RMSNORM if cfg.norm == "rmsnorm" else OpType.RMSNORM

        n1 = self._act(f"{p}.norm1", (1, H))
        g.op(norm_op, [x, x], [n1], name=f"{p}.rmsnorm1",
             elem1=1*H, dim=H, elem2=1*H, e=cfg.rms_e, r=cfg.rms_r,
             put_scale="scaleAR", next_type=0)

        attn = self.build_decode_attention(n1, layer, token_idx)

        r1 = self._act(f"{p}.res1", (1, H))
        g.op(OpType.RESIDUAL, [attn, x], [r1], name=f"{p}.residual1",
             elem1=1*H, dim=H, elem2=1*H, put_scale="none", next_type=1)

        n2 = self._act(f"{p}.norm2", (1, H))
        g.op(norm_op, [r1, r1], [n2], name=f"{p}.rmsnorm2",
             elem1=1*H, dim=H, elem2=1*H, e=cfg.rms_e, r=cfg.rms_r,
             put_scale="scaleAR", next_type=0)

        mlp = self.build_mlp(n2, layer, token_num=1)

        r2 = self._act(f"{p}.res2", (1, H))
        g.op(OpType.RESIDUAL, [mlp, r1], [r2], name=f"{p}.residual2",
             elem1=1*H, dim=H, elem2=1*H, put_scale="none", next_type=1)
        return r2

    # -- attention (full prefill path, Step-2 lowered) ---------------------
    def build_attention(self, x: str, layer: int) -> str:
        g, cfg, T = self.g, self.cfg, self.T
        H, hd, n_h = cfg.hidden, cfg.head_dim, cfg.n_heads
        p = f"L{layer}.attn"
        R = round_up_64(T)                     # 64-padded sequence length

        # Shared RoPE tables + causal mask (FP16 weights, not INT8)
        rope_cos = self._w_fp16(f"{p}.rope_cos", (cfg.max_token, hd)) if cfg.rope else None
        rope_sin = self._w_fp16(f"{p}.rope_sin", (cfg.max_token, hd)) if cfg.rope else None
        mask = self._w_fp16(f"{p}.mask", (R, R)) if cfg.rope else None

        out_heads = []
        for h in range(n_h):
            # ---- QKV linear projections ----
            wq = self._w(f"{p}.q_w{h}", (H, hd)); sq = self._scale(f"{p}.q_s{h}", (hd,))
            wk = self._w(f"{p}.k_w{h}", (H, hd)); sk = self._scale(f"{p}.k_s{h}", (hd,))
            wv = self._w(f"{p}.v_w{h}", (H, hd)); sv = self._scale(f"{p}.v_s{h}", (hd,))
            q = self._act(f"{p}.q{h}", (T, hd))
            k = self._act(f"{p}.k{h}", (T, hd))
            v = self._act(f"{p}.v{h}", (T, hd))
            g.op(OpType.VEC_MATMUL, [x, wq, sq], [q], name=f"{p}.q_mm{h}",
                 a_row=T, b_row=H, b_col=hd, get_scale="scaleA", put_scale="none", next_type=1)
            g.op(OpType.VEC_MATMUL, [x, wk, sk], [k], name=f"{p}.k_mm{h}",
                 a_row=T, b_row=H, b_col=hd, get_scale="scaleA", put_scale="none", next_type=1)
            g.op(OpType.VEC_MATMUL, [x, wv, sv], [v], name=f"{p}.v_mm{h}",
                 a_row=T, b_row=H, b_col=hd, get_scale="scaleA", put_scale="none", next_type=1)

            # ---- QKV bias (FP16, matching the firmware's length register) ----
            if cfg.qkv_bias:
                # Bias 预展开为 max_token×hd，HBM 中存 1024 份
                q_b = self._w_fp16(f"{p}.q_b{h}", (cfg.max_token, hd))
                k_b = self._w_fp16(f"{p}.k_b{h}", (cfg.max_token, hd))
                v_b = self._w_fp16(f"{p}.v_b{h}", (cfg.max_token, hd))
                q_ba = self._act(f"{p}.q_ba{h}", (T, hd))
                k_ba = self._act(f"{p}.k_ba{h}", (T, hd))
                v_ba = self._act(f"{p}.v_ba{h}", (T, hd))
                g.op(OpType.VU_ADD, [q, q_b], [q_ba], name=f"{p}.q_bias{h}",
                     elem1=T*hd, dim=hd, elem2=T*hd, put_scale="none", next_type=1)
                g.op(OpType.VU_ADD, [k, k_b], [k_ba], name=f"{p}.k_bias{h}",
                     elem1=T*hd, dim=hd, elem2=T*hd, put_scale="none", next_type=1)
                g.op(OpType.VU_ADD, [v, v_b], [v_ba], name=f"{p}.v_bias{h}",
                     elem1=T*hd, dim=hd, elem2=T*hd, put_scale="none", next_type=1)
                q, k, v = q_ba, k_ba, v_ba

            if cfg.rope:
                # ---- RoPE Q (4 ops) ----
                q_cos = self._act(f"{p}.q_cos{h}", (T, hd))
                g.op(OpType.VU_MUL, [q, rope_cos], [q_cos], name=f"{p}.rope_q_cos{h}",
                     elem1=T*hd, dim=hd, elem2=T*hd, put_scale="scaleA", next_type=1)
                q_sin = self._act(f"{p}.q_sin{h}", (T, hd))
                g.op(OpType.VU_MUL, [q, rope_sin], [q_sin], name=f"{p}.rope_q_sin{h}",
                     elem1=T*hd, dim=hd, elem2=T*hd, put_scale="scaleA", next_type=1)
                q_swapped = self._act(f"{p}.q_swapped{h}", (T, hd))
                g.op(OpType.SWAP, [q_sin], [q_swapped], name=f"{p}.rope_q_swap{h}",
                     elem=T*hd, token_num=T, dim=hd, put_scale="none", next_type=1)
                q_rope = self._act(f"{p}.qr{h}", (T, hd))
                g.op(OpType.VU_ADD, [q_swapped, q_cos], [q_rope], name=f"{p}.rope_q_add{h}",
                     elem1=T*hd, dim=hd, elem2=T*hd, put_scale="scaleA", next_type=0)

                # ---- RoPE K (4 ops + scale to HBM) ----
                k_cos = self._act(f"{p}.k_cos{h}", (T, hd))
                g.op(OpType.VU_MUL, [k, rope_cos], [k_cos], name=f"{p}.rope_k_cos{h}",
                     elem1=T*hd, dim=hd, elem2=T*hd, put_scale="scaleA", next_type=1)
                k_sin = self._act(f"{p}.k_sin{h}", (T, hd))
                g.op(OpType.VU_MUL, [k, rope_sin], [k_sin], name=f"{p}.rope_k_sin{h}",
                     elem1=T*hd, dim=hd, elem2=T*hd, put_scale="scaleA", next_type=1)
                k_swapped = self._act(f"{p}.k_swapped{h}", (T, hd))
                g.op(OpType.SWAP, [k_sin], [k_swapped], name=f"{p}.rope_k_swap{h}",
                     elem=T*hd, token_num=T, dim=hd, put_scale="none", next_type=1)
                k_scale = self._scale(f"{p}.k_scale{h}", (T,))
                k_rope = self._act(f"{p}.kr{h}", (T, hd))
                g.op(OpType.VU_ADD, [k_swapped, k_cos], [k_rope], name=f"{p}.rope_k_add{h}",
                     elem1=T*hd, dim=hd, elem2=T*hd, put_scale="hbm", next_type=0,
                     addr_out_scale=k_scale)
                k = k_rope

                # ---- Rearrange K (pad to 64) ----
                k_rearr = self._act(f"{p}.kr_rearr{h}", (R, hd), DType.INT8)
                g.op(OpType.REARRANGE, [k], [k_rearr], name=f"{p}.k_rearr{h}",
                     elem_in=T*hd, elem_out=R*hd, token_num=T, dim=hd, heads=1,
                     put_scale="none", next_type=0)

                # ---- QK MatMul (Q x K_rearr, K_scale as addr3) ----
                qk = self._act(f"{p}.qk{h}", (T, R))
                g.op(OpType.VEC_MATMUL, [q_rope, k_rearr, k_scale], [qk], name=f"{p}.qk_mm{h}",
                     a_row=T, b_row=hd, b_col=R, get_scale="scaleAR", put_scale="none", next_type=1)
            else:
                # No RoPE: direct QK
                qk = self._act(f"{p}.qk{h}", (T, R))
                g.op(OpType.VEC_MATMUL, [q, k, sk], [qk], name=f"{p}.qk_mm{h}",
                     a_row=T, b_row=hd, b_col=R, get_scale="scaleA", put_scale="none", next_type=1)

            # ---- Mask (causal) ----
            qk_m = self._act(f"{p}.qk_m{h}", (T, R))
            g.op(OpType.VU_MASK, [qk, mask], [qk_m], name=f"{p}.mask{h}",
                 elem1=T*R, dim=T, elem2=R*R, put_scale="none", next_type=1)

            # ---- Softmax ----
            sm = self._act(f"{p}.sm{h}", (T, R), DType.INT8)
            g.op(OpType.SOFTMAX, [qk_m, qk_m], [sm], name=f"{p}.softmax{h}",
                 elem1=T*R, dim=R, elem2=T*R, vld_len=T, put_scale="scaleA", next_type=0)

            # ---- V path: transpose -> rearrange ----
            if cfg.rope:
                v_scale = self._scale(f"{p}.v_scale{h}", (T,))
                v_trans = self._act(f"{p}.v_trans{h}", (R, hd))
                g.op(OpType.TRANSPOSE, [v], [v_trans], name=f"{p}.v_trans{h}",
                     elem_in=T*hd, elem_out=R*hd, token_num=T, dim=hd, heads=1,
                     addr_out_scale=v_scale, put_scale="hbm", next_type=0)
                v_rearr = self._act(f"{p}.v_rearr{h}", (R, hd), DType.INT8)
                # token_num=R because V has been transposed to R rows
                g.op(OpType.REARRANGE, [v_trans], [v_rearr], name=f"{p}.v_rearr{h}",
                     elem_in=R*hd, elem_out=R*hd, token_num=R, dim=hd, heads=1,
                     put_scale="none", next_type=0)
                v = v_rearr

            # ---- AV MatMul ----
            oh = self._act(f"{p}.o{h}", (T, hd))
            if cfg.rope:
                g.op(OpType.VEC_MATMUL, [sm, v, v_scale], [oh], name=f"{p}.av_mm{h}",
                     a_row=T, b_row=R, b_col=hd, get_scale="scaleAR", put_scale="none", next_type=1)
            else:
                g.op(OpType.VEC_MATMUL, [sm, v, sv], [oh], name=f"{p}.av_mm{h}",
                     a_row=T, b_row=R, b_col=hd, get_scale="scaleA", put_scale="none", next_type=1)
            out_heads.append(oh)

        # ---- Concat (firmware expects out_heads contiguous in HBM) ----
        concat = self._act(f"{p}.concat", (T, H), DType.INT8)
        g.op(OpType.CONCAT, out_heads, [concat], name=f"{p}.concat",
             token_num=T, dim=H, heads=n_h, put_scale="none", next_type=0)

        # ---- Output projection ----
        wo = self._w(f"{p}.o_w", (H, H)); so = self._scale(f"{p}.o_s", (H,))
        out = self._act(f"{p}.out", (T, H))
        g.op(OpType.VEC_MATMUL, [concat, wo, so], [out], name=f"{p}.o_mm",
             a_row=T, b_row=H, b_col=H, get_scale="scaleA", put_scale="none", next_type=1)
        return out
    
    # -- one decoder layer -------------------------------------------------
    def build_layer(self, x: str, layer: int) -> str:
        g, cfg, T, H = self.g, self.cfg, self.T, self.cfg.hidden
        p = f"L{layer}"
        norm_op = OpType.RMSNORM if cfg.norm == "rmsnorm" else OpType.RMSNORM
        n1 = self._act(f"{p}.norm1", (T, H))
        g.op(norm_op, [x, x], [n1], name=f"{p}.rmsnorm1",
             elem1=T * H, dim=H, elem2=T * H, e=cfg.rms_e, r=cfg.rms_r,
             put_scale="scaleAR", next_type=0)
        attn = self.build_attention(n1, layer)
        r1 = self._act(f"{p}.res1", (T, H))
        g.op(OpType.RESIDUAL, [attn, x], [r1], name=f"{p}.residual1",
             elem1=T * H, dim=H, elem2=T * H, put_scale="none", next_type=1)
        n2 = self._act(f"{p}.norm2", (T, H))
        g.op(norm_op, [r1, r1], [n2], name=f"{p}.rmsnorm2",
             elem1=T * H, dim=H, elem2=T * H, e=cfg.rms_e, r=cfg.rms_r,
             put_scale="scaleAR", next_type=0)
        mlp = self.build_mlp(n2, layer)
        r2 = self._act(f"{p}.res2", (T, H))
        g.op(OpType.RESIDUAL, [mlp, r1], [r2], name=f"{p}.residual2",
             elem1=T * H, dim=H, elem2=T * H, put_scale="none", next_type=1)
        return r2

    # -- whole model (or MLP-only slice) -----------------------------------
    def build(self, n_layers: Optional[int] = None, mlp_only: bool = False,
              decode: bool = False, decode_tokens: int = 1) -> Graph:
        cfg, T, H = self.cfg, self.T, self.cfg.hidden
        n_layers = cfg.n_layers if n_layers is None else n_layers

        if decode:
            # Decode phase: generate decode_tokens new tokens
            x = self.g.tensor("hidden_in", (1, H), DType.FP16, Kind.INPUT)
            for ti in range(decode_tokens):
                for li in range(n_layers):
                    x = self.build_decode_layer(x, li, ti)
            out = x
        else:
            # Prefill phase (existing)
            x = self.g.tensor("hidden_in", (T, H), DType.FP16, Kind.INPUT)
            if mlp_only:
                n = self._act("L0.norm", (T, H))
                self.g.op(OpType.RMSNORM, [x, x], [n], name="L0.rmsnorm",
                          elem1=T * H, dim=H, elem2=T * H, e=cfg.rms_e, r=cfg.rms_r,
                          put_scale="scaleAR", next_type=0)
                mlp = self.build_mlp(n, 0)
                out = self._act("L0.out", (T, H))
                self.g.op(OpType.RESIDUAL, [mlp, x], [out], name="L0.residual",
                          elem1=T * H, dim=H, elem2=T * H, put_scale="none", next_type=1)
            else:
                for li in range(n_layers):
                    x = self.build_layer(x, li)
                out = x

        self.g.tensors[out].kind = Kind.OUTPUT
        return self.g


def build_graph(cfg_name: str, token_num: int = 16,
                n_layers: Optional[int] = None, mlp_only: bool = False,
                decode: bool = False, decode_tokens: int = 1) -> Graph:
    cfg = PRESETS[cfg_name]
    return ModelBuilder(cfg, token_num).build(
        n_layers=n_layers, mlp_only=mlp_only,
        decode=decode, decode_tokens=decode_tokens
    )
