import os

import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
try:
    flash_attn_available = True
    from flash_attn import flash_attn_func
except:
    flash_attn_available = False
    pass
from tracer import executing, trace_tensor


@dataclass
class KLMConfig:
    max_length: int = 131072
    effective_length: int = 512
    vocab_size: int = 512
    hidden_size: int = 512
    tie_word_embeddings: bool = True
    num_layers: int = 6
    num_attention_heads: int = 8
    num_kv_heads: int = 2
    ffn_hidden_size: int = 2048


class KLMEmbedding(nn.Module):
    def __init__(self, config: KLMConfig):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, input_ids):
        return self.word_embeddings(input_ids)

    def export_onnx(self, path):
        torch.onnx.export(
            self,
            (torch.arange(0, self.config.effective_length, dtype=torch.long).unsqueeze(0),),
            path,
            input_names=["input_ids"],
            output_names=["input_embed"],
            dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"}},
        )


class KLMHead(nn.Module):
    def __init__(self, config: KLMConfig):
        super().__init__()
        self.config = config
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        return self.lm_head(hidden_states)

    def export_onnx(self, path):
        torch.onnx.export(
            self,
            (torch.randn(1, self.config.effective_length, self.config.hidden_size),),
            path,
            input_names=["hidden_states"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"}},
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, rotary_pos_emb):
    cos = rotary_pos_emb[0]
    sin = rotary_pos_emb[1]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000, max_position: int = 2048):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position = max_position

        # Pre-compute inv_freq
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        # Pre-compute position embeddings for all possible positions
        position = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', position, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Pre-compute cos and sin
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    @torch.no_grad()
    def forward(self, position_ids):
        # Simply index into pre-computed cos and sin
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]
        return torch.stack([cos, sin], dim=0)

    def export_onnx(self, path):
        torch.onnx.export(
            self,
            (torch.arange(0, self.config.effective_length, dtype=torch.long).unsqueeze(0),),
            path,
            input_names=["position_ids"],
            output_names=["cos", "sin"],
            dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"}},
        )


class KLMAttention(nn.Module):
    def __init__(self, config: KLMConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        kv_size = self.head_dim * config.num_kv_heads
        self.group_size = config.num_attention_heads // config.num_kv_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, kv_size)
        self.v_proj = nn.Linear(config.hidden_size, kv_size)

        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, attn_mask, rotary_pos_emb, past_k=None, past_v=None):
        # hidden_states = hidden_states * attn_mask # TODO: just for adding mask to graph
        q_state = self.q_proj(hidden_states)
        k_state = self.k_proj(hidden_states)
        v_state = self.v_proj(hidden_states)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = q_state.view(*hidden_shape).transpose(1, 2)
        k = k_state.view(*hidden_shape).transpose(1, 2)
        v = v_state.view(*hidden_shape).transpose(1, 2)
        # shape (bsz, n_heads, seq, head_dim)

        q, k = apply_rotary_pos_emb(q, k, rotary_pos_emb)

        if past_k is not None:
            assert past_v is not None
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        q = q.contiguous()
        k_repeated = repeat_kv(k, self.group_size).contiguous()
        v_repeated = repeat_kv(v, self.group_size).contiguous()

        trace_tensor("q", q[:, :, -1])
        trace_tensor("k_repeated", k_repeated)
        trace_tensor("v_repeated", v_repeated)

        # TODO: attn mask
        if past_k is None and self.training and flash_attn_available:
            # use flash attn
            q = q.transpose(1, 2)
            k_repeated = k_repeated.transpose(1, 2)
            v_repeated = v_repeated.transpose(1, 2)
            attn_output = flash_attn_func(q, k_repeated, v_repeated, causal=True)
        else:
            attn_output = F.scaled_dot_product_attention(
                q, k_repeated, v_repeated, is_causal=past_k is None,
            )
            attn_output = attn_output.transpose(1, 2)
            # shape (bsz, seq, n_heads, head_dim)

        trace_tensor("attn_output", attn_output[:, -1])

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output, k, v


class KLMMLP(nn.Module):
    def __init__(self, config: KLMConfig):
        super().__init__()
        self.config = config
        self.up_proj = nn.Linear(config.hidden_size, config.ffn_hidden_size)
        self.gate_proj = nn.Linear(config.hidden_size, config.ffn_hidden_size)
        self.down_proj = nn.Linear(config.ffn_hidden_size, config.hidden_size)

    def forward(self, hidden_states):
        up = self.up_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        hidden_states = self.down_proj(F.silu(up) * gate)
        return hidden_states


class KLMBlock(nn.Module):
    def __init__(self, config: KLMConfig):
        super().__init__()
        self.attention = KLMAttention(config)
        self.mlp = KLMMLP(config)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states, attn_mask, rotary_pos_emb, past_k=None, past_v=None):
        residual = hidden_states

        hidden_states = self.ln1(hidden_states)
        hidden_states, past_k, past_v = self.attention(hidden_states, attn_mask, rotary_pos_emb, past_k, past_v)

        if self.training:
            past_k = past_v = None

        hidden_states = hidden_states + residual
        residual = hidden_states

        hidden_states = self.ln2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states, past_k, past_v

    def export_onnx(self, path, mode="prefill"):
        if mode == "prefill":
            torch.onnx.export(
                self,
                (
                    torch.randn(1, self.config.effective_length, self.config.hidden_size),
                    torch.ones(1, self.config.effective_length),
                    torch.randn(1, self.config.effective_length, self.config.hidden_size // self.config.num_attention_heads),
                ),
                path,
                input_names=["hidden_states", "attn_mask", "rotary_pos_emb"],
                output_names=["hidden_states"],
                dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"}},
            )
        elif mode == "decode":
            torch.onnx.export(
                self,
                (
                    torch.randn(1, 1, self.config.hidden_size),
                    torch.ones(1, 1),
                    torch.randn(1, 1, self.config.hidden_size // self.config.num_attention_heads),
                    torch.randn(1, self.config.effective_length - 1, 2, self.config.hidden_size // self.config.num_attention_heads),
                    torch.randn(1, self.config.effective_length - 1, 2, self.config.hidden_size // self.config.num_attention_heads),
                ),
                path,
                input_names=["hidden_states", "attn_mask", "rotary_pos_emb", "past_k", "past_v"],
                output_names=["hidden_states", "past_k_out", "past_v_out"],
                dynamic_axes={"input_ids": {0: "batch_size"}},
            )
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'prefill' or 'decode'.")


class KLMBody(nn.Module):
    def __init__(self, config: KLMConfig):
        super().__init__()
        self.config = config
        self.rotary_embedding = RotaryEmbedding(config.hidden_size // config.num_attention_heads, max_position=config.max_length)
        self.blocks = nn.ModuleList([KLMBlock(config) for _ in range(config.num_layers)])

    def forward(self, hidden_states, attn_mask=None, position_ids=None, past_k=None, past_v=None):
        ks = []
        vs = []
        if position_ids is None:
            position_ids = torch.arange(hidden_states.size(1), dtype=torch.long, device=hidden_states.device).unsqueeze(0)
            position_ids = position_ids.repeat(hidden_states.size(0), 1)
        if attn_mask is None:
            attn_mask = torch.ones(hidden_states.size(0), hidden_states.size(1), device=hidden_states.device)
        rotary_pos_emb = self.rotary_embedding(position_ids)
        trace_tensor("rotary_pos_emb", rotary_pos_emb[:, :, -1])
        for i, block in enumerate(self.blocks):
            if past_k is not None:
                pk = past_k[i]
                pv = past_v[i]
            else:
                pk = pv = None
            with executing(f"layer{i}"):
                hidden_states, k, v = block(hidden_states, attn_mask, rotary_pos_emb, pk, pv)
            ks.append(k)
            vs.append(v)
        if not self.training:
            ks = torch.stack(ks, dim=0)
            vs = torch.stack(vs, dim=0)
        return hidden_states, ks, vs

    def export_onnx(self, path, mode="prefill"):
        if mode == "prefill":
            torch.onnx.export(
                self,
                (
                    torch.randn(1, self.config.effective_length, self.config.hidden_size),
                    torch.ones(1, self.config.effective_length),
                    torch.arange(1, self.config.effective_length, dtype=torch.long).unsqueeze(0),
                ),
                path,
                input_names=["hidden_states", "attn_mask", "position_ids"],
                output_names=["hidden_states", "past_k_out", "past_v_out"],
                dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"}},
            )
        elif mode == "decode":
            torch.onnx.export(
                self,
                (
                    torch.randn(1, 1, self.config.hidden_size),
                    None,
                    torch.tensor([[0]], dtype=torch.long),
                ),
                path,
                input_names=["hidden_states", "attn_mask", "position_ids", "past_k", "past_v"],
                output_names=["hidden_states", "past_k_out", "past_v_out"],
                dynamic_axes={"input_ids": {0: "batch_size"}},
            )


class KLM(nn.Module):
    def __init__(self, config: KLMConfig):
        super().__init__()
        self.config = config
        self.embedding = KLMEmbedding(config)
        self.body = KLMBody(config)
        self.final_ln = nn.LayerNorm(config.hidden_size)
        self.lm_head = KLMHead(config)

        if config.tie_word_embeddings:
            self.lm_head.lm_head.weight = self.embedding.word_embeddings.weight

    def forward(self, input_ids, attn_mask=None, position_ids=None, past_k=None, past_v=None):
        hidden_states = self.embedding(input_ids)
        trace_tensor("input_embed", hidden_states[:, -1])

        hidden_states, past_k, past_v = self.body(hidden_states, attn_mask, position_ids, past_k, past_v)

        hidden_states = self.final_ln(hidden_states)
        logits = self.lm_head(hidden_states)
        trace_tensor("logits", logits[:, -1])

        return logits, past_k, past_v

    def export_submodule_onnx(self, path):
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        self.embedding.export_onnx(path / "embedding.onnx")
        self.lm_head.export_onnx(path / "lm_head.onnx")

        self.body.export_onnx(path / "prefill_body.onnx", mode="prefill")
        self.body.export_onnx(path / "decode_body.onnx", mode="decode")

        self.rotary_embedding.export_onnx(path / "rotary_embedding.onnx")

    def export_onnx(self, path, mode):
        if mode == "prefill":
            torch.onnx.export(
                self,
                (
                    torch.randint(0, self.config.vocab_size, (1, self.config.effective_length)),
                    torch.ones(1, self.config.effective_length),
                ),
                path,
                input_names=["input_ids", "attn_mask"],
                output_names=["logits", "past_k_out", "past_v_out"],
                dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"}, "attn_mask": {0: "batch_size", 1: "seq_len"}},
            )
        elif mode == "decode":
            torch.onnx.export(
                self,
                (
                    torch.randint(0, self.config.vocab_size, (1, 1)),
                    None,
                    torch.tensor([[0]], dtype=torch.long),
                    torch.randn(1, self.config.num_kv_heads, self.config.effective_length - 1, self.config.hidden_size // self.config.num_attention_heads),
                    torch.randn(1, self.config.num_kv_heads, self.config.effective_length - 1, self.config.hidden_size // self.config.num_attention_heads),
                ),
                path,
                input_names=["input_ids", "position_ids", "past_k", "past_v"],
                output_names=["logits", "past_k_out", "past_v_out"],
                dynamic_axes={"input_ids": {0: "batch_size"}, "position_ids": {0: "batch_size"}},
            )


def save_rope(config: KLMConfig):
    import numpy as np
    rotary_embedding = RotaryEmbedding(config.hidden_size // config.num_attention_heads, max_position=8192)
    position_ids = torch.arange(0, 8192, dtype=torch.long).unsqueeze(0)
    rope = rotary_embedding(position_ids).to(torch.bfloat16)
    # save bf16 npz
    rope = rope.view(torch.uint16).cpu().numpy()
    np.savez_compressed("rope.npz", rope=rope)


if __name__ == '__main__':
    config = KLMConfig()
    # save_rope(config)
    model = KLM(config)

    # torch.nn.init.normal_(model.lm_head.lm_head.weight, std=0.02)
    logits = model(torch.randint(0, config.vocab_size, (1, 1)), None, torch.tensor([[0]], dtype=torch.long))[0]
    print(logits.max())
    # breakpoint()
    #
    # # count params
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params}")
    #
    # os.makedirs("klm_models", exist_ok=True)
    # model.export_onnx("klm_models/prefill_model.onnx", mode="prefill")
    # model.export_onnx("klm_models/decode_model.onnx", mode="decode")
