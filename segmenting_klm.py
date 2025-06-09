import torch
from torch import nn
from modeling_klm import KLM, KLMConfig, repeat_kv, KLMMLP, KLMEmbedding
from modeling_klm import apply_rotary_pos_emb
import torch.nn.functional as F
import os
import numpy as np

from tracer import trace_tensor, executing


class LayerHead(nn.Module):
    """ln1 + first half of attn"""

    def __init__(self, config: KLMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.head_dim = config.hidden_size // config.num_attention_heads
        kv_size = self.head_dim * config.num_kv_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, kv_size)
        self.v_proj = nn.Linear(config.hidden_size, kv_size)

    def forward(self, hidden_states, rotary_pos_emb):
        residual = hidden_states

        hidden_states = self.ln1(hidden_states)
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

        return residual, q, k, v

class LayerBottom(nn.Module):
    """attn + mlp"""
    def __init__(self, config: KLMConfig):
        super().__init__()
        self.group_size = config.num_attention_heads // config.num_kv_heads
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.mlp = KLMMLP(config)
        self.ln2 = nn.LayerNorm(config.hidden_size)

    def forward(self, residual, q, ks, vs):
        # attention
        q = q.contiguous()
        k_repeated = repeat_kv(ks, self.group_size).contiguous()
        v_repeated = repeat_kv(vs, self.group_size).contiguous()

        trace_tensor("q", q[:, :, -1])
        trace_tensor("k_repeated", k_repeated)
        trace_tensor("v_repeated", v_repeated)

        # TODO: attn mask
        attn_output = F.scaled_dot_product_attention(
            q, k_repeated, v_repeated, is_causal=False, # False because we have kv cache
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        # shape (bsz, seq, n_heads, head_dim)

        trace_tensor("attn_output", attn_output[:, -1])

        input_shape = residual.shape[:-1]
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        hidden_states = attn_output + residual
        residual = hidden_states

        hidden_states = self.ln2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


class Head(nn.Module):
    """embedding layer + first layer's head"""

    def __init__(self, config: KLMConfig):
        super().__init__()
        self.config = config
        self.embedding = KLMEmbedding(config)
        self.layer_head = LayerHead(config)

    def forward(self, input_ids, rotary_pos_emb):
        hidden_states = self.embedding(input_ids)
        residual, q, k, v = self.layer_head(hidden_states, rotary_pos_emb)
        return residual, q, k, v

    def export_onnx(self, path):
        torch.onnx.export(
            self,
            (
                torch.randint(0, self.config.vocab_size, (1, 1)),
                torch.randn(2, 1, 1, self.config.hidden_size // self.config.num_attention_heads),
            ),
            path,
            input_names=["input_ids", "rotary_pos_emb"],
            output_names=["residual", "q", "k", "v"],
        )


class Segment(nn.Module):
    """previous bottom + next head"""

    def __init__(self, config: KLMConfig):
        super().__init__()
        self.config = config
        self.layer_bottom = LayerBottom(config)
        self.layer_head = LayerHead(config)

    def forward(self, residual, rotary_pos_emb, q, ks, vs):
        hidden_states = self.layer_bottom(residual, q, ks, vs)
        residual, q, k, v = self.layer_head(hidden_states, rotary_pos_emb)
        return residual, q, k, v

    def export_onnx(self, path):
        torch.onnx.export(
            self,
            (
                torch.randn(1, 1, self.config.hidden_size),
                torch.randn(2, 1, 1, self.config.hidden_size // self.config.num_attention_heads),
                torch.randn(1, self.config.num_attention_heads, 1, self.config.hidden_size // self.config.num_attention_heads),
                torch.randn(1, self.config.num_kv_heads, self.config.effective_length - 1, self.config.hidden_size // self.config.num_attention_heads),
                torch.randn(1, self.config.num_kv_heads, self.config.effective_length - 1, self.config.hidden_size // self.config.num_attention_heads),
            ),
            path,
            input_names=["residual", "rotary_pos_emb", "q", "ks", "vs"],
            output_names=["residual", "q", "k", "v"],
        )


class Tail(nn.Module):
    """last layer's bottom + lm_head"""

    def __init__(self, config: KLMConfig):
        super().__init__()
        self.config = config
        self.layer_bottom = LayerBottom(config)
        self.final_ln = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, residual, q, ks, vs):
        hidden_states = self.layer_bottom(residual, q, ks, vs)
        hidden_states = self.final_ln(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def export_onnx(self, path):
        torch.onnx.export(
            self,
            (
                torch.randn(1, 1, self.config.hidden_size),
                torch.randn(1, self.config.num_attention_heads, 1, self.config.hidden_size // self.config.num_attention_heads),
                torch.randn(1, self.config.num_kv_heads, self.config.effective_length - 1, self.config.hidden_size // self.config.num_attention_heads),
                torch.randn(1, self.config.num_kv_heads, self.config.effective_length - 1, self.config.hidden_size // self.config.num_attention_heads),
            ),
            path,
            input_names=["residual", "q", "ks", "vs"],
            output_names=["logits"],
        )


class SegmentedKLM(nn.Module):
    """segmented klm"""

    def __init__(self, config: KLMConfig):
        super().__init__()
        self.config = config
        self.head = Head(config)
        self.segments = nn.ModuleList([Segment(config) for _ in range(config.num_layers - 1)])
        self.tail = Tail(config)

    def forward(self, input_ids, rotary_pos_emb, past_ks, past_vs):
        residual, q, k, v = self.head(input_ids, rotary_pos_emb)
        new_past_k = []
        new_past_v = []
        for i, segment in enumerate(self.segments):
            # n_heads = k.shape[1]
            # head_dim = k.shape[3]
            ks = torch.cat([past_ks[i], k], dim=2)
            vs = torch.cat([past_vs[i], v], dim=2)
            new_past_k.append(ks)
            new_past_v.append(vs)
            with executing(f"layer{i}"):
                residual, q, k, v = segment(residual, rotary_pos_emb, q, ks, vs)
        ks = torch.cat([past_ks[len(self.segments)], k], dim=2)
        vs = torch.cat([past_vs[len(self.segments)], v], dim=2)
        new_past_k.append(ks)
        new_past_v.append(vs)
        with executing(f"layer{len(self.segments)}"):
            logits = self.tail(residual, q, ks, vs)
        trace_tensor("logits", logits[:, -1])
        return logits, torch.stack(new_past_k, dim=0), torch.stack(new_past_v, dim=0)

    def import_klm(self, klm_model_path):
        # Load the original KLM model
        original_klm = KLM(self.config)
        original_klm.load_state_dict(torch.load(klm_model_path, weights_only=True, map_location='cpu'))

        # Mapping from original KLM to segmented KLM
        state_dict = {}

        # Copy embedding weights
        state_dict['head.embedding.word_embeddings.weight'] = original_klm.embedding.word_embeddings.weight

        # Copy lm_head weights
        state_dict['tail.final_ln.weight'] = original_klm.final_ln.weight
        state_dict['tail.final_ln.bias'] = original_klm.final_ln.bias
        state_dict['tail.lm_head.weight'] = original_klm.lm_head.lm_head.weight

        # Process each layer
        for i in range(self.config.num_layers):
            # LayerNorm weights
            if i == 0:
                # First layer's head (in Head module)
                prefix = 'head.layer_head.'
                original_prefix = f'body.blocks.{i}.'
            else:
                # Middle layers (in Segment modules)
                prefix = f'segments.{i - 1}.layer_head.'
                original_prefix = f'body.blocks.{i}.'

            # Copy LayerNorm weights
            state_dict[prefix + 'ln1.weight'] = original_klm.state_dict()[original_prefix + 'ln1.weight']
            state_dict[prefix + 'ln1.bias'] = original_klm.state_dict()[original_prefix + 'ln1.bias']

            # Copy attention projection weights
            state_dict[prefix + 'q_proj.weight'] = original_klm.state_dict()[
                original_prefix + 'attention.q_proj.weight']
            state_dict[prefix + 'q_proj.bias'] = original_klm.state_dict()[original_prefix + 'attention.q_proj.bias']
            state_dict[prefix + 'k_proj.weight'] = original_klm.state_dict()[
                original_prefix + 'attention.k_proj.weight']
            state_dict[prefix + 'k_proj.bias'] = original_klm.state_dict()[original_prefix + 'attention.k_proj.bias']
            state_dict[prefix + 'v_proj.weight'] = original_klm.state_dict()[
                original_prefix + 'attention.v_proj.weight']
            state_dict[prefix + 'v_proj.bias'] = original_klm.state_dict()[original_prefix + 'attention.v_proj.bias']

            # Copy attention output projection weights (in LayerBottom)
            if i < self.config.num_layers - 1:
                bottom_prefix = f'segments.{i}.layer_bottom.'
            else:
                bottom_prefix = 'tail.layer_bottom.'

            state_dict[bottom_prefix + 'out_proj.weight'] = original_klm.state_dict()[
                original_prefix + 'attention.out_proj.weight']
            state_dict[bottom_prefix + 'out_proj.bias'] = original_klm.state_dict()[
                original_prefix + 'attention.out_proj.bias']

            # Copy MLP weights (in LayerBottom)
            state_dict[bottom_prefix + 'mlp.up_proj.weight'] = original_klm.state_dict()[
                original_prefix + 'mlp.up_proj.weight']
            state_dict[bottom_prefix + 'mlp.up_proj.bias'] = original_klm.state_dict()[
                original_prefix + 'mlp.up_proj.bias']
            state_dict[bottom_prefix + 'mlp.gate_proj.weight'] = original_klm.state_dict()[
                original_prefix + 'mlp.gate_proj.weight']
            state_dict[bottom_prefix + 'mlp.gate_proj.bias'] = original_klm.state_dict()[
                original_prefix + 'mlp.gate_proj.bias']
            state_dict[bottom_prefix + 'mlp.down_proj.weight'] = original_klm.state_dict()[
                original_prefix + 'mlp.down_proj.weight']
            state_dict[bottom_prefix + 'mlp.down_proj.bias'] = original_klm.state_dict()[
                original_prefix + 'mlp.down_proj.bias']

            # Copy LayerNorm2 weights (in LayerBottom)
            state_dict[bottom_prefix + 'ln2.weight'] = original_klm.state_dict()[original_prefix + 'ln2.weight']
            state_dict[bottom_prefix + 'ln2.bias'] = original_klm.state_dict()[original_prefix + 'ln2.bias']

        # Load the mapped state dict into the segmented model
        self.load_state_dict(state_dict, strict=True)


    def export_onnx(self, path):
        self.head.export_onnx(os.path.join(path, "seg_head.onnx"))
        for i, segment in enumerate(self.segments):
            segment.export_onnx(os.path.join(path, f"seg_{i}.onnx"))
        self.tail.export_onnx(os.path.join(path, "seg_tail.onnx"))


if __name__ == "__main__":
    config = KLMConfig()
    model = SegmentedKLM(config)
    if os.path.exists("klm_models/segmented/pytorch_model.bin"):
        model.load_state_dict(torch.load("klm_models/segmented/pytorch_model.bin", weights_only=True))
    model.import_klm("ckpt/klm-utb-4/step_19000/model.pt")
    os.makedirs("klm_models/segmented", exist_ok=True)
    torch.save(model.state_dict(), "klm_models/segmented/pytorch_model.bin")
    rope = np.load("rope.npz")["rope"]
    rope = torch.from_numpy(rope).view(torch.bfloat16).to(torch.float32)
    np.savez("klm_models/segmented/rope.npz", rope=rope.numpy())
    # logits = model(torch.tensor([[1]]), rope[:, :, 0])
    # torch.save(logits, "logits.pt")
    model.export_onnx("klm_models/segmented/")
