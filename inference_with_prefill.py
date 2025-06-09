from segmenting_klm import SegmentedKLM
from modeling_klm import KLMConfig, KLM
import numpy as np
import torch
from tqdm import tqdm
from getch import getch
from tracer import trace_tensor, running, executing, compare


print("initializing")
model = SegmentedKLM(KLMConfig())
m = KLM(KLMConfig())
dtype = torch.bfloat16
model.import_klm("ckpt/klm-utb-4/step_19000/model.pt")
model.to(dtype)
m.to(dtype)
model.eval()

INTERACT = False

past_ks = torch.load("prefilled/past_k.pt", weights_only=True)
past_vs = torch.load("prefilled/past_v.pt", weights_only=True)
rope = torch.load("prefilled/rope.pt", weights_only=True)

prefill_length = past_ks.shape[-2]
print(f"{prefill_length=}")

correct = 0
total = 0
text = ""

token = torch.tensor([[ord('T')]])

for i in range(500):
    with running("seg", dtype=torch.float32, enable=False):
        rotary_pos_emb = rope[:, :, [i]].to(dtype)
        # rotary_pos_emb = m.body.rotary_embedding(torch.tensor([[i + prefill_length]]))
        # trace_tensor("rotary_pos_emb", rotary_pos_emb)
        residual, q, k, v = model.head(token, rotary_pos_emb)
        # trace_tensor("head_residual", residual)
        # trace_tensor("head_q", q)
        # trace_tensor("head_k", k)
        # trace_tensor("head_v", v)
        new_past_k = []
        new_past_v = []
        for i in range(5):
            # k = torch.from_numpy(k)
            # v = torch.from_numpy(v)
            # n_heads = k.shape[1]
            # head_dim = k.shape[3]
            ks = torch.cat([past_ks[i, :, :, -510:], k], dim=2).to(dtype)
            vs = torch.cat([past_vs[i, :, :, -510:], v], dim=2).to(dtype)
            # ks = torch.cat([past_ks[i], k], dim=2)
            # vs = torch.cat([past_vs[i], v], dim=2)
            with executing(f"seg{i}"):
                trace_tensor(f"ks", ks)
                trace_tensor(f"vs", vs)
                trace_tensor(f"residual", residual)
                trace_tensor(f"q", q)
                trace_tensor(f"k", k)
                trace_tensor(f"v", v)
            new_past_k.append(ks)
            new_past_v.append(vs)
            residual, q, k, v = model.segments[i](residual, rotary_pos_emb, q, ks, vs)
        # k = torch.from_numpy(k)
        # v = torch.from_numpy(v)
        ks = torch.cat([past_ks[5, :, :, -510:], k], dim=2).to(dtype)
        vs = torch.cat([past_vs[5, :, :, -510:], v], dim=2).to(dtype)
        # ks = torch.cat([past_ks[5], k], dim=2)
        # vs = torch.cat([past_vs[5], v], dim=2)
        new_past_k.append(ks)
        new_past_v.append(vs)
        with executing(f"seg5"):
            trace_tensor(f"ks", ks)
            trace_tensor(f"vs", vs)
            trace_tensor(f"residual", residual)
            trace_tensor(f"q", q)
            trace_tensor(f"k", k)
            trace_tensor(f"v", v)
        logits = model.tail(residual, q, ks, vs)
        trace_tensor("tail_logits", logits)
        token = logits.argmax().item()
        print(chr(token), end="", flush=True)
        if INTERACT:
            print(flush=True)
            print(f"[{correct / max(total, 1) * 100:.1f}%] {text}", end="", flush=True)
            c = getch()
            print(c, end="", flush=True)
            text += c
            total += 1
            correct += token == ord(c)
            token = ord(c)
        token = torch.tensor([[token]])
        past_ks = torch.stack(new_past_k, dim=0)
        past_vs = torch.stack(new_past_v, dim=0)

# compare(ignore_shape=True)
