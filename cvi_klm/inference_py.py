from cvi_klm import KLM
import numpy as np
import torch
from tqdm import tqdm
from tracer import trace_tensor, running
from getch import getch


INTERACT = False


print("initializing")
model = KLM("/home/debian/workspace/klm_models")

past_ks = torch.load("prefilled/past_k.pt", weights_only=True)
past_vs = torch.load("prefilled/past_v.pt", weights_only=True)
rope = torch.load("prefilled/rope.pt", weights_only=True)

token = torch.tensor([[ord('T')]], dtype=torch.long)
text = ""
correct = 0
total = 0

for i in range(500):
    with running("cvi_klm", enable=False):
        rotary_pos_emb = rope[:, :, [i]].numpy()
        trace_tensor("rotary_pos_emb", rotary_pos_emb)
        residual, q, k, v = model.call_head(token.numpy(), rotary_pos_emb)
        trace_tensor("head_residual", residual)
        trace_tensor("head_q", q)
        trace_tensor("head_k", k)
        trace_tensor("head_v", v)
        new_past_k = []
        new_past_v = []
        for i in range(5):
            k = torch.from_numpy(k)
            v = torch.from_numpy(v)
            # n_heads = k.shape[1]
            # head_dim = k.shape[3]
            ks = torch.cat([past_ks[i, :, :, -510:], k], dim=2).contiguous()
            vs = torch.cat([past_vs[i, :, :, -510:], v], dim=2).contiguous()
            trace_tensor(f"seg{i}_ks", ks)
            trace_tensor(f"seg{i}_vs", vs)
            new_past_k.append(ks)
            new_past_v.append(vs)
            residual, q, k, v = model.call_segment(i, residual, rotary_pos_emb, q, ks.numpy(), vs.numpy())
            trace_tensor(f"seg{i}_residual", residual)
            trace_tensor(f"seg{i}_q", q)
            trace_tensor(f"seg{i}_k", k)
            trace_tensor(f"seg{i}_v", v)
            # print(f"{residual.shape=} {q.shape=} {k.shape=} {v.shape=}")
        k = torch.from_numpy(k)
        v = torch.from_numpy(v)
        ks = torch.cat([past_ks[5, :, :, -510:], k], dim=2).contiguous()
        vs = torch.cat([past_vs[5, :, :, -510:], v], dim=2).contiguous()
        new_past_k.append(ks)
        new_past_v.append(vs)
        logits = model.call_tail(residual, q, ks.numpy(), vs.numpy())
        trace_tensor("tail_logits", logits)
        token = logits.argmax()
        print(chr(token), end="", flush=True)
        if INTERACT:
            print(flush=True)
            print(f"[{correct / max(total, 1) * 100:.1f}%] {text}", end="", flush=True)
            c = getch()
            print(c, end="", flush=True)
            text += c
            total += 1
            correct += token.item() == ord(c)
            token = ord(c)
            # print(text, end="", flush=True)
        token = torch.tensor([[token]], dtype=torch.long)
        past_ks = torch.stack(new_past_k, dim=0)
        past_vs = torch.stack(new_past_v, dim=0)
