import numpy as np
import torch
from tqdm import tqdm
from tracer import trace_tensor, running, compare
import onnxruntime as ort

head = ort.InferenceSession("klm_models/segmented/seg_head.onnx")
segments = []
for i in range(5):
    segments.append(ort.InferenceSession(f"klm_models/segmented/seg_{i}.onnx"))
tail = ort.InferenceSession("klm_models/segmented/seg_tail.onnx")

dtype = torch.float32
past_ks = torch.load("prefilled/past_k.pt", weights_only=True)
past_vs = torch.load("prefilled/past_v.pt", weights_only=True)
rope = torch.load("prefilled/rope.pt", weights_only=True)

prefill_length = past_ks.shape[-2]
print(f"{prefill_length=}")

token = torch.tensor([[ord(' ')]])

for i in range(100):
    with running("seg", dtype=torch.float32, enable=False):
        rotary_pos_emb = rope[:, :, [i]].to(dtype).numpy()
        # rotary_pos_emb = m.body.rotary_embedding(torch.tensor([[i + prefill_length]]))
        trace_tensor("rotary_pos_emb", rotary_pos_emb)
        residual, q, k, v = head.run(None, {"input_ids": token.numpy(), "rotary_pos_emb": rotary_pos_emb})
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
            ks = torch.cat([past_ks[i, :, :, -510:], k], dim=2).to(dtype)
            vs = torch.cat([past_vs[i, :, :, -510:], v], dim=2).to(dtype)
            # ks = torch.cat([past_ks[i], k], dim=2)
            # vs = torch.cat([past_vs[i], v], dim=2)
            trace_tensor(f"seg{i}_ks", ks)
            trace_tensor(f"seg{i}_vs", vs)
            new_past_k.append(ks)
            new_past_v.append(vs)
            residual, q, k, v = segments[i].run(None, {"residual.1": residual, "rotary_pos_emb": rotary_pos_emb, "q.3": q, "ks": ks.numpy(), "vs": vs.numpy()})
            trace_tensor(f"seg{i}_residual", residual)
            trace_tensor(f"seg{i}_q", q)
            trace_tensor(f"seg{i}_k", k)
            trace_tensor(f"seg{i}_v", v)
        k = torch.from_numpy(k)
        v = torch.from_numpy(v)
        ks = torch.cat([past_ks[5, :, :, -510:], k], dim=2).to(dtype)
        vs = torch.cat([past_vs[5, :, :, -510:], v], dim=2).to(dtype)
        # ks = torch.cat([past_ks[5], k], dim=2)
        # vs = torch.cat([past_vs[5], v], dim=2)
        new_past_k.append(ks)
        new_past_v.append(vs)
        logits = tail.run(None, {"residual": residual, "q": q, "ks": ks.numpy(), "vs": vs.numpy()})[0]
        trace_tensor("tail_logits", logits)
        token = logits.argmax().item()
        print(chr(token), end="", flush=True)
        token = torch.tensor([[token]])
        past_ks = torch.stack(new_past_k, dim=0)
        past_vs = torch.stack(new_past_v, dim=0)

# compare(ignore_shape=True)
