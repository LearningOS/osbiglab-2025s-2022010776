from cvi_klm import KLM
import numpy as np
# from tracer import trace_tensor, running, executing
# import torch
from tqdm import tqdm

print("initializing")
model = KLM("/home/debian/workspace/klm_models")

past_ks = np.load("prefilled/past_k.npy")
past_vs = np.load("prefilled/past_v.npy")
rope = np.load("prefilled/rope.npy")

token = np.array([[ord('T')]], dtype=np.long)
model.set_kv_cache(past_ks, past_vs)

# model.trace(trace_tensor, executing)

for i in range(500):
    # with running("cvi_klm"):
    logits = model.call_model(token, np.ascontiguousarray(rope[:, :, [i]]))
    # print(logits.dtype, logits.shape)
    token = logits.argmax()
    print(f"{chr(token)}", end="", flush=True)
    token = np.array([[token]], dtype=np.long)
# input_ids = np.array([[token]], dtype=np.long)
