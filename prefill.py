from modeling_klm import KLM, KLMConfig
from segmenting_klm import SegmentedKLM
import torch
from getch import getch
from tracer import running, compare
import numpy as np


device = torch.device("cpu")
model_path = "ckpt/klm-utb-4/step_19000/model.pt"
model = KLM(KLMConfig()).to(device)
model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
model.eval()


prompt = """Nine-year-old Mira lived in a small town nestled between olive hills and a swift, blue river. Her favorite place was her grandmother's tangled old garden, where stone rabbits hid in moss and morning glories curled up every fencepost.

On the evening before her birthday, Mira tiptoed outside after dinner, clutching her sketchbook. The garden shimmered with moonlight. She made her way to the oldest tree, a giant fig, and sat beneath it.

Suddenly, a gentle chime rang in the leaves above. Looking up, Mira spotted a lantern dangling from a high branch. It flickered with a blue-green glow-quite unlike any lantern she'd seen in the market.
"""

tokens = [ord(c) for c in prompt]
print(max(tokens))
print(f"prefilling {len(tokens)} tokens")
tokens = torch.tensor(tokens).unsqueeze(0).to(device)

with torch.no_grad():
    logits, past_k, past_v = model(tokens)

    rope = model.body.rotary_embedding(torch.arange(tokens.shape[1], tokens.shape[1] + 32768, device=device).unsqueeze(0))


torch.save(past_k, "prefilled/past_k.pt")
torch.save(past_v, "prefilled/past_v.pt")
torch.save(rope, "prefilled/rope.pt")

np.save("prefilled/past_k.npy", past_k.cpu().numpy())
np.save("prefilled/past_v.npy", past_v.cpu().numpy())
np.save("prefilled/rope.npy", rope.cpu().numpy())
