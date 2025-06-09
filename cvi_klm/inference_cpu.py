from modeling_klm import KLM, KLMConfig
from segmenting_klm import SegmentedKLM
import torch
from getch import getch
from tracer import running, compare

USE_KV_CACHE = True
USE_SEGMENTED = False
INTERACT = False
PREDICT = 100

device = torch.device("cuda:0")
model_path = "../ckpt/klm-utb-4/step_19000/model.pt"
model = KLM(KLMConfig()).to(torch.bfloat16).to(device)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

if USE_SEGMENTED:
    seg = SegmentedKLM(model.config)
    seg.import_klm(model_path)
    seg.to(torch.bfloat16).to(device)
    seg.eval()

prompt = ""
inp = prompt + "Hi,"
correct = 0
total = 0
pred = ""
while True:
    if INTERACT:
        print(f"[{correct / max(total, 1) * 100:.1f}%] {inp[len(prompt):]}", end="", flush=True)
        new_inp = getch()
        total += 1
        correct += pred == new_inp
        print(new_inp, end="", flush=True)
    else:
        new_inp = pred
    inp += new_inp
    tokens = [ord(c) for c in inp]
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)

    past_k = None
    past_v = None
    position_ids = torch.arange(0, tokens.shape[1], device=device).unsqueeze(0)
    for i in range(PREDICT):
        with torch.no_grad():
            if not USE_KV_CACHE:
                past_k = None
                past_v = None
            # breakpoint()
            with running(f"{USE_KV_CACHE=}&{USE_SEGMENTED=}", enable=i == PREDICT - 1):
                if past_k is None or not USE_SEGMENTED:
                    logits, past_k, past_v = model(tokens, past_k=past_k, past_v=past_v, position_ids=position_ids)
                else:
                    rot = model.body.rotary_embedding(position_ids)
                    logits, past_k, past_v = seg(tokens, rot, past_k, past_v)
            logits = logits[:, -1, :]
            token = torch.argmax(logits, dim=-1).item()
            if USE_KV_CACHE:
                tokens = torch.tensor([[token]], device=device)
                position_ids = position_ids[:, -1:] + 1
            else:
                tokens = torch.cat([tokens, torch.tensor([[token]], device=device)], dim=1)
                position_ids = torch.arange(0, tokens.shape[1], device=device).unsqueeze(0)
            pred = chr(token)
            print(pred.replace(' ', '#' if INTERACT else ' '), end="", flush=True)
    print(flush=True)
    if not INTERACT:
        break

# compare(eps=1e-2)
