from cvi_klm import KLM
import numpy as np


print("loading models")
model = KLM("/home/debian/workspace/klm_models")

past_ks = np.load("prefilled/past_k.npy")
past_vs = np.load("prefilled/past_v.npy")
rope = np.load("prefilled/rope.npy")

token = np.array([[ord(' ')]], dtype=np.int64)

prefill_length = past_ks.shape[-2]
ptr = 0
buffer = ""
predict = 1
char = ""
eos = ['\n']
eos_replace = " "

def pred():
    global ptr, buffer, predict, past_ks, past_vs, rope, token, char
    model.move_context_ptr(prefill_length + ptr)
    for i in range(predict):
        rotary_pos_emb = rope[:, :, [i + ptr]]
        logits = model.call_model(token, np.ascontiguousarray(rotary_pos_emb))
        token = np.argmax(logits)
        char = chr(token)
        buffer += char
        token = np.array([[token]], dtype=np.int64)
        if char in eos:
            break

print("## READY ##")

while True:
    inp = input()
    do_predict = False
    if not inp:
        continue
    if inp[0] == "@":
        cmd = inp[1:]
        if cmd == "accept":
            ptr += len(buffer)
            buffer = ""
        elif cmd == "back":
            if ptr > 0:
                ptr -= 1
        elif cmd == "clear":
            ptr = 0
        elif cmd.startswith("pred"):
            if len(cmd) > 5:
                predict = int(cmd[5:])
            else:
                predict = 8
            pred()
            if char in eos:
                buffer = buffer[:-1]
                token = np.array([[ord(eos_replace)]], dtype=np.int64)
            print(buffer)
        elif cmd == "debug":
            print(f"{buffer=} {ptr=} {predict=} {char=} {token=}")
            print(f"{past_ks.shape=} {past_vs.shape=} {rope.shape=} {prefill_length=}")
    elif inp[0] == "#":
        c = inp[1]
        token = np.array([[ord(c)]], dtype=np.int64)
        predict = 1
        buffer = ""
        pred()
        ptr += 1
