from contextlib import contextmanager

import torch
import os

traced_tensors = {}
run_name = ""
module_name = ""
trace_dtype = None


def load_traced_tensors():
    global traced_tensors
    if os.path.exists("traced_tensors.pt"):
        traced_tensors = torch.load("traced_tensors.pt", weights_only=False)
    else:
        traced_tensors = {}


@contextmanager
def running(name, enable=True, save=True, dtype=None, no_grad=True):
    if not enable:
        yield
        return
    global run_name, traced_tensors, trace_dtype
    run_name = name
    load_traced_tensors()
    if dtype is not None:
        trace_dtype = dtype
    try:
        if no_grad:
            with torch.no_grad():
                yield
        else:
            yield
    finally:
        run_name = ""
        trace_dtype = None
        if save:
            torch.save(traced_tensors, f"traced_tensors.pt")


@contextmanager
def executing(name):
    global module_name
    module_name += "." + name
    try:
        yield
    finally:
        module_name = module_name.rsplit(".", 1)[0]


def trace_tensor(name, tensor):
    if not run_name:
        return
    global traced_tensors

    name = f"{module_name}.{name}"
    if name not in traced_tensors:
        traced_tensors[name] = {}
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, dtype=trace_dtype)
    if trace_dtype is not None:
        tensor = tensor.to(trace_dtype)
    traced_tensors[name][run_name] = tensor


def compare(eps=1e-5, ignore_shape=False):
    global traced_tensors
    load_traced_tensors()
    if not traced_tensors:
        return

    for name, record in traced_tensors.items():
        if len(record) < 2:
            print(f"[SKIP] {name} has less than 2 tensors, skipping comparison.")
            continue

        keys = list(record.keys())
        first_key = keys[0]
        first_tensor = record[first_key]
        if not isinstance(first_tensor, torch.Tensor):
            first_tensor = torch.from_numpy(first_tensor)

        all_matched = True
        for key in keys[1:]:
            other_tensor = record[key]
            if not isinstance(other_tensor, torch.Tensor):
                other_tensor = torch.from_numpy(other_tensor)
            if not first_tensor.shape == other_tensor.shape:
                if not ignore_shape:
                    print(f"[WRONG] Shape mismatch in {name}: {first_key} vs {key} {first_tensor.shape=} vs {other_tensor.shape=}")
                    all_matched = False
                    continue
                else:
                    other_tensor = other_tensor.reshape(first_tensor.shape)
            if not torch.allclose(first_tensor, other_tensor, atol=eps, rtol=eps):
                a = first_tensor
                b = other_tensor
                diff = a - b
                print(f"[WRONG] Tensor mismatch in {name}: {first_key} vs {key} off by {(diff.norm() / a.norm() * 100).item():.2f}% {a.norm().item()=} {b.norm()=} {diff.norm()=} {diff.abs().max()=}")
                all_matched = False
        if all_matched:
            print(f"[OK] Tensor {name} all matched.")


    # Reset the traced tensors after comparison
    traced_tensors = {}
