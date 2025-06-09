import argparse
import json
import os
import random
import glob
import bisect
import shutil

import numpy as np
import torch
import torch.distributed as dist
import wandb
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_from_disk

from modeling_klm import KLM, KLMConfig


def parse_config(config_path):
    if config_path.endswith('.json'):
        with open(config_path) as f:
            config = json.load(f)
    elif config_path.endswith(('.yml', '.yaml')):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        config["learning_rate"] = float(config["learning_rate"])
    else:
        raise ValueError("Config must be json or yaml format.")
    return config


class UltraTextbookDataset(Dataset):
    def __init__(self, ds_path, max_seq_length):
        # Use streaming False for deterministic shuffling/packing
        self.dataset = load_from_disk(ds_path, keep_in_memory=True)
        self.max_seq_length = max_seq_length
        self.lengths = self.dataset.data.column('length').to_numpy()
        self.texts = self.dataset.data.column('text').to_numpy()
        self.cumsum = np.cumsum(self.lengths)
        self.total_length = self.cumsum[-1]

    def __len__(self):
        return self.total_length // self.max_seq_length

    def __getitem__(self, idx):
        chunk_start = idx * self.max_seq_length
        chunk_end = min((idx + 1) * self.max_seq_length, self.total_length)
        chunk_len = chunk_end - chunk_start
        text_idx = bisect.bisect_right(self.cumsum, chunk_start)
        offset = chunk_start - self.cumsum[text_idx]
        result = []
        need = chunk_len
        while need > 0 and text_idx < len(self.texts):
            text = self.texts[text_idx]
            take = min(len(text) - offset, need)
            result.extend(min(ord(c), 128) for c in text[offset:offset+take])
            need -= take
            text_idx += 1
            offset = 0
        if len(result) < self.max_seq_length:
            result += [0] * (self.max_seq_length - len(result))
        input_ids = result
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long)}

def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()


def save_checkpoint_with_limit(model, state, output_dir, global_step, max_checkpoints):
    ckpt_path = os.path.join(output_dir, f"step_{global_step}")
    os.makedirs(ckpt_path, exist_ok=True)
    torch.save(state, os.path.join(ckpt_path, "state.pt"))
    torch.save(model, os.path.join(ckpt_path, "model.pt"))
    # Clean up old checkpoints
    checkpoint_files = sorted(
        glob.glob(os.path.join(output_dir, f"step_*")),
        key=lambda x: int(x.split("_")[-1]),
    )
    if len(checkpoint_files) > max_checkpoints:
        files_to_remove = checkpoint_files[:-max_checkpoints]
        for f in files_to_remove:
            try:
                shutil.rmtree(f)
            except Exception as e:
                print(f"Failed to remove old checkpoint: {f} due to {e}")
    print(f"Saved checkpoint to {ckpt_path}, total kept: {min(len(checkpoint_files), max_checkpoints)}")

def load_checkpoint(model, optimizer, scheduler, path, map_location, skip_scheduler=False):
    ckpt = torch.load(os.path.join(path, "model.pt"), map_location=map_location, weights_only=False)
    state = torch.load(os.path.join(path, "state.pt"), map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt)
    optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None and 'scheduler' in state and not skip_scheduler:
        scheduler.load_state_dict(state['scheduler'])
    step = state.get('step', 0)
    if skip_scheduler:
        step = 0
    return step

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--skip-scheduler', action='store_true')
    args = parser.parse_args()

    local_rank = setup_distributed()
    config = parse_config(args.config)

    random.seed(config.get('seed', 42))
    np.random.seed(config.get('seed', 42))
    torch.manual_seed(config.get('seed', 42))
    torch.cuda.manual_seed_all(config.get('seed', 42))

    device = torch.device(f'cuda:{local_rank}')
    os.makedirs(config['output_dir'], exist_ok=True)

    # Dataset & DataLoader
    dataset = UltraTextbookDataset(config['data_path'], config['max_seq_length'])
    train_sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'] // dist.get_world_size(),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Model, optimizer, scheduler
    model = KLM(KLMConfig()).to(torch.bfloat16)
    torch.nn.init.normal_(model.lm_head.lm_head.weight, mean=0, std=0.02)
    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 0.01))
    total_steps = config['epochs'] * len(dataloader)
    if "max_steps" in config:
        total_steps = min(total_steps, config['max_steps'])
    warmup_steps = config.get('warmup_steps', int(0.05 * total_steps))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, 0.5 * (1.0 + np.cos(np.pi * (current_step - warmup_steps) / (total_steps - warmup_steps)))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    resume_epoch, resume_step_in_epoch, global_step = 0, 0, 0
    if args.resume:
        resume_epoch, resume_step_in_epoch, global_step = load_checkpoint(
            model, optimizer, scheduler, args.resume, device, skip_scheduler=args.skip_scheduler,
        )

    model.train()

    # Wandb logging setup
    if local_rank == 0:
        wandb.init(
            project=config.get('wandb_project', 'DistributedPretrain'),
            name=config.get('run_name', None),
            config=config
        )

    for epoch in range(config['epochs']):
        train_sampler.set_epoch(epoch)
        if epoch == resume_epoch and resume_step_in_epoch > 0:
            dataloader_iter = iter(dataloader)
            for _ in range(resume_step_in_epoch):
                next(dataloader_iter)
            batch_iter = dataloader_iter
        else:
            batch_iter = iter(dataloader)
        step_in_epoch = 0
        for batch in batch_iter:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            logits, _, _ = model(input_ids)
            # shift labels
            labels = input_ids[:, 1:].contiguous()
            logits = logits[:, :-1, :].contiguous()
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('max_grad_norm', 1.0))
            optimizer.step()
            scheduler.step()

            if local_rank == 0 and global_step % config.get('log_interval', 20) == 0:
                wandb.log({
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "total_tokens": global_step * config['batch_size'] * config['max_seq_length'],
                    "step": global_step,
                }, step=global_step)
                print(f"[Epoch {epoch} Step {global_step}] loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.7f}")

            if local_rank == 0 and global_step % config.get('save_interval', 1000) == 0 and global_step > 0:
                save_checkpoint_with_limit(
                    model.module.state_dict(),
                    {
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'step_in_epoch': step_in_epoch + 1,
                        'global_step': global_step,
                    },
                    os.path.join(config['output_dir'], config['run_name']),
                    global_step,
                    config.get("max_checkpoints", 5)
                )
                print(f"Saved checkpoint")

            global_step += 1
            step_in_epoch += 1

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    # Final checkpoint
    if local_rank == 0:
        final_ckpt = os.path.join(config['output_dir']. config['run_name'], f"{config['run_name']}_final.pt")
        torch.save(model.module.state_dict(), final_ckpt)

    cleanup()

if __name__ == "__main__":
    main()