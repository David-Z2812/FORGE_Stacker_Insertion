import torch

ckpt_path = "/home/david/IsaacLab/cluster_data/Forge.pth"

# explizit erlauben, vollständiges Pickle zu laden
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

# move model weights to cpu
if "model" in ckpt:
    for k, v in ckpt["model"].items():
        if isinstance(v, torch.Tensor):
            ckpt["model"][k] = v.cpu()

# move central_val_stats
if "central_value_stats" in ckpt and isinstance(ckpt["central_value_stats"], dict):
    for k, v in ckpt["central_value_stats"].items():
        if isinstance(v, torch.Tensor):
            ckpt["central_value_stats"][k] = v.cpu()

# move obs_rms (running mean/std for observations)
if "obs_rms" in ckpt and isinstance(ckpt["obs_rms"], dict):
    for k, v in ckpt["obs_rms"].items():
        if isinstance(v, torch.Tensor):
            ckpt["obs_rms"][k] = v.cpu()

# save new CPU-only checkpoint
cpu_ckpt_path = "/home/david/IsaacLab/cluster_data/Forge_cpu_clean.pth"
torch.save(ckpt, cpu_ckpt_path)

print(f"✅ Saved CPU-only checkpoint: {cpu_ckpt_path}")

