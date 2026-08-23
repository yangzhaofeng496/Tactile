import torch
from collections import Counter

from model import VQVAEForceEncoder

CACHE_PATH = "/home/yang/TactileEncoder/outputs/act_cache/act_cache.pt"
CKPT_PATH = "/home/yang/TactileEncoder/TactileSelfencoder/vqvae_checkpoints/decoder_mlp/16tokens/checkpoint_epoch_200.pth"
BATCH_SIZE = 256

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

encoder = VQVAEForceEncoder(vqvae_checkpoint_path=CKPT_PATH).to(device)
encoder.eval()

cache = torch.load(CACHE_PATH, map_location="cpu")
print(f"total windows: {len(cache)}")

counter = Counter()
all_indices = []
with torch.no_grad():
    keys = sorted(cache.keys())
    for i in range(0, len(keys), BATCH_SIZE):
        batch = torch.stack(
            [cache[k]["tactile_history"] for k in keys[i:i + BATCH_SIZE]]
        ).float().to(device)
        _, token_ids = encoder(batch, return_token_id=True)
        token_ids = token_ids.cpu().reshape(-1).tolist()
        all_indices.extend(token_ids)
        counter.update(token_ids)

n = len(all_indices)
print(f"\ntoken count: {len(counter)} / 16")
print("=" * 40)
for token_id in sorted(counter.keys()):
    count = counter[token_id]
    print(f"token {token_id:2d}: {count:7d}  ({count / n * 100:6.2f}%)")
print("=" * 40)
print(f"total: {n}")
