import os
import glob
import argparse

import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
import numpy as np
import librosa
from tqdm import tqdm

from stable_vae import Autoencoder


class SimpleAudioDataset(Dataset):
    def __init__(self, datapath: str, audio_len: int = 10, sr: int = 24000):
        assert os.path.exists(datapath), f"Path {datapath} does not exist."
        self.datapath = datapath
        self.audio_len = audio_len
        self.sr = sr
        self.dataset = self.get_files()

    def get_files(self):
        files = list(glob.glob(os.path.join(self.datapath, "*denoised.wav")))
        if len(files) == 0:
            files = list(glob.glob(os.path.join(self.datapath, "*.mp4")))
        assert len(files) > 0, f"No audio files found in {self.datapath}."
        return files

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        f = self.dataset[idx]
        audio, sr = librosa.load(f, sr=self.sr)
        desired_length = self.audio_len * sr
        if len(audio) < desired_length:
            padding = desired_length - len(audio)
            audio = np.pad(audio, (0, padding), mode="constant")
        if np.abs(audio).max() > 1:
            audio /= np.abs(audio).max()
        return (
            torch.tensor(audio)[None].float(),
            os.path.splitext(os.path.basename(f))[0],
        )


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup a feature folder:
    if rank == 0:
        os.makedirs(args.features_path, exist_ok=True)
        os.makedirs(
            os.path.join(args.features_path, f"stable_vae_features_{args.sr}hz"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(
                args.features_path, f"stable_vae_features_{args.sr}hz_unitstd"
            ),
            exist_ok=True,
        )
    vae = Autoencoder(
        "./pretrained_models/audio-vae.pt",
        model_type="stable_vae",
        quantization_first=True,
    ).to(device)
    dataset = SimpleAudioDataset(args.data_path, args.audio_len, args.sr)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    latents = []
    fns = []
    for x, fn in tqdm(loader):
        if type(fn) == list:
            fn = fn[0]
        x = x.to(device)
        with torch.no_grad():
            for i in range(0, x.size(-1), args.sr * args.audio_len):
                audio = x[..., i : i + args.sr * args.audio_len]
                if audio.size(-1) < args.sr * args.audio_len:
                    padding = args.sr * args.audio_len - audio.size(-1)
                    audio = torch.nn.functional.pad(audio, (0, padding))
                z = vae(audio=audio)[0].cpu()
                latents.append(z)
                fns.append(f"{fn}_{int(i / args.sr * 1000)}")
                torch.save(
                    z,
                    f"{args.features_path}/stable_vae_features_{args.sr}hz/{fn}_{int(i / args.sr * 1000)}.pt",
                )

    latents_combined = torch.cat(latents, dim=-1)
    scaler = 1 / latents_combined.std().item()

    for i, x in enumerate(latents):
        x.mul_(scaler)
        torch.save(
            x,
            f"{args.features_path}/stable_vae_features_{args.sr}hz_unitstd/{fns[i]}.pt",
        )


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--features-path", type=str, default="features")
    parser.add_argument("--audio_len", type=int, default=5)
    parser.add_argument("--sr", type=int, default=24000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["stable_vae"], default="stable_vae")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
