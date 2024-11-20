import os
import json

import torch
from torch import nn
from torch.nn.utils import remove_weight_norm

from .autoencoder import create_autoencoder_from_config


def remove_all_weight_norm(model):
    for name, module in model.named_modules():
        if hasattr(module, "weight_g"):
            remove_weight_norm(module)


def load_vae(ckpt_path, config_file, remove_weight_norm=False):

    # Load the model configuration
    with open(config_file) as f:
        model_config = json.load(f)

    # Create the model from the configuration
    model = create_autoencoder_from_config(model_config)

    # Load the state dictionary from the checkpoint
    model_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]

    # Strip the "autoencoder." prefix from the keys
    model_dict = {
        key[len("autoencoder.") :]: value
        for key, value in model_dict.items()
        if key.startswith("autoencoder.")
    }

    # Load the state dictionary into the model
    model.load_state_dict(model_dict)

    # Remove weight normalization
    if remove_weight_norm:
        remove_all_weight_norm(model)

    # Set the model to evaluation mode
    model.eval()

    return model


class Autoencoder(nn.Module):
    def __init__(
        self,
        ckpt_path,
        config_file="./stable_vae/configs/default.json",
        model_type="stable_vae",
        quantization_first=True,
    ):
        super(Autoencoder, self).__init__()
        self.model_type = model_type
        if self.model_type == "stable_vae":
            model = load_vae(ckpt_path, config_file)
        else:
            raise NotImplementedError(f"Model type not implemented: {self.model_type}")
        self.ae = model.eval()
        self.quantization_first = quantization_first
        print(f"Autoencoder quantization first mode: {quantization_first}")

    @torch.no_grad()
    def forward(self, audio=None, embedding=None):
        if self.model_type == "stable_vae":
            return self.process_stable_vae(audio, embedding)
        else:
            raise NotImplementedError(f"Model type not implemented: {self.model_type}")

    def process_stable_vae(self, audio=None, embedding=None):
        if audio is not None:
            z = self.ae.encoder(audio)
            if self.quantization_first:
                z = self.ae.bottleneck.encode(z)
            return z
        if embedding is not None:
            z = embedding
            if self.quantization_first:
                audio = self.ae.decoder(z)
            else:
                z = self.ae.bottleneck.encode(z)
                audio = self.ae.decoder(z)
            return audio
        else:
            raise ValueError("Either audio or embedding must be provided.")
