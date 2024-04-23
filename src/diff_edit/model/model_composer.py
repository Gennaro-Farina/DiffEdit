import logging

import torch

from transformers import CLIPTextModel, CLIPTokenizer
# Imports from diffusers and transformers libraries
from diffusers import (AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel,
                       StableDiffusionInpaintPipeline, )

from diff_edit.model.diff_edit_model import DiffEdit


class ModelComposer:
    def __init__(self, vae_model: str, tokenizer: str, text_encoder: str, unet: str, inpainting: str,
                 scheduler: LMSDiscreteScheduler, torch_dev: str = "cpu"):
        """
        The ModelComposer class is used to compose the DiffEdit model from the provided components.

        Args: vae_model (str): The path to the VAE model. tokenizer (str): The path to the tokenizer model.
        text_encoder (str): The path to the text encoder model. unet (str): The path to the UNet model. inpainting (
        str): The path to the inpainting model. scheduler (str): The path to the scheduler model. torch_dev (str):
        The device to use for the model. Default is "cpu". Can be "cpu", "mps", "cuda" or "best". For "best",
        the best available device is selected, the order of preference is "mps", "cuda" and "cpu".
        """
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.unet = unet
        self.inpainting = inpainting
        self.scheduler = scheduler

        # Select the best device available
        if torch_dev == "best":
            if torch.backends.mps.is_built():
                torch_dev = "mps"
            elif torch.cuda.is_available():
                torch_dev = "cuda"
            else:
                torch_dev = "cpu"

        if torch_dev == "mps":
            if not torch.backends.mps.is_built():
                logging.warning("MPS is not available, falling back to CUDA.")
                # fallback to cuda if mps is not available
                if torch.cuda.is_available():
                    logging.info("CUDA is available, using CUDA.")
                    torch_dev = "cuda"
                else:
                    logging.warning("CUDA is not available, falling back to CPU.")
                    torch_dev = "cpu"
        elif torch_dev == "cuda":
            if not torch.cuda.is_available():
                logging.warning("CUDA is not available, falling back to CPU.")
                torch_dev = "cpu"
        elif torch_dev != "cpu":
            raise Exception("Invalid device, please use cpu, mps or cuda. Received: " + str(torch_dev) + " instead."
                                )
        self.torch_dev = torch_dev

    def compose(self):
        tokenizer = CLIPTokenizer.from_pretrained(self.tokenizer)
        text_encoder = CLIPTextModel.from_pretrained(self.text_encoder)

        vae = AutoencoderKL.from_pretrained(self.vae_model)
        unet = UNet2DConditionModel.from_pretrained(self.unet, subfolder="unet")

        inpainting = StableDiffusionInpaintPipeline.from_pretrained(self.inpainting)

        return DiffEdit(tokenizer, text_encoder, vae, unet, inpainting, self.scheduler, self.torch_dev)

    def __str__(self):
        return f"ModelComposer(vae_model={self.vae_model}, tokenizer={self.tokenizer}, text_encoder=" \
               f"{self.text_encoder}, unet={self.unet}, inpainting={self.inpainting}, scheduler={self.scheduler}, " \
               f"torch_dev={self.torch_dev})"

    def __eq__(self, other):
        if not isinstance(other, ModelComposer):
            return False
        return self.vae_model == other.vae_model and self.tokenizer == other.tokenizer and \
            self.text_encoder == other.text_encoder and self.unet == other.unet and \
            self.inpainting == other.inpainting and self.scheduler == other.scheduler and \
            self.torch_dev == other.torch_dev
