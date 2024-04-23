import logging

import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from diff_edit.model.constants import TORCH_SEED, IMG_RESIZE_DIM, VAE_CONST


class MaskGenerator:
    def __init__(self, unet, scheduler, tokenizer, text_encoder, image_processor, torch_device):
        self.unet = unet
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.image_processor = image_processor
        self.torch_device = torch_device
        self.rough_mask = None
        self.processed_mask = None

        # For a given reference prompt and a query prompt;
    # Run the diffusion process 10 times; Calculating a "noise distance" for each sample
    def get_embedding_for_prompt(self, prompt):
        """
            This method gets the embedding for a prompt.

            prompt (str): The prompt to get the embedding for.
        """
        max_length = self.tokenizer.model_max_length
        tokens = self.tokenizer([prompt], padding="max_length", max_length=max_length, truncation=True,
                                return_tensors="pt")
        with torch.no_grad():  # we are using for inference, no gradients needed
            return self.text_encoder(tokens.input_ids.to(self.torch_device))[0]


    def calc_diffedit_samples(self, encoded: torch.Tensor, prompt1: str, prompt2: str, n: int = 2,
                               seed: int = TORCH_SEED, **kwargs):
        """
        This method calculates a 'noise distance' for each sample.

        encoded (torch.Tensor): The encoded tensor to use for the calculation.
        prompt1 (str): The first prompt to use for the calculation.
        prompt2 (str): The second prompt to use for the calculation.
        n (int): The number of times to run the diffusion process.
        seed (int): The seed to use for reproducibility.

        return (torch.Tensor): The 'noise distance' for each sample.
        """
        diffs = []
        logging.debug(f"Running {n} times the diffusion process to calculate a 'noise distance' for each sample.")
        # So we can reproduce mask generation we generate a list of n seeds
        torch.manual_seed(seed)
        seeds = torch.randint(0, 2 ** 62, (n,)).tolist()
        for i in tqdm(range(n)):
            seed = seeds[i]  # Important to use same seed for the two noise samples
            emb1 = self.get_embedding_for_prompt(prompt1)
            _im1, n1 = self.predict_noise(emb1, encoded, seed)
            emb2 = self.get_embedding_for_prompt(prompt2)
            _im2, n2 = self.predict_noise(emb2, encoded, seed)

            # Aggregate the channel components by taking the Euclidean distance.
            diffs.append((n1 - n2)[0].pow(2).sum(dim=0).pow(0.5)[None])
        all_masks = torch.cat(diffs)
        return all_masks

        # Given an image latent and two prompts; generate a grayscale diff by sampling the noise predictions
        # between the prompts.

    def calc_diffedit_diff(self, im_latent: torch.Tensor, p1: str, p2: str, n: int, seed: int = TORCH_SEED):
        """
        This method generates a grayscale diff by sampling the noise predictions between the prompts.

        im_latent (torch.Tensor): The image latent to use for the generation.
        p1 (str): The first prompt to use for the generation.
        p2 (str): The second prompt to use for the generation.
        n (int): The number of times to run the diffusion process.
        seed (int): The seed to use for reproducibility.

        return (PIL.Image): The generated grayscale diff.
        """
        m = self.calc_diffedit_samples(im_latent, p1, p2, n, seed)
        m = m.mean(axis=0)  # average samples together
        m = (m - m.min()) / (m.max() - m.min())  # rescale to interval [0,1]
        m = (m * 255.).cpu().numpy().astype(np.uint8)  # rescale to [0, 255]
        m = Image.fromarray(m)
        return m

    # Try to improve the mask through convolutions, this method assumes m is a PIL object containing a grayscale 'diff'
    def process_diffedit_mask(self, mask: Image, threshold: float = 0.35):
        """
        This method tries to improve the mask through convolutions.

        mask (PIL.Image): The mask to improve.
        threshold (float): The threshold to use for the improvement.

        return (PIL.Image): The improved mask.
        """
        mask = np.array(np.array(mask).astype(np.float32))  # convert to numpy
        mask = cv2.GaussianBlur(mask, (5, 5), 1)
        mask = (mask > (255. * threshold)).astype(np.float32) * 255  # threshold and convert back to [0,255]
        mask = Image.fromarray(mask.astype(np.uint8))  # convert to PIL
        return mask

    # Given an image latent and two prompts; generate a binarized mask (PIL) appropriate for inpainting
    def calc_diffedit_mask(self, im_latent: torch.Tensor, p1: str, p2: str, n: int = 10, seed: int = TORCH_SEED):
        """
        This method generates a binarized mask appropriate for inpainting.

        im_latent (torch.Tensor): The image latent to use for the generation.
        p1 (str): The first prompt to use for the generation.
        p2 (str): The second prompt to use for the generation.
        n (int): The number of times to run the diffusion process.
        seed (int): The seed to use for reproducibility.
        """
        m = self.calc_diffedit_diff(im_latent, p1, p2, n, seed)
        self.rough_mask = m.copy()
        m = self.process_diffedit_mask(m)
        self.processed_mask = m.copy()
        m = m.resize((IMG_RESIZE_DIM, IMG_RESIZE_DIM))
        return m
    # Given an image latent and two prompts; generate a binarized mask (PIL) appropriate for inpainting

    # Given a starting image latent and a prompt; predict the noise that should be removed to transform
    # the noised source image to a denoised image guided by the prompt.
    def predict_noise(self, text_embeddings: torch.Tensor, im_latents: torch.Tensor, seed: int = TORCH_SEED,
                       guidance_scale: float = 7., strength: float = 0.5, num_inference_steps: int = 50):
        """
        This method predicts the noise that should be removed to transform the noised source image to a denoised image

        text_embeddings (torch.Tensor): The text embeddings to use for the prediction.
        im_latents (torch.Tensor): The image latents to use for the prediction.
        seed (int): The seed to use for reproducibility.
        guidance_scale (float): The scale to use for guidance.
        strength (float): The strength to use for the prediction.
        num_inference_steps (int): The number of inference steps to use for the prediction.

        return (tuple): A tuple containing the denoised image and the predicted noise.
        """
        # num_inference_steps is Number of denoising steps
        torch.manual_seed(seed)  # initial latent noise

        uncond = self.get_embedding_for_prompt('')  # unconditional prompt
        text_embeddings = torch.cat([uncond, text_embeddings])  # concatenate unconditional and conditional prompts

        # Prep Scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor([timesteps] * 1, device=self.torch_device).float()  # [timesteps] * 1 * 1

        noise = torch.randn_like(im_latents)
        latents = self.scheduler.add_noise(im_latents, noise, timesteps=timesteps)
        latents = latents.to(self.torch_device).float()

        t_start = max(num_inference_steps - init_timestep + offset, 0)
        timesteps = self.scheduler.timesteps[t_start:].to(self.torch_device)

        noisy_latent = latents.clone()

        noise_pred = None
        for i, tm in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, tm)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, tm, encoder_hidden_states=text_embeddings)["sample"]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

            u = noise_pred_uncond
            g = guidance_scale
            t = noise_pred_text

            # perform guidance
            noise_pred = u + g * (t - u)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, tm, latents).prev_sample

        return self.image_processor.latents2imgs(latents, VAE_CONST)[0], noise_pred