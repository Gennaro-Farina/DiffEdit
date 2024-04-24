# Standard library imports
import logging
import os
import pickle
from typing import Union

# Related third-party imports
import torch
from PIL import Image
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    LMSDiscreteScheduler,
    StableDiffusionInpaintPipeline,
)
from transformers import CLIPTextModel, CLIPTokenizer

# local application/library specific imports
from diff_edit.model.constants import VAE_CONST, TORCH_SEED, IMG_RESIZE_DIM
from diff_edit.model.image_processing import ImageProcessor
from diff_edit.model.mask_generation import MaskGenerator
from diff_edit.model.mask_inpainting import Inpainter

torch.manual_seed(TORCH_SEED)


class DiffEdit:
    def __init__(self, tokenizer: CLIPTokenizer = None,
                 text_encoder: CLIPTextModel = None,
                 vae: AutoencoderKL = None,
                 unet: UNet2DConditionModel = None,
                 inpainting: StableDiffusionInpaintPipeline = None,
                 scheduler: LMSDiscreteScheduler = None,
                 torch_device: str = None,
                 ):
        """
        This class represents the DiffEdit model. It is a wrapper around the components of the model, such as the
        VAE, the UNet, the tokenizer and the text encoder. It provides methods to generate masks, inpaint images and
        perform the DiffEdit algorithm.
        """
        # attributes for the components
        self.tokenizer: Union[CLIPTokenizer, None] = tokenizer
        self.text_encoder: Union[CLIPTextModel, None] = text_encoder

        self.vae: Union[AutoencoderKL, None] = vae
        self.unet: Union[UNet2DConditionModel, None] = unet
        self.inpainting: Union[StableDiffusionInpaintPipeline, None] = inpainting

        self.scheduler: Union[LMSDiscreteScheduler, None] = scheduler

        # attributes for the device
        self.torch_device: Union[str, None] = torch_device
        logging.debug(f"Setting the device to {torch_device}")
        self.to(torch_device)
        logging.debug(f"Device set to {torch_device}")

        self.image_processor = ImageProcessor(vae, torch_device)
        self.mask_generator = MaskGenerator(unet, scheduler, tokenizer, text_encoder, self.image_processor,
                                            torch_device)
        self.inpainter = Inpainter(inpainting, torch_device)

    def to(self, device: str):
        """
             This method moves the components of the model to the specified device.
             device (str): The device to move the components to. Valid values are "cpu", "mps" and "cuda".

             return (DiffEdit): The DiffEdit model with the components moved to the specified device.
         """
        # move the components to the device
        self.vae = self.vae.to(device)
        self.text_encoder = self.text_encoder.to(device)
        self.unet = self.unet.to(device)
        self.inpainting = self.inpainting.to(device)
        return self

    def get_mask(self, im_path: str, p1: str, p2: str, seed: int = TORCH_SEED, n: int = 10):
        """
            This method returns the mask generated by the DiffEdit algorithm.

            im_path (str): The path to the image to edit.
            p1 (str): The prompt to remove.
            p2 (str): The prompt to add.
            n (int): The number of iterations to perform to get the mask. Each iteration is a diffusion process.
            seed (int): The seed to use for reproducibility.

            return (list): A list containing the processed mask, the rough mask, the blended mask (visualization only)
        """
        with Image.open(im_path) as im:
            im = im.resize((IMG_RESIZE_DIM, IMG_RESIZE_DIM))
            im_latent = self.image_processor.img2latent(im, VAE_CONST)

        mask = self.mask_generator.calc_diffedit_mask(im_latent, p1, p2, n, seed)
        return self.mask_generator.processed_mask, self.mask_generator.rough_mask, \
            self.image_processor.get_blended_mask(
            im, mask)

    def refine_mask(self, mask: Image, n=10, **kwargs):
        """
            This method refines the mask generated by the DiffEdit algorithm.
            E.G. Mix togheter mask coming from different prompts pairs. To do this, we can use the mask from the
            previous prompt pair as a starting point for the next prompt pair, having the same prompt for the object
            to remove.

            E.G.
                mask1 = get_mask(im_path, "a guitar", "an ukulele")
                mask2 = get_mask(im_path, "a guitar", "a violin")
        """
        pass

    def create_mask(self, im_path: str, p1: str, p2: str, n: int = 10, seed: int = TORCH_SEED):
        """
            This method creates a mask for the specified image using the DiffEdit algorithm.

            im_path (str): The path to the image to edit.
            p1 (str): The prompt to remove.
            p2 (str): The prompt to add.
            n (int): The number of iterations to perform to get the mask. Each iteration is a diffusion process.
            seed (int): The seed to use for reproducibility.

            return (Image): The mask generated by the DiffEdit algorithm.
        """

        if not os.path.exists(im_path):
            raise ValueError(f"Image path {im_path} does not exist. Check the provided path")

        logging.info(f"Obtaining the mask by running the diffusion process {n} times.")
        mask, rough_mask, blended_mask = self.get_mask(im_path, p1, p2, seed, n)

        return mask, rough_mask, blended_mask

    def save_mask(self, mask: Image, rough_mask: Image, blended_mask: Image, im_path: str, workdir: str = "./"):
        """
        This method saves the mask and the image to disk.

        mask (Image): The mask to save.
        rough_mask (Image): The rough mask to save.
        blended_mask (Image): The blended mask to save.
        im_path (str): The path to the image to save.
        workdir (str): The directory to save the mask. Optional, default is "./".
        """
        # save the mask to disk
        with open(os.path.join(workdir, "mask.bmp"), "wb") as f:
            mask.save(f)
        with open(os.path.join(workdir, "rough_mask.bmp"), "wb") as f:
            rough_mask.save(f)
        with open(os.path.join(workdir, "blended_mask.bmp"), "wb") as f:
            blended_mask.save(f)
        with open(os.path.join(workdir, "original_image.bmp"), "wb") as f:
            Image.open(im_path).resize((IMG_RESIZE_DIM, IMG_RESIZE_DIM)).save(f)

    def save_inpainted_image(self, inpainted_image: Image, workdir: str = "./"):
        """
        This method saves the inpainted image to disk.

        inpainted_image (Image): The inpainted image to save.
        workdir (str): The directory to save the inpainted image. Optional, default is "./".
        """
        # save the inpainted image to disk
        with open(os.path.join(workdir, "inpainted_image.bmp"), "wb") as f:
            inpainted_image.save(f)


    def load_mask(self, workdir: str = "./"):
        # Load the mask from disk
        mask_path = os.path.join(workdir, "mask.bmp")
        rough_mask_path = os.path.join(workdir, "rough_mask.bmp")
        blended_mask_path = os.path.join(workdir, "blended_mask.bmp")

        if not all([os.path.exists(mask_path), os.path.exists(rough_mask_path), os.path.exists(blended_mask_path),]):
            raise ValueError(f"Mask files not found in {workdir}. Run create_mask to generate the mask.")
        with open(mask_path, "rb") as f:
            mask = Image.open(f).copy()
        with open(rough_mask_path, "rb") as f:
            rough_mask = Image.open(f).copy()
        with open(blended_mask_path, "rb") as f:
            blended_mask = Image.open(f).copy()

        return mask, rough_mask, blended_mask

    def inpaint_mask_with_prompt(self, im_path: str, mask: Image, p2: str, seed: int = TORCH_SEED):
        """
            This method inpaints the image using the mask generated by the DiffEdit algorithm. The mask is loaded from
            disk.

            im_path (str): The path to the image to edit.
            p2 (str): The prompt to add.
            workdir (str): The directory to load the masks from. Optional, default is "./".
            seed (int): The seed to use for reproducibility.
        """
        if not os.path.exists(im_path):
            raise ValueError(f"Image path {im_path} does not exist. Check the provided path")

        im = Image.open(im_path).resize((IMG_RESIZE_DIM, IMG_RESIZE_DIM)).copy()

        logging.info(f"Inpainting the image using the mask.")
        inpainted_image = self.inpainter.inpaint_mask(im, mask, p2, seed)

        return inpainted_image

    def demo_diffedit(self, im_path: str, p1: str, p2: str, n: int = 10, seed: int = TORCH_SEED):
        """
        This method performs the DiffEdit algorithm on the specified image.
        im_path (str): The path to the image to edit.
        p1 (str): The prompt to remove.
        p2 (str): The prompt to add.
        n (int): The number of iterations to perform to get the mask. Each iteration is a diffusion process.
        seed (int): The seed to use for reproducibility.

        return (list): A list containing the original image, the original image with mask and the resulting inpainted
         image.
        """
        if not os.path.exists(im_path):
            raise ValueError(f"Image path {im_path} does not exist. Check the provided path")
        out = []

        # getting image latent to put them in the output
        im = Image.open(im_path).resize((IMG_RESIZE_DIM, IMG_RESIZE_DIM)).copy()
        out.append(im)

        logging.info(f"Obtaining the mask by running the diffusion process {n} times.")
        mask, rough_mask, blended_mask = self.create_mask(im_path, p1, p2, n, seed=seed)
        self.save_mask(mask, rough_mask, blended_mask, im_path)

        out.append(blended_mask)  # blended mask is the visualization of the mask on the image with some transparency

        # load mask from disk
        mask, rough_mask, blended_mask = self.load_mask(workdir="./")
        logging.info(f"Inpainting the image using the mask.")
        inpainted_image = self.inpaint_mask_with_prompt(im_path, mask, p2, seed=seed)
        self.save_inpainted_image(inpainted_image)
        out.append(inpainted_image)
        return out

    def _get_embedding_for_prompt(self, prompt):
        """
        This method gets the embedding for a prompt using the text encoder.

        prompt (str): The prompt to get the embedding for.

        return (torch.Tensor): The embedding for the prompt.
        """
        max_length = self.tokenizer.model_max_length
        tokens = self.tokenizer([prompt], padding="max_length", max_length=max_length, truncation=True,
                                return_tensors="pt")
        with torch.no_grad():  # we are using for inference, no gradients needed
            return self.text_encoder(tokens.input_ids.to(self.torch_device))[0]
