import torch

from PIL import Image
from torchvision import transforms


class ImageProcessor:
    def __init__(self, vae, torch_device):
        self.vae = vae
        self.torch_device = torch_device

    def img2latent(self, im, vae_magic_number):
        """
        This method converts an image to a latent representation using the VAE.

        im (PIL.Image): The image to convert to a latent representation.
        vae_magic_number (float): The magic number to scale the latent representation by.

        return (torch.Tensor): The latent representation of the image.
        """
        # convert image to tensor
        im = transforms.ToTensor()(im).unsqueeze(0)
        with torch.no_grad():
            # encode the image into latent space through the VAE
            latent = self.vae.encode(im.to(self.torch_device) * 2 - 1)
        latent = latent.latent_dist.sample() * vae_magic_number
        return latent

    def latents2imgs(self, latents, vae_magic_number):
        """
        This method converts latents to images using the VAE.

        latents (torch.Tensor): The latents to convert to images.
        vae_magic_number (float): The magic number to scale the latents by.

        return (list): A list of images.
        """
        # getting the latents from VAE (decoder layers)
        latents = latents * 1 / vae_magic_number
        with torch.no_grad():
            imgs = self.vae.decode(latents).sample
        # let's convert images to PIL, so we can display them.
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")
        imgs = [Image.fromarray(im) for im in imgs]
        return imgs

    # Composite the mask over the provided image; for demonstration purposes
    def get_blended_mask(self, im, mask_gray):  # Both expected to be PIL images
        """
        This method blends the mask over the provided image.

        im (PIL.Image): The image to blend the mask over.
        mask_gray (PIL.Image): The mask to blend over the image.

        return (PIL.Image): The blended image.
        """
        mask_rgb = mask_gray.convert('RGB')
        return Image.blend(im, mask_rgb, 0.40)
