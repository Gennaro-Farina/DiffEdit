"""
This script is used to edit an image using the DiffEdit model, given the image path and the prompts to remove and add.
The usage is the following:
    > python image_edit.py --image <image_path> --remove-prompt <prompt_to_remove> --add-prompt <prompt_to_add> \
                        --save-path <save_path>
An example usage is:
    > python image_edit.py --image-link "https://github.com/Gennaro-Farina/DiffEdit/blob/main/static/ai_gen_lion.jpeg" \
                        --remove-prompt "lion" --add-prompt "dog" --save-path "result.png"

In this example, an image of a lion is downloaded from the link provided and the lion is replaced with a dog.

The script will download the image from the link, remove what semantically represent the prompt "lion" and add
 the prompt "dog". It works by watching at the difference between the two prompts in the latent space and
 generating a mask that is used to inpaint the image. In order to achieve this, It runs the diffusion model for a
 certain number of steps (default is 10) and then inpaints the image using the mask generated and a final step of
 prompt conditioned diffusion.

 As all the diffusion processes, this also is influence by the seed. Given a mask, you'll get different results by
 changing the seed.
"""
import argparse
import torch
import os

from fastdownload import FastDownload
from diffusers import LMSDiscreteScheduler

from diff_edit.model.model import DiffEdit

os.environ['CURL_CA_BUNDLE'] = ''

def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="Diffusion Image Editing arguments")

    parser.add_argument(
        "--remove-prompt",
        required=False,
        type=str,
        help="What you want to remove from the image"
    )

    parser.add_argument(
        "--add-prompt",
        required=False,
        type=str,
        help="What you want to add to the image"
    )

    parser.add_argument(
        "--image",
        required=False,
        type=str,
        help="Path to the image to edit"
    )

    parser.add_argument(
        "--image-link",
        required=False,
        type=str,
        help="Link to the image to edit"
    )

    parser.add_argument(
        "--device",
        default="mps",
        type=str,
        choices=["cpu", "cuda", "mps"],
    )

    parser.add_argument(
        "--vae-model",
        required=False,
        default="stabilityai/sd-vae-ft-ema",
        type=str,
        help="Model name. E.g. stabilityai/sd-vae-ft-ema"
    )

    parser.add_argument(
        "--tokenizer",
        required=False,
        default="openai/clip-vit-large-patch14",
        type=str,
        help="Tokenizer to tokenize the text. E.g. openai/clip-vit-large-patch14"
    )

    parser.add_argument(
        "--text-encoder",
        required=False,
        default="openai/clip-vit-large-patch14",
        type=str,
        help="Text encoder to encode the text. E.g. openai/clip-vit-large-patch14"
    )

    parser.add_argument(
        "--unet",
        required=False,
        default="CompVis/stable-diffusion-v1-4",
        type=str,
        help="UNet model for generating the latents. E.g. CompVis/stable-diffusion-v1-4"
    )

    parser.add_argument(
        "--scheduler",
        required=False,
        default="LMSDiscreteScheduler",
        type=str,
        help="Noise scheduler. E.g. LMSDiscreteScheduler"
    )
    parser.add_argument(
        "--scheduler-start",
        required=False,
        default=0.0001,
        type=float,
        help="Scheduler start value"
    )
    parser.add_argument(
        "--scheduler-end",
        required=False,
        default=0.02,
        type=float,
        help="Scheduler end value"
    )

    parser.add_argument(
        "--num-train-timesteps",
        required=False,
        default=1000,
        type=int,
        help="Number of training timesteps"
    )
    parser.add_argument(
        "--beta-schedule",
        required=False,
        default="scaled_linear",
        type=str,
        help="Beta schedule"
    )

    parser.add_argument(
        "--inpainting",
        required=False,
        default="runwayml/stable-diffusion-inpainting",
        type=str,
        help="Inpainting model. E.g. runwayml/stable-diffusion-inpainting"
    )

    parser.add_argument(
        "--seed",
        required=False,
        default=42,
        type=int,
        help="Random seed"
    )

    parser.add_argument(
        "--num-samples",
        required=False,
        default=10,
        type=int,
        help="Number of diffusion steps to generate the mask"
    )

    parser.add_argument(
        "--save-path",
        required=False,
        default="./result.png",
        type=str,
        help="Path to save the result. Default is <script_folder>/result.png"
    )

    # Parse the arguments
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    print(f"> Setting up the DiffEdit components:\n{args}")

    scheduler = None
    if args.scheduler == "LMSDiscreteScheduler":
        scheduler = LMSDiscreteScheduler(beta_start=args.scheduler_start,
                                         beta_end=args.scheduler_end,
                                         beta_schedule=args.beta_schedule,
                                         num_train_timesteps=args.num_train_timesteps)
    else:
        raise ValueError(f"Unknown scheduler {args.scheduler}")

    diff_edit = DiffEdit()
    diff_edit.compose_model(args.vae_model,
                            args.tokenizer,
                            args.text_encoder,
                            args.unet,
                            args.inpainting,
                            scheduler,
                            torch_dev=args.device)

    # Validate the arguments
    assert args.remove_prompt and args.add_prompt, "Both remove and add prompts must be provided"
    assert args.image or args.image_link, "Either image or image_link must be provided"
    assert not (args.image and args.image_link), "Either image or image_link must be provided, not both"
    assert args.device in ["cpu", "cuda", "mps"], "Invalid device"
    assert args.scheduler in ["LMSDiscreteScheduler"], "Invalid scheduler"
    assert args.beta_schedule in ["scaled_linear"], "Invalid beta schedule"
    assert args.scheduler_start < args.scheduler_end, "Invalid beta values"

    print(f"> Setting random seed {args.seed}")
    torch.manual_seed(args.seed)

    # Download or use the provided image
    if args.image_link:
        im_path = FastDownload().download(args.image_link)
        if not os.path.exists(im_path):
            raise ValueError(f"Image path {im_path} does not exist. Check the link provided")
    elif args.image:
        im_path = args.image
        # verify the image to be a valid path
        if not os.path.exists(im_path):
            raise ValueError(f"Image path {im_path} does not exist")
    else:
        raise ValueError(f"Either image or image_link must be provided")


    print(f"> Editing image {im_path}")
    if args.remove_prompt:
        print(f"> Removing prompt {args.remove_prompt}")
        out = diff_edit.demo_diffedit(im_path, args.remove_prompt, args.add_prompt, seed=42, n=args.num_samples)
        if len(out) > 1:
            print('Saving result to', args.save_path)
            out[1].save(os.path.join(os.path.dirname(args.save_path), 'mask.png'))
            out[-1].save(args.save_path)  # inpainted image
        print(f"> Result saved to result.png")
