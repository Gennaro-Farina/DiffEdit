# DiffEdit
___

[![pypi wheel](https://github.com/Gennaro-Farina/DiffEdit/actions/workflows/publish-wheel-pypi.yml/badge.svg)](https://github.com/Gennaro-Farina/DiffEdit/actions/workflows/publish-wheel-pypi.yml)
[![build and test](https://github.com/Gennaro-Farina/DiffEdit/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/Gennaro-Farina/DiffEdit/actions/workflows/build_and_test.yml)

An unofficial implementation of <a href="https://arxiv.org/abs/2210.11427"> DiffEdit</a> based on <a href="https://huggingface.co"> 🤗 Hugging Face </a>, <a href="https://github.com/johnrobinsn/diffusion_experiments/blob/main/DiffEdit.ipynb"> this repo</a> and PyTorch.
This methodology leverage the diffusion process to automatically extract a mask from an image given a prompt. The mask is then used to inpaint the image with the new content.
To get a clearer overview of the process, you can take a look at the <a href="https://github.com/Gennaro-Farina/diffusion-nbs/blob/master/DiffEdit.ipynb"> DiffEdit.ipynb</a> notebook.

## Results

<table>
<head>
<th> Prompt: <i>remove</i> ⟶ <i>add</i>)</th><th>Original image</th> <th>Mask</th> <th>Edited</th>
</head>
<body>
<tr>
<td>"lion" ⟶ "dog"</td>
<td><img src="static/ai_gen_lion.jpeg" width="256" height="256" alt="An AI generated lion"></td>
<td><img src="static/ai_gen_lion_mask.png" width="256" height="256" 
         alt="The mask of the image region containing the 'lion'"></td>
<td><img src="static/ai_gen_lion_result.png" width="256" height="256"
         alt="The edited image with the 'dog' instead of the 'lion'"></td>
</tr>
<tr>
<td>"house" ⟶ "3-floor hotel"</td>
<td><img src="static/ai_gen_house.jpeg" width="256" height="256" alt="An AI generated house"></td>
<td><img src="static/ai_gen_house_mask.png" width="256" height="256" 
         alt="The mask of the image region containing the 'house'"></td>
<td><img src="static/ai_gen_house_result.png" width="256" height="256"
         alt="The edited image with the '3-floor hotel' instead of the 'house'"></td>
</tr>
<tr>
<td>"an F1 race" ⟶ "a motogp race"</td>
<td><img src="static/ai_gen_f1.jpeg" width="256" height="256" alt="An AI generated image of an F1 competition"></td>
<td><img src="static/ai_gen_f1_mask.png" width="256" height="256" 
         alt="The mask of the image region containing the F1 cars"></td>
<td><img src="static/ai_gen_f1_result.png" width="256" height="256"
         alt="The edited image with the 'motogp' instead of the 'F1'"></td>
</tr>
</body>
</table>

All the previous masks was generated with: `num-samples = 10`

## Installation

You can install the package in different ways, depending on your needs.



<details>
  <summary> Optional step (recommended)</summary>
  
Create a virtual environment, to avoid conflicts with other packages. Here are some alternatives:

- with `venv`:
```bash
python -m venv venv
source venv/bin/activate
```

- with `poetry`:
```bash
poetry shell
```

- with `conda`:
```bash
conda create -n diff-edit python=3.10
conda activate diff-edit
```

</details>


Install the package from PyPi:
```bash
pip install diff-edit
```
<details>
    <summary> Alternative ways</summary>

Install the package from source:
```bash
poetry install
```

Install the package in editable mode, suggested for further development:
```bash
pip install -e .
```
</details>

## Usage

For a fast evaluation use the script <a href="https://github.com/Gennaro-Farina/DiffEdit/blob/main/src/diff_edit/examples/image_edit.py">image_edit.py</a>:

```bash
python image_edit.py --input_image <path_to_image> --output_image <path_to_output_image> --prompt <prompt>
```

An example of usage is the following (resulting in <a href="https://github.com/Gennaro-Farina/DiffEdit/blob/main/static/ai_gen_lion_result.png"> this</a> image):

```bash
python image_edit.py --remove-prompt "lion" --add-prompt "dog" --image-link "https://github.com/Gennaro-Farina/DiffEdit/blob/main/static/ai_gen_lion.jpeg" --num-samples 10
```

You can further customize the process by tuning the parameters of the script. Here is the full list of parameters that you can tune:

```bash
python image_edit.py --help
usage: image_edit.py [-h] [--remove-prompt REMOVE_PROMPT] [--add-prompt ADD_PROMPT] [--image IMAGE] [--image-link IMAGE_LINK] [--device {cpu,cuda,mps}]
                     [--vae-model VAE_MODEL] [--tokenizer TOKENIZER] [--text-encoder TEXT_ENCODER] [--unet UNET] [--scheduler SCHEDULER]
                     [--scheduler-start SCHEDULER_START] [--scheduler-end SCHEDULER_END] [--num-train-timesteps NUM_TRAIN_TIMESTEPS] [--beta-schedule BETA_SCHEDULE]
                     [--inpainting INPAINTING] [--seed SEED] [--n N] [--save-path SAVE_PATH]

Diffusion Image Editing arguments

options:
  -h, --help            show this help message and exit
  --remove-prompt REMOVE_PROMPT
                        What you want to remove from the image
  --add-prompt ADD_PROMPT
                        What you want to add to the image
  --image IMAGE         Path to the image to edit
  --image-link IMAGE_LINK
                        Link to the image to edit
  --device {cpu,cuda,mps}
  --vae-model VAE_MODEL
                        Model name. E.g. stabilityai/sd-vae-ft-ema
  --tokenizer TOKENIZER
                        Tokenizer to tokenize the text. E.g. openai/clip-vit-large-patch14
  --text-encoder TEXT_ENCODER
                        Text encoder to encode the text. E.g. openai/clip-vit-large-patch14
  --unet UNET           UNet model for generating the latents. E.g. CompVis/stable-diffusion-v1-4
  --scheduler SCHEDULER
                        Noise scheduler. E.g. LMSDiscreteScheduler
  --scheduler-start SCHEDULER_START
                        Scheduler start value
  --scheduler-end SCHEDULER_END
                        Scheduler end value
  --num-train-timesteps NUM_TRAIN_TIMESTEPS
                        Number of training timesteps
  --beta-schedule BETA_SCHEDULE
                        Beta schedule
  --inpainting INPAINTING
                        Inpainting model. E.g. runwayml/stable-diffusion-inpainting
  --seed SEED           Random seed
  --num-samples N       Number of diffusion steps to generate the mask
  --save-path SAVE_PATH
                        Path to save the result. Default is <script_folder>/result.png
```

