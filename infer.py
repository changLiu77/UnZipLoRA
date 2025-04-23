import argparse
import torch 
import os

from diffusers import (
    AutoencoderKL,
    StableDiffusionXLPipeline,
)

from inverse_ziplora_comb_separate.utils import *

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
# MODEL_ID="etri-vilab/koala-lightning-1b"
seeds = [0, 1000, 111, 1234]
device = "cuda" if torch.cuda.is_available() else "cpu"
weight_dtype = torch.float16

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--with_unziplora",
        action="store_true",
        help="Whether use different prompts to generate",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=4,
        help=("The number of generated figures of each seed."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help=("The directory for saved model"),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="example_output",
        help=("The directory for saved generated figures"),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help=("The prompt for validation"),
    )

    parser.add_argument(
        "--validation_prompt_content_forward",
        type=str,
        default=None,
        help=("The prompt for validation"),
    )
    parser.add_argument(
        "--validation_prompt_style_forward",
        type=str,
        default=None,
        help=("The prompt for validation"),
    )

    parser.add_argument(
        "--validation_prompt_content_recontext",
        type=str,
        default=None,
        help=("The content recontext prompt for validation"),
    )

    parser.add_argument(
        "--validation_prompt_style",
        type=str,
        default=None,
        help=("The style prompt for validation"),
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    
    
    args = parser.parse_args()
    
    return args 

def log_validation(pipeline, prompt, prompt_content="", prompt_style="", seed=0, num=4):
    generator = torch.Generator(device=device).manual_seed(seed)
    # Currently the context determination is a bit hand-wavy. We can improve it in the future if there's a better
    # way to condition it. Reference: https://github.com/huggingface/diffusers/pull/7126#issuecomment-1968523051
    if pipeline.__class__.__name__ == 'StableDiffusionXLSeperatePipeline':
        pipeline_args = {"prompt": prompt, 
                        "prompt_content": prompt_content, 
                        "prompt_style": prompt_style}
    else: 
        pipeline_args = {"prompt": prompt}
        
    images = [pipeline(**pipeline_args, generator=generator, num_inference_steps=50).images[0] for _ in range(num)]
    return images

def save_img(img_dir, images, img_num):
    for _, img in enumerate(images):
        image_path = os.path.join(img_dir, f"image_{img_num}.png")
        img_num += 1
        img.save(image_path, "PNG")
    return img_num

def generate_save_img(args, pipeline, prompt, prompt_catogory, prompt_content_forward=None, prompt_style_forward=None):
    for i in range(len(prompt)):
        img_num = 1
        prompt_dir = os.path.join(prompt_catogory, "_".join(prompt[i].split(" ")))
        if os.path.isdir(prompt_dir):
            continue
        os.makedirs(prompt_dir, exist_ok=True)
        for seed in seeds:
            print(prompt[i])
            if pipeline.__class__.__name__ == 'StableDiffusionXLSeperatePipeline':
                images = log_validation(
                    pipeline,
                    prompt[i],
                    prompt_content_forward[i],
                    prompt_style_forward[i],
                    seed = seed, 
                    num = args.num
                )
            else:
                images = log_validation(
                    pipeline,
                    prompt[i],
                    seed = seed, 
                    num = args.num
                )
            img_num = save_img(prompt_dir, images, img_num)
def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    with torch.no_grad():
        # * Generate combined images
        vae = AutoencoderKL.from_pretrained(
            MODEL_ID,
            subfolder="vae",
            revision=None
        )
        print("model dir: ", args.output_dir)
        
        if len(args.validation_prompt) != 0 and args.validation_prompt != ['']: 
            pipeline = load_pipeline_from_sdxl(
            MODEL_ID, vae = vae,
            )
            if args.with_unziplora:
                pipeline.unet = insert_unziplora_to_unet(pipeline.unet, 
                    f"{args.output_dir}_content", 
                    f"{args.output_dir}_style",
                    weight_content_path=f"{args.output_dir}_merger_content.pth",
                    weight_style_path=f"{args.output_dir}_merger_style.pth",
                    rank=args.rank)
            else:
                pipeline.unet = insert_unziplora_to_unet(pipeline.unet, 
                    f"{args.output_dir}_content", 
                    f"{args.output_dir}_style",
                    rank=args.rank)
            pipeline = pipeline.to(device, dtype=weight_dtype)
            prompt_catogory = os.path.join(args.save_dir, "combine_recontextual_outputs")
            os.makedirs(prompt_catogory, exist_ok=True)
            if args.with_unziplora:
                generate_save_img(args, pipeline, args.validation_prompt, prompt_catogory, \
                    args.validation_prompt_content_forward, args.validation_prompt_style_forward)
            else:
                generate_save_img(args, pipeline, args.validation_prompt, prompt_catogory)
            
            del pipeline
        
        if len(args.validation_prompt_content_recontext) != 0 and args.validation_prompt_content_recontext != ['']: 
            pipeline = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            )
            prompt_catogory = os.path.join(args.save_dir, "content_recontextual_outputs")
            os.makedirs(prompt_catogory, exist_ok=True)
            
            pipeline = pipeline.to(device, dtype=weight_dtype)
            pipeline.load_lora_weights(f"{args.output_dir}_content")
            # pipeline.load_lora_weights(f"{args.output_dir}")
            print(f"generate recontext prompt {args.validation_prompt_content_recontext}")
            generate_save_img(args, pipeline, args.validation_prompt_content_recontext, prompt_catogory)
            
            del pipeline        

        if len(args.validation_prompt_style) != 0 and args.validation_prompt_style != ['']: 
            prompt_catogory = os.path.join(args.save_dir, "style_recontextual_outputs")
            os.makedirs(prompt_catogory, exist_ok=True)
            
            pipeline = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            )
            pipeline = pipeline.to(device, dtype=weight_dtype)
            pipeline.load_lora_weights(f"{args.output_dir}_style")
            # pipeline.load_lora_weights(f"{args.output_dir}")
            print(f"generate recontext prompt {args.validation_prompt_style}")
            generate_save_img(args, pipeline, args.validation_prompt_style, prompt_catogory)
        
if __name__ == "__main__":
    args = parse_args()
    args.validation_prompt=args.validation_prompt.split(",")
    args.validation_prompt_style_forward=args.validation_prompt_style_forward.split(",")
    args.validation_prompt_content_forward=args.validation_prompt_content_forward.split(",")
    args.validation_prompt_content_recontext=args.validation_prompt_content_recontext.split(",")
    args.validation_prompt_style=args.validation_prompt_style.split(",")
    
    main(args)
