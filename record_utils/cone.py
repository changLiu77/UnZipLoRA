import argparse
from io import BytesIO
import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns
import torch
from safetensors.torch import load_file

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--weight_dir",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--gradient_dir",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--evaluate",
        choices=["sparsity", "avg"],  # Restrict the argument to these two values
        default="sparsity",
        help="The used evaluation"
    )
    parser.add_argument(
        "--transformer_ele",
        choices=["to_k", "to_q", "to_v"],  # Restrict the argument to these two values
        default="to_k",
        help="The used evaluation"
    )
    parser.add_argument(
        "--transformer_type",
        choices=["attn1", "attn2"],  # Restrict the argument to these two values
        default="attn2",
        help="The used evaluation"
    )
    parser.add_argument(
        "--save_path",
        default=None, 
        help="The used evaluation"
    )
    args = parser.parse_args()
    return args
def merge_weight_gradient(weight_path, gradient_path):
    gradient_state_dict = torch.load(gradient_path)
    weight_state_dict = load_file(weight_path)
    weight_state_dict = {key: value.to(device) for key, value in weight_state_dict.items()}
    transformer_key = set([key.split("lora")[0] for key in gradient_state_dict.keys()])
    merge_weight = {}
    merge_gradient = {}
    for key in transformer_key:
        gradient_up_key = f"{key}lora.up.weight"
        gradient_down_key = f"{key}lora.down.weight"
        weight_up_key = f"unet.{key}lora.up.weight"
        weight_down_key = f"unet.{key}lora.down.weight"
        merge_weight[key] = weight_state_dict[weight_down_key].T @ weight_state_dict[weight_up_key].T
        merge_gradient[key] = gradient_state_dict[gradient_down_key].T @ weight_state_dict[weight_up_key].T +\
                        weight_state_dict[weight_down_key].T @ gradient_state_dict[gradient_up_key].T
    return merge_gradient, merge_weight

def cone_matrix(merge_gradient, merge_weight):
    cone = {key: weight * gradient for (key, weight), (key, gradient) in zip(merge_weight.items(), merge_gradient.items())}
    return cone 

def cone_column_avg(cone):
    cone_avg = {key: value.mean(dim=0) for (key, value) in cone.items()}
    return cone_avg 

def cone_column_sparsity(cone, thresh=1e-8):
    cone_sparsity = {key: torch.sum(torch.abs(value) > thresh, dim=0).float() / value.shape[0] for (key, value) in cone.items()}
    return cone_sparsity

def draw_concatenated_heatmap(data_dic, save_path=None):
    num_heatmaps = len(list(data_dic.keys()))
    if num_heatmaps == 0:
        print(f"no heatmaps to record!!")
        return None  
    fig, axes = plt.subplots(1, num_heatmaps, figsize=(num_heatmaps * 15, 15))

    if num_heatmaps == 1:
        axes = [axes]
    for i, (key, value) in enumerate(data_dic.items()):
        # print(value.shape)
        sns.heatmap(np.tile(value.squeeze().detach().cpu(), (10, 1)), ax=axes[i], cmap='viridis', cbar=True)
        axes[i].set_title(key, fontsize=32)
        axes[i].axis('off')
    plt.tight_layout()
    if save_path is None:
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        # Close the figure to free up memory
        plt.close(fig)
        buffer_np = np.array(Image.open(buffer))
        buffer.close()
        return buffer_np
    else:
        plt.savefig(save_path)

def main(args):
    merge_gradient, merge_weight = merge_weight_gradient(args.weight_dir, args.gradient_dir)
    
    logged_layers = [
        "down_blocks.1.attentions.0.transformer_blocks.0",
        "down_blocks.1.attentions.1.transformer_blocks.0",
        "down_blocks.2.attentions.0.transformer_blocks.5",
        "down_blocks.2.attentions.0.transformer_blocks.9",
        "down_blocks.2.attentions.1.transformer_blocks.2",
        "down_blocks.2.attentions.1.transformer_blocks.3",
        "up_blocks.0.attentions.0.transformer_blocks.8",
        "up_blocks.0.attentions.0.transformer_blocks.3",
        "up_blocks.0.attentions.1.transformer_blocks.1",
        "up_blocks.0.attentions.1.transformer_blocks.9",
        "up_blocks.0.attentions.2.transformer_blocks.0",
        "up_blocks.0.attentions.2.transformer_blocks.4",
        "up_blocks.1.attentions.0.transformer_blocks.0",
        "up_blocks.1.attentions.1.transformer_blocks.0",
        "up_blocks.1.attentions.2.transformer_blocks.0", 
    ]
    logged_layers_key = [f"{layers}.{args.transformer_type}.{args.transformer_ele}." for layers in logged_layers]
    logged_gradient = {key: value  for key, value in merge_gradient.items() if key in logged_layers_key}
    logged_weight = {key: value  for key, value in merge_weight.items() if key in logged_layers_key}
    cone = cone_matrix(logged_gradient, logged_weight)
    if args.evaluate == "sparsity":
       cone_heatmap_matrix = cone_column_sparsity(cone)
    if args.evaluate == "avg":
        cone_heatmap_matrix = cone_column_avg(cone) 
    save_path = os.path.join(args.save_path, args.transformer_type, args.transformer_ele)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    saved_path = os.path.join(save_path, f"{args.evaluate}_heatmap.jpg")
    draw_concatenated_heatmap(cone_heatmap_matrix, saved_path)
if __name__ == "__main__":
    args = parse_args()
    main(args)