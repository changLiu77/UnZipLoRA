import os
from typing import Optional, Dict, Union
from huggingface_hub import hf_hub_download
import copy 

import torch
from safetensors import safe_open
from diffusers.loaders.lora import LORA_WEIGHT_NAME_SAFE
import itertools

from inverse_ziplora_comb_separate.inverse_ziplora_separate import InverseZipLoRALinearLayer, InverseZipLoRALinearLayerInfer
from inverse_ziplora_comb_separate.pipeline_stable_diffusion_xl_seperate import StableDiffusionXLSeperatePipeline
from inverse_ziplora_comb_separate.unet_2d_seperate_condition import UNet2DConditionSeparateModel

from diffusers import (
     DDPMScheduler, 
     AutoencoderKL,
    )
from transformers import (
    AutoTokenizer, 
    CLIPTextModel, 
    CLIPTextModelWithProjection,
    )
from record_utils.cone import cone_matrix, cone_column_sparsity, draw_concatenated_heatmap

SDXL_group_list = [
    "up_blocks.0.attentions.0.",
    "up_blocks.0.attentions.1.",
    "up_blocks.0.attentions.2.",
    "up_blocks.1.attentions.0.",
    "up_blocks.1.attentions.1.",
    "up_blocks.1.attentions.2.",
    "down_blocks.1.attentions.0.",
    "down_blocks.1.attentions.1.",
    "down_blocks.2.attentions.0.",
    "down_blocks.2.attentions.1.",
    "mid_block.attentions.0.",
]

SDXL_up_layer_dictionary = {
    1: "up_blocks.0.attentions.1.",
    2: "up_blocks.0.attentions.1.",
    3: "up_blocks.0.attentions.1.",
    4: "up_blocks.0.attentions.1.",
    5: "up_blocks.0.attentions.1.",
    6: "up_blocks.0.attentions.1.",
    7: "up_blocks.0.attentions.1.",
    8: "up_blocks.0.attentions.1.",
    9: "up_blocks.0.attentions.1.",
    10: "up_blocks.0.attentions.1.",
    11: "up_blocks.1.attentions.0.",
    12: "up_blocks.1.attentions.0.",
    13: "up_blocks.1.attentions.1.",
    14: "up_blocks.1.attentions.1.",
    15: "up_blocks.1.attentions.2.",
    16: "up_blocks.1.attentions.2.",
    17: "up_blocks.0.attentions.2.",
    18: "up_blocks.0.attentions.2.",
    19: "up_blocks.0.attentions.2.",
    20: "up_blocks.0.attentions.2.",
    21: "up_blocks.0.attentions.2.",
    22: "up_blocks.0.attentions.2.",
    23: "up_blocks.0.attentions.2.",
    24: "up_blocks.0.attentions.2.",
    25: "up_blocks.0.attentions.2.",
    26: "up_blocks.0.attentions.2.",
    27: "up_blocks.0.attentions.0.",
    28: "up_blocks.0.attentions.0.",
    29: "up_blocks.0.attentions.0.",
    30: "up_blocks.0.attentions.0.",
    31: "up_blocks.0.attentions.0.",
    32: "up_blocks.0.attentions.0.",
    33: "up_blocks.0.attentions.0.",
    34: "up_blocks.0.attentions.0.",
    35: "up_blocks.0.attentions.0.",
    36: "up_blocks.0.attentions.0.",
}

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

SDXL_content_layer_mask = {
    "up_blocks.0.attentions.0.": False,
    "up_blocks.0.attentions.1.": False,
    "up_blocks.0.attentions.2.": False,
    "up_blocks.1.attentions.0.": False,
    "up_blocks.1.attentions.1.": False,
    "up_blocks.1.attentions.2.": False,
    "down_blocks.2.attentions.0.": False,
    "down_blocks.2.attentions.1.": False,
    "down_blocks.1.attentions.1.": False,
    "down_blocks.1.attentions.0.": False,
    "mid_block.attentions.0.": False,
}

SDXL_style_layer_mask = {
    "up_blocks.0.attentions.0.": False,
    "up_blocks.0.attentions.1.": False,
    "up_blocks.0.attentions.2.": False,
    "up_blocks.1.attentions.0.": False,
    "up_blocks.1.attentions.1.": False,
    "up_blocks.1.attentions.2.": False,
}

global_seeds = [0, 111, 1111, 1000, 1234, 123, 101, 42, 24, 26, 53]

def get_lora_weights(
    lora_name_or_path: str, subfolder: Optional[str] = None, **kwargs
) -> Dict[str, torch.Tensor]:
    """
    get lora weights from saved path 
    
    Args:
        lora_name_or_path (str): huggingface repo id or folder path of lora weights
        subfolder (Optional[str], optional): sub folder. Defaults to None.
    """
    if os.path.exists(lora_name_or_path):
        if subfolder is not None:
            lora_name_or_path = os.path.join(lora_name_or_path, subfolder)
        if os.path.isdir(lora_name_or_path):
            lora_name_or_path = os.path.join(lora_name_or_path, LORA_WEIGHT_NAME_SAFE)
    else:
        lora_name_or_path = hf_hub_download(
            repo_id=lora_name_or_path,
            filename=LORA_WEIGHT_NAME_SAFE,
            subfolder=subfolder,
            **kwargs,
        )
    assert lora_name_or_path.endswith(
        ".safetensors"
    ), "Currently only safetensors is supported"
    tensors = {}
    with safe_open(lora_name_or_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def inverse_ziplora_set_forward_type(unet: UNet2DConditionSeparateModel, type: str = "both"):
    """
    set forward type for the network: using either content part style part or both
    """
    assert type in ["both", "content", "style"]

    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            if lora_layer is not None:
                assert hasattr(lora_layer, "set_forward"), lora_layer
                lora_layer.set_forward(type)
    return unet

# * Functions about insert blocked mask
def generate_mask_in_unet(mask_dictionary={}):
    """
    based on the masked dictionary, generate the blocked parts for UNet
    """
    masked_layers = {}
    for key, value in mask_dictionary.items():
        block_name = key 
        for pattern in value:
            element = pattern.split("_")
            if element[0] == "N":
                block_nums = ['']
            elif element[0] == "A":
                if key == "up_blocks.":
                    block_nums = ['0', '1']
                elif key == "down_blocks.":
                    block_nums = ['1', '2']
            else:
                block_nums = element[0].split(',')
                
            if element[1] == "A":
                if key == "up_blocks.":
                    group_nums = ['0', '1', '2']
                elif key == "down_blocks.":
                    group_nums = ['1', '0']
            else:
                group_nums = element[1].split(',')
            
            if element[2] == "A":
                attention_types = ["attn1", "attn2"]
            else:
                attention_types = [f"attn{i}" for i in element[2].split(',')]
                
            if element[3] == "A":
                module_types = ["to_k", "to_v", "to_q", "to_out.0"]
            else:
                module_types = [f"to_{i}" for i in element[3].split(',')]
            
            key_combinations = itertools.product(
                block_nums, group_nums
            )
            value_combinations = [f"{attention_type}.{module_type}" \
                for attention_type, module_type in itertools.product(
                attention_types, module_types)]
            
            for block_num, group_num in key_combinations:
                masked_layers_key = f"{block_name}{block_num}.attentions.{group_num}."
                if key not in masked_layers.keys():
                    masked_layers[masked_layers_key] = copy.deepcopy(value_combinations)
                else:
                    masked_layers[masked_layers_key] += value_combinations
    return masked_layers

def insert_layer_dic_mask(unet, masked_layers, key, set_value=False, insert_toggle=False, toggle_key="content"):
    """
    Given the masked layers generated by "generate_mask_in_unet" 
    set the hard mask for the blocks for block separation
    """
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Module) and 'attn' in name:
                                                                                 
                before_key = name.split("transformer")[0]
                if before_key in masked_layers.keys():
                    attention_name = "attn" + name.split("attn")[-1]
                    if attention_name in masked_layers[before_key]:
                        lora_layer = getattr(module, "lora_layer")
                        if lora_layer is not None:
                            assert hasattr(lora_layer, "set_layer_mask"), lora_layer
                            lora_layer.set_layer_mask(key)
                            if set_value:
                                lora_layer.set_parameter_zero(key)
                    elif insert_toggle:
                        lora_layer = getattr(module, "lora_layer")
                        if lora_layer is not None:
                            assert hasattr(lora_layer, "set_layer_mask"), lora_layer
                            lora_layer.set_layer_mask(toggle_key)
                            if set_value:
                                lora_layer.set_parameter_zero(toggle_key)
                elif insert_toggle:
                    lora_layer = getattr(module, "lora_layer")
                    if lora_layer is not None:
                        assert hasattr(lora_layer, "set_layer_mask"), lora_layer
                        lora_layer.set_layer_mask(toggle_key)
                        if set_value:
                            lora_layer.set_parameter_zero(toggle_key)
    return unet
def insert_mask(unet: UNet2DConditionSeparateModel, key, mask_dictionary=None, set_value=False, insert_toggle=False, toggle_key="content"):
    """
    set the hard mask for the blocks for block separation
    """
    masked_layers = generate_mask_in_unet(mask_dictionary)
    unet = insert_layer_dic_mask(unet, masked_layers, key, set_value, insert_toggle, toggle_key)
    return unet


def inverse_ziplora_compute_weight_similarity(unet):
    '''
    Compute the cosine similarity between two merge matrix
    '''
    similarities = []
    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            if lora_layer is not None:
                assert hasattr(lora_layer, "compute_mergers_similarity"), lora_layer
                attn_module = name.split(".")[-1]
                # if attn_module == "to_v" or attn_module == "to_k":
                similarities.append(lora_layer.compute_mergers_similarity())
                # print(similarities)
    # similarity = torch.stack(similarities).sum(dim=0)
    similarity = torch.stack(similarities).mean(dim=0)
    return similarity.squeeze()

def unet_inverse_ziplora_state_dict(
    unet: UNet2DConditionSeparateModel, key: str, quick_release: bool = False
) -> Dict[str, torch.Tensor]:
    r"""
    Returns:
        A state dict containing the state with the input key
    Usage:
        For save the state dict of input
    """
    lora_state_dict = {}
    mask_state_dict = {}
    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            if lora_layer is not None:
                assert hasattr(lora_layer, "get_inverselora_weight"), lora_layer
                weight_down, weight_up = lora_layer.get_inverselora_weight(key)
                lora_state_dict[f"unet.{name}.lora.up.weight"] = weight_up.contiguous()
                lora_state_dict[f"unet.{name}.lora.down.weight"] = weight_down.contiguous()
                merge_matrix = lora_layer.get_merger_mask(key)
                mask_state_dict[f"unet.{name}.lora.merge_{key}"] = merge_matrix.contiguous()
                if quick_release:
                    lora_layer.cpu()
    return lora_state_dict, mask_state_dict


def initialize_inverse_ziplora_layer_for_inference(state_dict_content_down, state_dict_content_up, \
    state_dict_style_down, state_dict_style_up, \
    state_dict_merger_content, state_dict_merger_style, \
    part, **model_kwargs):
    '''
    Insert inverse ziplora to inference
    '''
    inverse_ziplora_layer = InverseZipLoRALinearLayerInfer(**model_kwargs)
    if state_dict_merger_content is not None and state_dict_merger_style is not None:
        inverse_ziplora_layer.load_state_dict(
            {
                "lora_matrix_dic.content_down.weight": state_dict_content_down[part],
                "lora_matrix_dic.content_up.weight": state_dict_content_up[part],
                "lora_matrix_dic.style_down.weight": state_dict_style_down[part],
                "lora_matrix_dic.style_up.weight": state_dict_style_up[part],
                "merge_content": state_dict_merger_content[part],
                "merge_style": state_dict_merger_style[part]
            },
            strict=False,
        )
    else:
        inverse_ziplora_layer.load_state_dict(
            {
                "lora_matrix_dic.content_down.weight": state_dict_content_down[part],
                "lora_matrix_dic.content_up.weight": state_dict_content_up[part],
                "lora_matrix_dic.style_down.weight": state_dict_style_down[part],
                "lora_matrix_dic.style_up.weight": state_dict_style_up[part],
            },
            strict=False,
        )
    return inverse_ziplora_layer

def use_lora_weights_for_inference(
    tensors: Dict[str, torch.Tensor], key: str, prefix: str = "unet.unet."
) -> Dict[str, torch.Tensor]:
    """
    Initialize the saved model value for down and up LoRA matrix
    Args:
        tensors (torch.Tensor): state dict of lora weights
        key (str): target attn layer's key
        prefix (str, optional): prefix for state dict.Defaults to "unet.unet.".
    """
    target_key = prefix + key
    out_down = {}
    out_up = {}
    # print(tensors.keys())
    for part in ["to_q", "to_k", "to_v", "to_out.0"]:
        down_key = target_key + f".{part}.lora.down.weight"
        up_key = target_key + f".{part}.lora.up.weight"
        out_down[part] = tensors[down_key]
        out_up[part] = tensors[up_key]
    return out_down, out_up

def use_lora_mergers_for_inference(
    tensors_content: Dict[str, torch.Tensor], tensors_style: Dict[str, torch.Tensor], key: str, prefix: str = "unet.unet."
) -> Dict[str, torch.Tensor]:
    """
    Initialize the saved model value for content and style soft mask merger
    Args:
        tensors (torch.Tensor): state dict of lora weights
        key (str): target attn layer's key
        prefix (str, optional): prefix for state dict.Defaults to "unet.unet.".
    """
    target_key = prefix + key
    merger_content  = {}
    merger_style  = {}
    # print(tensors.keys())
    for part in ["to_q", "to_k", "to_v", "to_out.0"]:
        merge_content = target_key + f".{part}.lora.merge_content"
        merge_style = target_key + f".{part}.lora.merge_style"
        merger_content[part] = tensors_content[merge_content]
        merger_style[part] = tensors_style[merge_style]
    return merger_content, merger_style
def insert_inverse_ziplora_to_unet(
    unet: UNet2DConditionSeparateModel, content_lora_path: str, style_lora_path: str, weight_content_path: str = None, weight_style_path: str = None, \
        rank: int = 64, device: Optional[Union[torch.device, str]] = None, **kwargs
):
    """
    Initialize the saved model value for content and style soft mask merger
    Args:
        tensors (torch.Tensor): state dict of lora weights
        key (str): target attn layer's key
        prefix (str, optional): prefix for state dict.Defaults to "unet.unet.".
    """
    tensors_content = get_lora_weights(content_lora_path, **kwargs)
    tensors_style = get_lora_weights(style_lora_path, **kwargs)
    if weight_content_path is not None:
        weight_content_state_dict = torch.load(weight_content_path)
    if weight_style_path is not None:
        weight_style_state_dict = torch.load(weight_style_path)
    # weight_state_dict = {key: value.to(device) for key, value in weight_state_dict.items()}
    for attn_processor_name, attn_processor in unet.attn_processors.items():
        # Parse the attention module.
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)
        # Get prepared for ziplora
        attn_name = ".".join(attn_processor_name.split(".")[:-1])
        state_dict_content_down, state_dict_content_up = use_lora_weights_for_inference(tensors_content, key=attn_name)
        state_dict_style_down, state_dict_style_up = use_lora_weights_for_inference(tensors_style, key=attn_name)
        if weight_content_path is not None and weight_style_path is not None:
            state_dict_merge_content, state_dict_merge_style = use_lora_mergers_for_inference(weight_content_state_dict, weight_style_state_dict, key=attn_name, prefix="unet.")
        else:
            state_dict_merge_style = None
            state_dict_merge_content = None
        # Set the `lora_layer` attribute of the attention-related matrices.
        kwargs = {
            "state_dict_content_down": state_dict_content_down, 
            "state_dict_content_up": state_dict_content_up,
            "state_dict_style_down": state_dict_style_down,
            "state_dict_style_up": state_dict_style_up,
            "state_dict_merger_content": state_dict_merge_content,
            "state_dict_merger_style": state_dict_merge_style 
        }

        attn_module.to_q.set_lora_layer(
            initialize_inverse_ziplora_layer_for_inference(
                in_features=attn_module.to_q.in_features,
                out_features=attn_module.to_q.out_features,
                part="to_q",
                lora_matrix_key=["content", "style"],
                rank=rank,
                device=device,
                **kwargs,
            )
        )
        attn_module.to_k.set_lora_layer(
            initialize_inverse_ziplora_layer_for_inference(
                in_features=attn_module.to_k.in_features,
                out_features=attn_module.to_k.out_features,
                part="to_k",
                lora_matrix_key=["content", "style"],
                device=device,
                rank=rank,
                **kwargs,
            )
        )
        attn_module.to_v.set_lora_layer(
            initialize_inverse_ziplora_layer_for_inference(
                in_features=attn_module.to_v.in_features,
                out_features=attn_module.to_v.out_features,
                part="to_v",
                device=device,
                lora_matrix_key=["content", "style"],
                rank=rank,
                **kwargs,
            )
        )
        attn_module.to_out[0].set_lora_layer(
            initialize_inverse_ziplora_layer_for_inference(
                in_features=attn_module.to_out[0].in_features,
                out_features=attn_module.to_out[0].out_features,
                part="to_out.0",
                device=device,
                lora_matrix_key=["content", "style"],
                rank=rank,
                **kwargs,
            )
        )
    return unet

def load_pipeline_from_sdxl(MODEL_ID,
    tokenizer_one: AutoTokenizer=None,
    tokenizer_two: AutoTokenizer=None,
    vae: AutoencoderKL=None,
    unet: UNet2DConditionSeparateModel=None,
    noise_scheduler: DDPMScheduler=None,
    text_encoder_one: CLIPTextModel=None, 
    text_encoder_two: CLIPTextModelWithProjection=None, 
    ):
    """
    log UnZipLoRA sdxl pipeline, enabling use
    """
    if tokenizer_one is None:
        tokenizer_one = AutoTokenizer.from_pretrained(
        MODEL_ID,
        subfolder="tokenizer",
        revision=None
        )
    if tokenizer_two is None:
        tokenizer_two = AutoTokenizer.from_pretrained(
            MODEL_ID,
            subfolder="tokenizer_2",
            revision=None
        )
    if vae is None:
        vae = AutoencoderKL.from_pretrained(
                MODEL_ID,
                subfolder="vae",
                revision=None
            )
    if unet is None:
        unet = UNet2DConditionSeparateModel.from_pretrained(
            MODEL_ID, subfolder="unet", revision=None
        )
    if noise_scheduler is None:
        noise_scheduler = DDPMScheduler.from_pretrained(
            MODEL_ID, subfolder="scheduler"
        )
    if text_encoder_one is None:
        text_encoder_one = CLIPTextModel.from_pretrained(
            MODEL_ID,
            subfolder="text_encoder",
            revision=None,
        )
    if text_encoder_two is None:
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            MODEL_ID,
            subfolder="text_encoder_2",
            revision=None,
        )
    pipeline = StableDiffusionXLSeperatePipeline(
        vae = vae,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        unet=unet,
        scheduler=noise_scheduler,
    )
    return pipeline

# * Functions that help record statistics / record in wandb

'''
Get the current value of matrix
'''
def lora_original_value(unet: UNet2DConditionSeparateModel, key):
    original_mat = {}
    with torch.no_grad():
        for name, module in unet.named_modules():
            if hasattr(module, "set_lora_layer"):
                lora_layer = getattr(module, "lora_layer")
                if lora_layer is not None:
                    assert hasattr(lora_layer, "get_inverselora_weight"), lora_layer
                    weight_down, weight_up = lora_layer.get_inverselora_weight(key)
                    original_mat[name] = copy.deepcopy(weight_down.T @ weight_up.T)
    return original_mat

'''
clamp merger
'''
def lora_merge_clamp(unet: UNet2DConditionSeparateModel, key):
    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            if lora_layer is not None:
                assert hasattr(lora_layer, "clamp_merger"), lora_layer
                lora_layer.clamp_merger(key)

'''
Log norm value block wise
'''
def lora_norm_log(unet: UNet2DConditionSeparateModel, key, quick_log=False, dim="L2", multiple=False):
    norm_dict = {}
    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            before_key = f"{key}_{name.split('transformer')[0]}_norm"
            if lora_layer is not None:
                assert hasattr(lora_layer, "get_inverse_lora_norm"), lora_layer
                if before_key in norm_dict.keys():
                    norm_dict[before_key].append(lora_layer.get_inverse_lora_norm(key, dim=dim, quick_log=quick_log, multiple=multiple))
                else:
                    norm_dict[before_key] = [lora_layer.get_inverse_lora_norm(key, dim=dim, quick_log=quick_log, multiple=multiple)]
    norm_dict = {key: torch.mean(torch.tensor(value)) for key,value in norm_dict.items()}
    return norm_dict
'''
Log merge value block wise
'''
def lora_merge_log(unet: UNet2DConditionSeparateModel, key, quick_log=False):
    norm_dict = {}
    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            before_key = f"{key}_{name.split('transformer')[0]}_merge"
            if lora_layer is not None:
                assert hasattr(lora_layer, f"merge_{key}"), lora_layer
                if before_key in norm_dict.keys():
                    norm_dict[before_key] += torch.mean(getattr(lora_layer, f"merge_{key}"))
                else:
                    norm_dict[before_key] = torch.mean(getattr(lora_layer, f"merge_{key}"))
    return norm_dict

def lora_merge_all_activate(unet: UNet2DConditionSeparateModel, value=True):
# * If value => True: will use filter       + not train merger gradient 
# * if value => False:will not use filter   + not train merger gradient
    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            if lora_layer is not None:
                assert hasattr(lora_layer, "set_layer_mask"), lora_layer
                lora_layer.set_layer_mask("content", value)
                lora_layer.set_layer_mask("style", value)
                lora_layer.set_merger_gradient("content")
                lora_layer.set_merger_gradient("style")

def lora_cone_spectrum_log(merge_gradient, merge_weight):
    '''
    input: current value, current gradient
    output: logged image
    '''
    logged_layers_key = [f"{layers}.attn2.to_k" for layers in logged_layers]
    logged_gradient = {key: value  for key, value in merge_gradient.items() if key in logged_layers_key}
    logged_weight = {key: value  for key, value in merge_weight.items() if key in logged_layers_key}
    cone = cone_matrix(logged_gradient, logged_weight)
    cone_heatmap_matrix = cone_column_sparsity(cone)
    buffer = draw_concatenated_heatmap(cone_heatmap_matrix)
    return buffer

def lora_merge_cone_select(unet: UNet2DConditionSeparateModel, mask_dictionary_style={}, mask_dictionary_content={}, logged=False, \
    column_ratio=0.05, avoid=True, accumulate=True):
    '''
    Initialize the saved model value for content and style soft mask merger
    Args:
        mask_dictionary_style/content: hard mask for block separation
        logged (optional): Whether saved the cone in wandb
        column_ratio: the ratio of selected/trained column features in each step 
        avoid: whether avoid training the same features
        accumulate: whether to add the computed cone in the attributes of each layer
    '''
    masked_layers_style = generate_mask_in_unet(mask_dictionary_style)
    masked_layers_content = generate_mask_in_unet(mask_dictionary_content)
    # the layers that will be logged to wandb
    logged_layers_key = [f"{layers}.attn2.to_k" for layers in logged_layers]
    # * blocked layers are blocks that are not inserted any all blocks filters
    # * masked_layers_style: blocks that are only with columns; other layers will insert all block filters
    # * block inserted with "all block" will multiply with masked .. / all columns will be used
    blocked_layers_key = masked_layers_content.keys() & masked_layers_style.keys()
    blocked_layers = {key: masked_layers_content[key] for key in blocked_layers_key}
    for key in blocked_layers_key:
        del masked_layers_content[key]
        del masked_layers_style[key]
    logged_cone_layer_content = {}
    logged_cone_layer_style = {}
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Module) and 'attn' in name:
            if hasattr(module, "set_lora_layer"):                                                                
                lora_layer = getattr(module, "lora_layer")
                if lora_layer is not None:
                    before_key = name.split("transformer")[0]
                    masked_key = None 
                    if before_key in blocked_layers.keys():
                        attention_name = "attn" + name.split("attn")[-1]
                        if attention_name in blocked_layers[before_key]:
                            masked_key = None
                    elif before_key in masked_layers_style.keys():
                        attention_name = "attn" + name.split("attn")[-1]
                        if attention_name in masked_layers_style[before_key]:
                            masked_key = "style"
                    elif before_key in masked_layers_content.keys():
                        attention_name = "attn" + name.split("attn")[-1]
                        if attention_name in masked_layers_content[before_key]:
                            masked_key = "content"
                    else:
                        masked_key = None
                    assert hasattr(lora_layer, "get_inverselora_cone"), lora_layer
                    # * store cone in self.attr
                    # * if it is accumulating: added to attr
                    # * if not accumulate or prepare to "determine mask": compute cone then store the "mask"
                    lora_layer.get_inverselora_cone("style", accumulate=accumulate)
                    lora_layer.get_inverselora_cone("content", accumulate=accumulate)
                    if accumulate is False: 
                        lora_layer.set_layer_mask("content", value=True)
                        lora_layer.set_layer_mask("style", value=True)
                        lora_layer.mask_updated_elements(masked_key, step_ratio=column_ratio, avoid=avoid)
                        lora_layer.set_merger_gradient("content", value=True)
                        lora_layer.set_merger_gradient("style", value=True)
                        if logged and name in logged_layers_key:
                            logged_cone_layer_content[name] = lora_layer.log_selected_mask("content")
                            logged_cone_layer_style[name] = lora_layer.log_selected_mask("style")
                        
    if logged:
        return unet, draw_concatenated_heatmap(logged_cone_layer_content), draw_concatenated_heatmap(logged_cone_layer_style)
    else:
        return unet, logged_cone_layer_content, logged_cone_layer_style
def lora_gradient_zeroout(unet: UNet2DConditionSeparateModel, finetune_mask):
    '''
    mask out the gradient during training to only allow training of certain columns
    '''
    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            if lora_layer is not None:
                lora_layer.set_gradient_mask(finetune_mask)
