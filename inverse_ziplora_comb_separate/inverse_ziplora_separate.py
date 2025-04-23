from typing import Optional, Union, List
import numpy as np
import torch
from torch import nn
import copy

class UnZipLoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int, 
        out_features: int,
        rank: int = 64,
        lora_matrix_key: List[str] = None, 
        device: Optional[Union[torch.device, str]] = None,
        # dtype: Optional[torch.dtype] = torch.float32,
        dtype: Optional[torch.dtype] = None,
        **model_kwargs
    ):
        super().__init__()
    
        self.lora_matrix_dic = nn.ModuleDict()
        self.fixed_matrix = {}
        self.lora_matrix_dic_norm = {}
        # * If masked matrix => True: the column filter is used / not all columns are used
        # *                 => False: the filter is not used => all coluns are used
        self.masked_matrix = {}
        for key in lora_matrix_key:
            self.lora_matrix_dic[f"{key}_down"] = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
            self.lora_matrix_dic[f"{key}_up"] = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
            # setattr(self, f"merge_{key}", nn.Parameter(torch.ones((out_features,), device=device, dtype=dtype), requires_grad=True))
            nn.init.normal_(self.lora_matrix_dic[f"{key}_down"].weight, std=1 / rank)
            nn.init.normal_(self.lora_matrix_dic[f"{key}_up"].weight, std=1 / rank)
            self.lora_matrix_dic_norm[f"{key}_norm_down"] = torch.norm(self.lora_matrix_dic[f"{key}_down"].weight.detach(), dim=0, keepdim=True)
            self.lora_matrix_dic_norm[f"{key}_norm_up"] = torch.norm(self.lora_matrix_dic[f"{key}_up"].weight.detach(), dim=0, keepdim=True)
            # * Whether use column filter, initialized: do not use
            self.masked_matrix[key] = False
        # TODO: hard code for only one content and one style
        # * merge: the value for each columns
        # * mask: the column filter(bool)
        self.merge_content = nn.Parameter(torch.ones(out_features, device=device, dtype=dtype, requires_grad=True))
        self.merge_style = nn.Parameter(torch.ones(out_features, device=device, dtype=dtype, requires_grad=True))
        self.column_score_content = torch.ones(out_features)
        self.column_score_style = torch.ones(out_features)
        self.mask_content = torch.zeros(out_features, device=device, dtype=torch.bool)
        self.mask_style = torch.zeros(out_features, device=device, dtype=torch.bool)
        self.lora_matrix_key = lora_matrix_key
        self.out_features = out_features
        self.in_features = in_features
        self.rank = rank
        self.dtype = dtype
        self.forward_type = "both"
        self.device = device
    def set_cone_score(self, key):
        setattr(self, f"column_score_{key}", torch.zeros(self.out_features))
    def set_forward(self, type: str = "both"):
        assert type in ["both", "content", "style"]
        self.forward_type = type
        
    def compute_mergers_similarity(self):
        # * If not filtered, directly compute the cosine similarity
        if self.masked_matrix["content"] is False or self.masked_matrix["style"] is False:
            return (self.merge_content * self.merge_style).abs().mean().unsqueeze(0)
        # * If filtered, first combined with mask and then compute the coisne similarity
        else:    
            return ((self.merge_content * self.mask_content) * (self.merge_style * self.mask_style)).abs().mean().unsqueeze(0)
    
    def set_merger_gradient(self, key, value=False):
        merger_matrix = getattr(self, f"merge_{key}")
        merger_matrix.requires_grad = value
        setattr(self, f"merge_{key}", merger_matrix)
    # * Each merger should between (0, 1)
    def clamp_merger(self, key):
        merge_matrix = getattr(self, f"merge_{key}")
        merge_matrix.clamp_(0, 1)
        setattr(self, f"merge_{key}", merge_matrix)
    
    def get_merger_mask(self, key):
        merge_matrix = getattr(self, f"merge_{key}")
        return merge_matrix
            
    def set_layer_mask(self, key, value=True):
        self.masked_matrix[key] = value

    def get_unziplora_norm(self, key, dim="L2", quick_log=False, multiple=True):
        if multiple is True: 
            merge_matrix = getattr(self, f"merge_{key}")
            # merge_matrix = (torch.tanh(merge_matrix) + 1) / 2 #* merge_matrix
            merged_matrix = self.lora_matrix_dic[f"{key}_down"].weight.T @ \
                (self.lora_matrix_dic[f"{key}_up"].weight.data.T) * merge_matrix
        else: 
            merged_matrix = self.lora_matrix_dic[f"{key}_down"].weight.T @ \
                self.lora_matrix_dic[f"{key}_up"].weight.data.T 
        if self.masked_matrix[key] is True:
            merged_matrix *= getattr(self, f"mask_{key}")
        if dim == "L2":
            norm = torch.norm(merged_matrix, p="fro")
        elif dim == "L1":
            norm = torch.norm(merged_matrix, p=1)
        elif dim == "nuc":
            norm = torch.norm(merged_matrix, p='nuc')
        norm_return = torch.tensor(norm.item()).to(self.device) if quick_log else norm.unsqueeze(0)
        return norm_return 
    
    def get_unziplora_weight(self, key):
        # print(self.lora_matrix_dic.keys())
        # * Get weight without filter
        merge_matrix = getattr(self, f"merge_{key}")
        if self.masked_matrix[key] is False:
            # merge_matrix = (torch.tanh(merge_matrix) + 1) / 2#* merge_matrix  
            # return self.lora_matrix_dic[f"{key}_down"].weight.data, self.lora_matrix_dic[f"{key}_up"].weight.data
            return self.lora_matrix_dic[f"{key}_down"].weight.data, self.lora_matrix_dic[f"{key}_up"].weight.data * merge_matrix.unsqueeze(1)
        else:
            filter_matrix = getattr(self, f"mask_{key}")
            return self.lora_matrix_dic[f"{key}_down"].weight.data, self.lora_matrix_dic[f"{key}_up"].weight.data \
                 * filter_matrix.unsqueeze(1)
    
    def get_unziplora_cone(self, key, accumulate=True):
        '''
        Compute cone value for both style and content, store the value in self.column_score
        Will be used when all columns are used ==> The computed cone will help determine which columns \
        will be used in following training ==> the filter will not included in computation
        Theratically(if no bugs), every parameters except merger will have gradient
        '''
        merge_matrix = getattr(self, f"merge_{key}")
        merger_gradient = merge_matrix.grad
        merged_weight = self.lora_matrix_dic[f"{key}_down"].weight.data.T @ self.lora_matrix_dic[f"{key}_up"].weight.data.T
        if merger_gradient is None: 
            if self.lora_matrix_dic[f"{key}_down"].weight.grad is None:
                merged_gradient = self.lora_matrix_dic[f"{key}_down"].weight.data.T @ self.lora_matrix_dic[f"{key}_up"].weight.grad.T * merge_matrix        
            else:
                merged_gradient = self.lora_matrix_dic[f"{key}_down"].weight.grad.T @ self.lora_matrix_dic[f"{key}_up"].weight.data.T * merge_matrix +\
                                self.lora_matrix_dic[f"{key}_down"].weight.data.T @ self.lora_matrix_dic[f"{key}_up"].weight.grad.T * merge_matrix
        else:
            if self.lora_matrix_dic[f"{key}_down"].weight.grad is None:
                merged_gradient = self.lora_matrix_dic[f"{key}_down"].weight.data.T @ self.lora_matrix_dic[f"{key}_up"].weight.grad.T * merge_matrix + \
                                merged_weight * merger_gradient
            else:
                merged_gradient = self.lora_matrix_dic[f"{key}_down"].weight.grad.T @ self.lora_matrix_dic[f"{key}_up"].weight.data.T * merge_matrix +\
                                self.lora_matrix_dic[f"{key}_down"].weight.data.T @ self.lora_matrix_dic[f"{key}_up"].weight.grad.T * merge_matrix + \
                                merged_weight * merger_gradient
        cone = merged_weight * merged_gradient
        if accumulate: 
            setattr(self, f"column_score_{key}", getattr(self, f"column_score_{key}").to(self.device) + cone.to(self.device))
        else: 
            cone_sparsity = torch.sum(torch.abs(cone) > 1e-5, dim=0).float() / cone.shape[0]
            setattr(self, f"column_score_{key}", cone_sparsity)
    def set_gradient_mask(self, finetune_mask):
        '''
            set the gradient map for column features
            For up layers: if the filter is selected(if not been masked is True), trained
            if set finetune_mask as false: only overlapped part is trained
                                 as true : all are trained
            For merger: if the two are overlapped with each other, the merger will be trained
            Only called if only the overlapped part are trained
        '''
        merge_content = getattr(self, f"merge_content")
        merge_style = getattr(self, f"merge_style")

        merger_overlapped = self.mask_content & self.mask_style
        # print(merger_overlapped)
        if merge_content.grad is not None and merge_style.grad is not None:
            if not finetune_mask:
                merge_content.grad *= merger_overlapped 
                merge_style.grad *= merger_overlapped 
            else: 
                merge_content.grad *= self.mask_content
                merge_style.grad *= self.mask_style 
                
            setattr(self, "merge_content", merge_content)
            setattr(self, "merge_style", merge_style)
    # * use hard mask and creat filters from top k columns
    def mask_updated_elements(self, key=None, step_ratio=0.1, avoid=True):
        '''
            # Now we will set the mask == previous + current
            
            Args:
            key: compute the sparse masks for given keys 
                key = None: sparse masks for both content and style 
                key = content / style: the given keys will have sparse mask while features of 
                                       the other will train all 
            step_ratio: how many columns are seleted
            avoid: True: will choose both content and style but content is prioritized 
        
        '''
        
        selected_num = int(self.out_features * step_ratio)
        if key is None: 
            # if key is none, no blocks are masked, i.e: will generate columns mask for both content and style
            # style mask is the Complement of content
            top_content_values, _= torch.topk(self.column_score_content, selected_num)
            # _, top_indices_style = torch.topk(getattr(self, f"column_score_style"), top_k)
            if top_content_values.numel() > 0:
                threshold = top_content_values.min()
            else:
                threshold = float('inf')
            content_mask_current = self.column_score_content > threshold 
            self.mask_content = content_mask_current | self.mask_content
            masked_style = self.column_score_style.clone()
            if avoid:
                masked_style[self.mask_content] = float('-inf')
            top_style_values, _= torch.topk(masked_style, selected_num)
            if top_style_values.numel() > 0:
                threshold = top_style_values.min()
            else:
                threshold = float('inf')
            mask_style_current = masked_style > threshold 
            self.mask_style = mask_style_current | self.mask_style
        else:
            # * generate sparse mask for given key 
            top_values, _= torch.topk(getattr(self, f"column_score_{key}"), selected_num)
            if top_values.numel() > 0:
                threshold = top_values.min()
            else:
                threshold = float('inf')
            mask = getattr(self, f"column_score_{key}") > threshold 
            setattr(self, f"mask_{key}", mask | getattr(self, f"mask_{key}"))
            all_on_key = "content" if key == "style" else "style"
            setattr(self, f"mask_{all_on_key}", torch.ones(self.out_features, device=self.device, dtype=torch.bool))
    def log_selected_mask(self, key):
        return getattr(self, f"column_score_{key}") * getattr(self, f"mask_{key}")
    def forward(self, hidden_states_content: torch.Tensor, hidden_states_style: torch.Tensor=None) -> torch.Tensor:
        '''
        forward with content and style hidden states 
        the weight depend on soft mask self.merge and hard mask(bor block separation) self.mask
        if set forward type as both: content and style are used
        '''
        dtype = self.dtype
        if self.forward_type == "both":
            orig_dtype = hidden_states_content.dtype
                # print(hidden_states.shape)
            if hidden_states_style is None: 
                hidden_states_style = hidden_states_content
            merged_content_weight = self.lora_matrix_dic["content_down"].weight.T @ \
                                    self.lora_matrix_dic["content_up"].weight.T 
            masked_content_weight = merged_content_weight * self.merge_content
            if self.masked_matrix["content"] is True:
                masked_content_weight *= self.mask_content
            up_hidden_states_content = hidden_states_content.to(dtype) @ masked_content_weight
            
            merged_style_weight = self.lora_matrix_dic["style_down"].weight.T @ \
                                    self.lora_matrix_dic["style_up"].weight.T 
            masked_style_weight = merged_style_weight * self.merge_style
            if self.masked_matrix["style"] is True: 
                masked_style_weight *= self.mask_style
            up_hidden_states_style = hidden_states_style.to(dtype) @ masked_style_weight
            added_hidden_states = up_hidden_states_style.to(orig_dtype) + up_hidden_states_content.to(orig_dtype)
        if self.forward_type == "content":
            orig_dtype = hidden_states_content.dtype
            merged_content_weight = self.lora_matrix_dic["content_down"].weight.T @ \
                                    self.lora_matrix_dic["content_up"].weight.T 
            if self.masked_matrix["content"] is True:
                merged_content_weight = merged_content_weight * self.mask_content
            up_hidden_states_content = hidden_states_content.to(dtype) @ merged_content_weight
            added_hidden_states = up_hidden_states_content.to(orig_dtype)
        if self.forward_type == "style":
            orig_dtype = hidden_states_style.dtype
            merged_style_weight = self.lora_matrix_dic["style_down"].weight.T @ \
                                    self.lora_matrix_dic["style_up"].weight.T 
            if self.masked_matrix["style"] is True:
                merged_style_weight = merged_style_weight * self.mask_style
            up_hidden_states_style = hidden_states_style.to(dtype) @ merged_style_weight
            added_hidden_states = up_hidden_states_style.to(orig_dtype)
        return added_hidden_states
    
class UnZipLoRALinearLayerInfer(nn.Module):
    def __init__(
        self,
        in_features: int, 
        out_features: int, 
        rank: int = 64,
        lora_matrix_key: List[str] = None, 
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.lora_matrix_dic = nn.ModuleDict()
        self.masked_matrix = {}
        for key in lora_matrix_key:
            self.lora_matrix_dic[f"{key}_down"] = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
            self.lora_matrix_dic[f"{key}_up"] = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
            nn.init.normal_(self.lora_matrix_dic[f"{key}_down"].weight, std=1 / rank)
            nn.init.normal_(self.lora_matrix_dic[f"{key}_up"].weight, std=1 / rank)
            self.masked_matrix[key] = False
        self.lora_matrix_key = lora_matrix_key
        self.out_features = out_features
        self.forward_type = "both"
        self.dtype = dtype
        self.merge_content = nn.Parameter(torch.ones(out_features, device=device, dtype=dtype))
        self.merge_style = nn.Parameter(torch.ones(out_features, device=device, dtype=dtype))

    def set_layer_mask(self, key, value=True):
        self.masked_matrix[key] = value
    
    def set_forward(self, type: str = "both"):
        assert type in ["both", "content", "style"]
        self.forward_type = type

    def forward(self, hidden_states_content: torch.Tensor, hidden_states_style: torch.Tensor=None) -> torch.Tensor:
        dtype = self.dtype
        merged_content = self.merge_content
        merged_style = self.merge_style
        
        if self.forward_type == "both":
            orig_dtype = hidden_states_content.dtype
                # print(hidden_states.shape)
            if hidden_states_style is None: 
                hidden_states_style = hidden_states_content
            if self.masked_matrix["content"] is True:
                up_hidden_states_content = torch.zeros((hidden_states_content.shape[0], hidden_states_content.shape[1], self.out_features)).to(hidden_states_content.device)
            else: 
                merged_content_weight = self.lora_matrix_dic["content_down"].weight.T @ \
                                        self.lora_matrix_dic["content_up"].weight.T 
                masked_content_weight = merged_content_weight * merged_content
                up_hidden_states_content = hidden_states_content.to(dtype) @ masked_content_weight
            
            if self.masked_matrix["style"] is True: 
                up_hidden_states_style = torch.zeros((hidden_states_style.shape[0], hidden_states_style.shape[1], self.out_features)).to(hidden_states_style.device)
            else: 
                merged_style_weight = self.lora_matrix_dic["style_down"].weight.T @ \
                                        self.lora_matrix_dic["style_up"].weight.T 
                masked_style_weight = merged_style_weight * merged_style
                up_hidden_states_style = hidden_states_style.to(dtype) @ masked_style_weight
            added_hidden_states = up_hidden_states_style.to(orig_dtype) + up_hidden_states_content.to(orig_dtype)
        if self.forward_type == "content":
            orig_dtype = hidden_states_content.dtype
            if self.masked_matrix["content"] is True:
                up_hidden_states_content = torch.zeros((hidden_states_content.shape[0], hidden_states_content.shape[1], self.out_features)).to(hidden_states_content.device)
            else: 
                merged_content_weight = self.lora_matrix_dic["content_down"].weight.T @ \
                                        self.lora_matrix_dic["content_up"].weight.T 
                masked_content_weight = merged_content_weight #* merged_content
                up_hidden_states_content = hidden_states_content.to(dtype) @ masked_content_weight
            added_hidden_states = up_hidden_states_content.to(orig_dtype)
        if self.forward_type == "style":
            orig_dtype = hidden_states_style.dtype
            if self.masked_matrix["style"] is True: 
                up_hidden_states_style = torch.zeros((hidden_states_style.shape[0], hidden_states_style.shape[1], self.out_features)).to(hidden_states_style.device)
            else: 
                merged_style_weight = self.lora_matrix_dic["style_down"].weight.T @ \
                                        self.lora_matrix_dic["style_up"].weight.T 
                masked_style_weight = merged_style_weight #* merged_style
                up_hidden_states_style = hidden_states_style.to(dtype) @ masked_style_weight
            added_hidden_states = up_hidden_states_style.to(orig_dtype)
        return added_hidden_states
