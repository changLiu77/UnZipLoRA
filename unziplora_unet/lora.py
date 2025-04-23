from typing import Optional
import torch
from diffusers.models.lora import LoRACompatibleLinear, LoRALinearLayer
from torch import nn 

class LoRACompatibleLinear(nn.Linear):
    """
    A Linear layer that can be used with LoRA.
    """

    def __init__(self, *args, lora_layer: Optional[LoRALinearLayer] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_layer = lora_layer

    def set_lora_layer(self, lora_layer: Optional[LoRALinearLayer]):
        self.lora_layer = lora_layer

    def _fuse_lora(self, lora_scale: float = 1.0, safe_fusing: bool = False):
        if self.lora_layer is None:
            return

        dtype, device = self.weight.data.dtype, self.weight.data.device

        w_orig = self.weight.data.float()
        w_up = self.lora_layer.up.weight.data.float()
        w_down = self.lora_layer.down.weight.data.float()

        if self.lora_layer.network_alpha is not None:
            w_up = w_up * self.lora_layer.network_alpha / self.lora_layer.rank

        fused_weight = w_orig + (lora_scale * torch.bmm(w_up[None, :], w_down[None, :])[0])

        if safe_fusing and torch.isnan(fused_weight).any().item():
            raise ValueError(
                "This LoRA weight seems to be broken. "
                f"Encountered NaN values when trying to fuse LoRA weights for {self}."
                "LoRA weights will not be fused."
            )

        self.weight.data = fused_weight.to(device=device, dtype=dtype)

        # we can drop the lora layer now
        self.lora_layer = None

        # offload the up and down matrices to CPU to not blow the memory
        self.w_up = w_up.cpu()
        self.w_down = w_down.cpu()
        self._lora_scale = lora_scale

    def _unfuse_lora(self):
        if not (getattr(self, "w_up", None) is not None and getattr(self, "w_down", None) is not None):
            return

        fused_weight = self.weight.data
        dtype, device = fused_weight.dtype, fused_weight.device

        w_up = self.w_up.to(device=device).float()
        w_down = self.w_down.to(device).float()

        unfused_weight = fused_weight.float() - (self._lora_scale * torch.bmm(w_up[None, :], w_down[None, :])[0])
        self.weight.data = unfused_weight.to(device=device, dtype=dtype)

        self.w_up = None
        self.w_down = None

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0, hidden_states_1: torch.Tensor = None, hidden_states_2: torch.Tensor = None) -> torch.Tensor:
        # * compute results with lora layers
        if self.lora_layer is None:
            out = super().forward(hidden_states)
            return out
        else:
            out = super().forward(hidden_states)
               
            lora_output = (scale * self.lora_layer(hidden_states_1, hidden_states_2))
            return out + lora_output