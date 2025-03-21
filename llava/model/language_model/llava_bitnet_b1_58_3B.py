from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM

from bitnet_b1_58_3B.configuration_bitnet import BitnetConfig
from bitnet_b1_58_3B.modeling_bitnet import BitnetForCausalLM,BitnetMLP


from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaBitnet_b1_58_3BConfig(BitnetConfig):
    model_type = "LlavaBitnet_b1_58_3B"


class LlavaBitnet_b1_58_3BModel(LlavaMetaModel, BitnetMLP):
    config_class = LlavaBitnet_b1_58_3BConfig

    def __init__(self, config: BitnetConfig):
        super(LlavaBitnet_b1_58_3BModel, self).__init__(config)


class LlavaBitne_b1_58_3BForCausalLM(BitnetForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaBitnet_b1_58_3BConfig
 
    def __init__(self, config):
        super(BitnetForCausalLM, self).__init__(config)
        self.model = LlavaBitnet_b1_58_3BModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.embed_layer = self.get_input_embeddings()
        self.model.vision_tower.load_model()

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("LlavaBitnet1_58-3B", LlavaOLMoBitnet1BConfig)
AutoModelForCausalLM.register(LlavaOLMoBitnet1BConfig, LlavaOLMoBitnet1BForCausalLM)