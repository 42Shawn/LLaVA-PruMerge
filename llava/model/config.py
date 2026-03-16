from transformers import PretrainedConfig

class LlavaConfig(PretrainedConfig):
    model_type = "llava-bitnet"

    def __init__(
        self,
        vision_tower=None,
        mm_hidden_size=None,
        mm_vision_select_layer=None,
        mm_use_im_start_end=True,
        mm_use_im_patch_token=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_tower = vision_tower
        self.mm_hidden_size = mm_hidden_size
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_use_im_start_end = mm_use_im_start_end
        self.mm_use_im_patch_token = mm_use_im_patch_token

