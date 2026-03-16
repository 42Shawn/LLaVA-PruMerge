import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from bitnet_b1_58_3B.tokenization_bitnet import BitnetTokenizer
from bitnet_b1_58_3B.configuration_bitnet import BitnetConfig
from llava.model import *
from bitnet_b1_58_3B.modeling_bitnet import BitnetForCausalLM
import os
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def get_image_preroccesor(model, tokenizer, device="cuda"):
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor
    return image_processor



def main():
    # Model
    disable_torch_init()

    #model_name = get_model_name_from_path(args.model_path)
    #tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    
    model_path = "/home/zl986/LLaVA-PruMerge-BitNet/checkpoints/llava-bitnet_b1_58_3B_pretrain_updated"
    config = BitnetConfig.from_pretrained(model_path)
    
    model = LlavaBitnet_b1_58_3BForCausalLM.from_pretrained("/home/zl986/LLaVA-PruMerge-BitNet/bitnet_b1_58_3B", config=config, torch_dtype = torch.float16)
    tokenizer = BitnetTokenizer.from_pretrained("/home/zl986/LLaVA-PruMerge-BitNet/bitnet_b1_58_3B")

    mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
    mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
    model.load_state_dict(mm_projector_weights, strict=False)

    model = model.cuda()
    

    # print(f"Pad token: {tokenizer.pad_token}")
    # print(f"Unk token: {tokenizer.unk_token}")
    
    # token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
    # if model.lm_head.weight.shape[0] != token_num:
    #     model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
    #     model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
    
    # if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
    #     non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
    
    # non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
    # if any(k.startswith('model.model.') for k in non_lora_trainables):
    #     non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
    # model.load_state_dict(non_lora_trainables, strict=False)

    # from peft import PeftModel
    # print('Loading LoRA weights...')
    # model = PeftModel.from_pretrained(model, model_path)
    # print('Merging LoRA weights...')
    # model = model.merge_and_unload()
    # print('Model is loaded...')
    
    # model = model.cuda()

   

    image_pre = get_image_preroccesor(model, tokenizer)
    vision_tower = model.get_vision_tower()
    print(f"Is vision tower loaded? {vision_tower.is_loaded}")


    image = load_image("/home/zl986/LLaVA-PruMerge-BitNet/playground/data/textvqa/train_images/0a0bc91825468c45.jpg")
    #image = None
    
    image_tensor = process_images([image], image_pre, model.config)
    #image_tensor = torch.randn(image_tensor.shape[0], image_tensor.shape[1], model.config.hidden_size, device=model.device) * 0.01
    
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)



    conv_mode = "llama_2"
    conv = conv_templates["llama_2"].copy()
    roles = conv.roles
    print(roles)

    try:
        inp = input(f"{roles[0]}: ")
    except EOFError:
        inp = ""
    if not inp:
        print("exit...")
        return

    print(f"{roles[1]}s: ", end="")

    if image is not None:
        # first message
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        image = None
    else:
            # later messages
        conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    #prompt = "Are you conscious?"

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    # print(input_ids.padding)
    # print(input_ids.type())
    # print(image_tensor.type())

    # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # keywords = [stop_str]
    # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    #prompt = "Are you conscious? Can you talk to me?"
    #inputs = tokenizer(prompt, return_tensors="pt")
    

    #inputs = tokenizer(prompt, return_tensors="pt")
    #print(input_ids)
    generate_ids = model.generate(input_ids, images = image_tensor, do_sample=True)

    # print("Generated Token IDs:", generate_ids.sequences)
    # print("Logits:", generate_ids.scores)

    outputs = tokenizer.decode(generate_ids[0, input_ids.shape[1]:])
    #outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(outputs)

if __name__ == "__main__":
    main()