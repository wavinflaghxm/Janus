# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import numpy as np
import PIL.Image

# specify the path to the model
# model_path = "deepseek-ai/Janus-1.3B"
model_path = "work_dirs/janus_finetune_x2i"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

conversation = [
    {
        "role": "User",
        "content": "Alter the posture of person in the picture <image_placeholder> to reflect pose <image_placeholder>.",
        "images": ["data/OmniGen/X2I-mm-instruction/fashiontryon/train/705/2.jpg",
                   "data/OmniGen/X2I-mm-instruction/fashiontryon/train/705/0_target/segment_vis.png"],
    },
    {"role": "Assistant", "content": f"{vl_chat_processor.image_start_tag}"},
]

pil_images = load_pil_images(conversation)
vl_chat_processor.system_prompt = ""
prepare_inputs = vl_chat_processor(
    conversations=conversation, images=pil_images, force_batchify=True
).to(vl_gpt.device)


@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prepare_inputs,
    temperature: float = 1,
    parallel_size: int = 16,
    cfg_weight: float = 5,
    img_cfg_weight: float = 0,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    input_ids = prepare_inputs.input_ids[0][:-1]  # remove the end token
    inputs_embeds = mmgpt.prepare_inputs_embeds(**prepare_inputs)[0][:-1]  # remove the end token

    # Use the VQ to encode the image
    pixel_values = prepare_inputs.pixel_values.squeeze(0)
    images_seq_mask = prepare_inputs.images_seq_mask.squeeze(0)[:-1]  # remove the end token
    encode_indices = mmgpt.gen_vision_model.encode(pixel_values)[2][2]
    encode_embeds = mmgpt.prepare_gen_img_embeds(encode_indices)
    inputs_embeds[images_seq_mask] = encode_embeds

    img_cond_mask = images_seq_mask | (input_ids == vl_chat_processor.image_start_id) | (input_ids == vl_chat_processor.image_end_id)
    assert img_cond_mask[-1] == True  # image_start_tag
    img_cond_mask[-1] = False
    img_cond_embeds = inputs_embeds[img_cond_mask]

    num_cond = 3 if img_cfg_weight > 0 else 2
    tokens = torch.zeros((parallel_size * num_cond, len(input_ids)), dtype=torch.int).cuda()
    token_embeds = torch.zeros((parallel_size * num_cond, *inputs_embeds.shape), dtype=inputs_embeds.dtype).cuda()
    for i in range(parallel_size * num_cond):
        tokens[i, :] = input_ids
        token_embeds[i, :] = inputs_embeds
        if i % num_cond != 0:
            tokens[i, 1: -1] = vl_chat_processor.pad_id
            token_embeds[i, :] = mmgpt.language_model.get_input_embeddings()(tokens[i, :])
            if i % num_cond == 2:
                token_embeds[i, 1: 1 + img_cond_embeds.shape[0]] = img_cond_embeds

    inputs_embeds = token_embeds
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::num_cond, :]
        logit_uncond = logits[1::num_cond, :]

        if num_cond == 3:
            logit_img_cond = logits[2::num_cond, :]
            logits = logit_uncond + img_cfg_weight * (logit_img_cond - logit_uncond) + cfg_weight * (logit_cond - logit_img_cond)
        else:
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1) for _ in range(num_cond)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    os.makedirs('generated_edit_samples', exist_ok=True)
    for i in range(parallel_size):
        save_path = os.path.join('generated_edit_samples', "img_{}.jpg".format(i))
        PIL.Image.fromarray(visual_img[i]).save(save_path)


generate(
    vl_gpt,
    vl_chat_processor,
    prepare_inputs,
    parallel_size=5,
    img_cfg_weight=2,
)