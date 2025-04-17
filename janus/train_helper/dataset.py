import io
import re
import sys
from typing import Dict

import torch
import torchvision.transforms as T
import transformers
from PIL import Image
from transformers.trainer_pt_utils import LabelSmoother
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from torchvision.transforms.functional import InterpolationMode

from janus.utils.conversation import get_conv_template
from janus.models.image_processing_vlm import expand2square

from .constants import (CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD,
                        IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN,
                        SIGLIP_MEAN, SIGLIP_STD)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class WeightedConcatDataset(ConcatDataset):
    def __init__(self, datasets, weights):
        super().__init__(datasets)
        self.weights = torch.DoubleTensor(weights)
        self.total_size = sum(len(d) for d in datasets)
        self.sampler = WeightedRandomSampler(weights=self.weights, num_samples=self.total_size, replacement=True)

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return self.total_size


def simulate_jpeg_degradation(quality):
    def jpeg_degrade(img):
        with io.BytesIO() as output:
            img.convert('RGB').save(output, format='JPEG', quality=quality)
            output.seek(0)  # Move the reading cursor to the start of the stream
            img_jpeg = Image.open(output).copy()  # Use .copy() to make sure the image is loaded in memory
        return img_jpeg
    return jpeg_degrade


# Define the JPEG compression quality range, pre-create all JPEG compression functions
qualities = list(range(75, 101))
jpeg_degrade_functions = {quality: simulate_jpeg_degradation(quality) for quality in qualities}


def build_transform(is_train, input_size, pad2square=False, normalize_type='imagenet', background_color=None):
    if normalize_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == 'clip':
        MEAN, STD = CLIP_MEAN, CLIP_STD
    elif normalize_type == 'siglip':
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
    elif isinstance(normalize_type, tuple) and len(normalize_type) == 2:
        MEAN, STD = normalize_type
    else:
        raise NotImplementedError
    if is_train:  # use data augumentation
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomChoice([T.Lambda(jpeg_degrade_functions[quality]) for quality in qualities]),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        if pad2square is False:  # now we use this transform function by default
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
        else:
            if background_color is None:
                background_color = tuple(int(x * 255) for x in MEAN)
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Lambda(lambda img: expand2square(img, background_color)),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])

    return transform


def build_transform_generation(input_size, normalize_type='imagenet'):
    if normalize_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == 'clip':
        MEAN, STD = CLIP_MEAN, CLIP_STD
    elif normalize_type == 'siglip':
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize(input_size, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def preprocess(
    template_name,
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    num_image_token_list: list,
    text_only: bool = False,
    group_by_length: bool = False,
    use_packed_ds: bool = False,
    ds_name: str = None,
    num_image: int = 1
) -> Dict:
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}'
                conversation = conversation.replace('<image>', image_tokens, 1)
            new_conversations.append(conversation)
        conversations = new_conversations

    # Tokenize conversations
    padding = False if group_by_length or use_packed_ds else 'max_length'
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=padding,
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ': '
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        # Handle the padding
        if padding and tokenizer.padding_side == 'left':
            cur_len += tokenizer.model_max_length - total_len
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == '':
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the space token
            if target[cur_len + instruction_len].item() == tokenizer.encode(' ')[-1]:
                instruction_len += 1

            # Ignore the user instructions
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            logger.info(tokenizer.decode(z))
            exit()

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}.'
                    f' #turn = {len(turns) - 1}. (ignored). This dataset is {ds_name}.'
                )
                sys.stdout.flush()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def preprocess_generation(
    template_name,
    instructions,
    tokenizer: transformers.PreTrainedTokenizer,
    num_image_token_list: list,
    group_by_length: bool = False,
    use_packed_ds: bool = False,
    ds_name: str = None,
    num_image: int = 1
) -> Dict:
    conv = get_conv_template(template_name)

    # Apply prompt templates
    conversations = []
    for instruction in instructions:
        text = re.sub(r'<img><\|image_\d+\|></img>', '<image>', instruction)
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], '<image>')
        conversations.append(conv.get_prompt())

    new_conversations = []
    for conversation in conversations:
        for i in range(num_image):
            image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}'
            conversation = conversation.replace('<image>', image_tokens, 1)
        new_conversations.append(conversation)
    conversations = new_conversations

    padding = False if group_by_length or use_packed_ds else 'max_length'
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=padding,
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ': '
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        # Handle the padding
        if padding and tokenizer.padding_side == 'left':
            cur_len += tokenizer.model_max_length - total_len
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == '':
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            instruction_len = len(tokenizer(parts[0]).input_ids)

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            logger.info(tokenizer.decode(z))
            exit()

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}.'
                    f' #turn = {len(turns) - 1}. (ignored). This dataset is {ds_name}.'
                )
                sys.stdout.flush()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images
