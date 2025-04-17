import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Union

try:
    import orjson as json
except:
    import json

import torch
import torch.distributed as dist
import numpy as np
import transformers
from PIL import Image, ImageFile, PngImagePlugin
from transformers import (HfArgumentParser, Trainer, TrainingArguments,
                          set_seed)
from transformers.trainer_utils import get_last_checkpoint
from transformers.processing_utils import ProcessorMixin
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)
from transformers.models.clip import CLIPVisionModel
from janus.models.siglip_vit import VisionTransformer
from janus.models.processing_vlm import VLChatProcessor
from janus.models.modeling_vlm import MultiModalityConfig, MultiModalityCausalLM
from janus.train_helper import (LazySupervisedDataset,
                                X2IGenDataset,
                                ConcatDataset,
                                WeightedConcatDataset,
                                PackedDataset,
                                init_dist,
                                packed_collate_fn,
                                replace_train_dataloader,
                                replace_train_sampler)

# Set constants for image processing and logging
IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to cache directory for pretrained models.'}
    )
    vision_type: Optional[str] = field(
        default='vit',
        metadata={'help': 'Specify the type of vision model, e.g., vit and patch'}
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM decoder.'},
    )
    freeze_vision: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the vision backbone of the model.'},
    )
    freeze_gen_vision: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the generation vision backbone of the model.'},
    )
    unfreeze_ffn: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the FFN layers of the model.'},
    )
    unfreeze_gen_ffn: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the generation FFN layers of the model.'},
    )
    unfreeze_attn: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the Attention layers of the model.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    unfreeze_aligner: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the aligner layers of the model.'},
    )
    unfreeze_gen_aligner: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the generation aligner layers of the model.'},
    )
    unfreeze_gen_head: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the generation head of the model.'},
    )
    unfreeze_gen_embed: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the generation embeddings of the model.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is last layer.'},
    )
    use_vision_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the vision model. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': "Set to True to unfreeze the language model's head."},
    )
    use_custom_trainer: bool = field(
        default=False,
        metadata={'help': 'Set to True to enable the use of a custom trainer.'},
    )
    grad_checkpoint: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use gradient checkpointing.'},
    )
    grad_checkpointing_kwargs: Optional[Union[dict, str]] = field(
        default=None,
        metadata={
            "help": "Gradient checkpointing key word arguments such as `use_reentrant`. Will be passed to `torch.utils.checkpoint.checkpoint` through `model.gradient_checkpointing_enable`."
        },
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT model. Default is 0.'},
    )
    ps_version: str = field(
        default='v1',
        metadata={'help': 'Specify the version of pixel shuffle implementation. Default is `v1`.'
                          'Please use `v2` to fix the bug of transposed image.'}
    )
    attn_implementation: Optional[str] = field(
        default='flash_attention_2',
        metadata={'help': 'The implementation of the attention. Default is `flash_attention_2`.'},
    )
    output_attentions: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether to output attentions. Default is False.'},
    )
    output_hidden_states: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether to output hidden states. Default is False.'},
    )
    return_dict: Optional[bool] = field(
        default=True,
        metadata={'help': 'Whether to return dict. Default is True.'},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    max_seq_length: Optional[int] = field(
        default=16384,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: Optional[int] = field(
        default=384,
        metadata={'help': 'Set the desired size for the image. Default is 384.'},
    )
    down_sample_ratio: Optional[float] = field(
        default=1.0,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 1.0.'},
    )
    pad2square: Optional[bool] = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True.'},
    )
    conv_style: Optional[str] = field(
        default='internvl_zh', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: Optional[str] = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    use_data_resampling: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling.'},
    )
    dynamic_image_size: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic image size.'},
    )
    use_thumbnail: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image.'},
    )
    min_dynamic_patch: Optional[int] = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: Optional[int] = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches. Default is 6.'},
    )
    neftune_alpha: Optional[float] = field(
        default=None,
        metadata={'help': 'The noise_alpha value for NEFTune. Default is None.'},
    )
    normalize_type: Optional[str] = field(
        default='imagenet',
        metadata={'help': 'The normalize type for the image. Default is imagenet.'},
    )
    use_packed_ds: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether to use packed dataset for training. Default is False.'},
    )
    num_images_expected: Optional[int] = field(
        default=12,
        metadata={'help': 'The maximum number of images per packed sample. Default is 12.'},
    )
    max_packed_tokens: Optional[int] = field(
        default=8192,
        metadata={'help': 'The required token length of per packed sample. Default is 8192.'},
    )
    max_buffer_size: Optional[int] = field(
        default=20,
        metadata={'help': 'The buffer size of the packed dataset. Default is 20.'},
    )
    log_freq: Optional[int] = field(
        default=1000,
        metadata={'help': 'The log frequence of the packed dataset. Default is 1000.'},
    )
    strict_mode: Optional[bool] = field(
        default=True,
        metadata={'help': 'Whether to pad the number of images to satisfy num_images_expected. Default is True.'},
    )
    replacement: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether to restart the dataset after it is exhausted. Default is False.'},
    )
    allow_overflow: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether to drop the sample over the specified max_packed_tokens. Default is False.'},
    )
    loss_reduction: Optional[str] = field(
        default='token',
        metadata={'help': 'Loss reduction method. Default is `token`'},
    )
    loss_reduction_all_gather: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether to all gahter when loss reduction. Default is False'},
    )


class CustomTrainer(Trainer):
    def __init__(self, *args, processor: Optional[ProcessorMixin] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor

    def _save_tpu(self, output_dir: Optional[str] = None):
        super()._save_tpu(output_dir)
        if self.processor is not None:
            self.processor.save_pretrained(output_dir)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super()._save(output_dir, state_dict)
        if self.processor is not None:
            self.processor.save_pretrained(output_dir)
    
    def _push_from_checkpoint(self, checkpoint_folder):
        super()._push_from_checkpoint(checkpoint_folder)
        output_dir = self.args.output_dir
        if self.processor is not None:
            self.processor.save_pretrained(output_dir)


def build_datasets(
    data_args,
    processor,
    tcs_loader,
    model,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    min_num_frame=8,
    max_num_frame=32,
    normalize_type='imagenet',
):
    datasets = []
    lengths = []
    data_rank = dist.get_rank()
    data_world_size = dist.get_world_size()
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_patch
        if ds_collections[ds_name].get('data_type') == 'generation':
            data_type = X2IGenDataset
        else:
            data_type = LazySupervisedDataset
        dataset = data_type(
            template_name=processor.sft_format if processor.sft_format is not None else data_args.conv_style,
            meta=ds_collections[ds_name],
            tokenizer=processor.tokenizer,
            tcs_loader=tcs_loader,
            ds_name=ds_name,
            num_image_token=model.vision_model.vision_tower_params.num_image_token,
            image_processor=processor.image_processor,
            image_size=data_args.force_image_size,
            is_train=ds_collections[ds_name]['data_augment'],
            pad2square=data_args.pad2square,
            background_color=processor.image_processor.background_color,
            group_by_length=group_by_length and not data_args.use_packed_ds,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            min_num_frame=min_num_frame,
            max_num_frame=max_num_frame,
            repeat_time=repeat_time,
            normalize_type=normalize_type,
            # hyperparameters for packed training
            use_packed_ds=data_args.use_packed_ds,
            data_rank=data_rank,
            data_world_size=data_world_size,
            distributed_mode=data_args.use_packed_ds,
            force_shuffle=data_args.use_packed_ds,
            random_seed=ds_idx,
        )
        logger.info(f'Add dataset: {ds_name} with length: {len(dataset)}')
        datasets.append(dataset)
        if data_args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))
        else:
            lengths.append(len(dataset))

    if data_args.use_packed_ds:
        total_length = sum(lengths)
        train_dataset = PackedDataset(
            processor=processor,
            data_rank=data_rank,
            data_world_size=data_world_size,
            datasets=datasets,
            dataset_weight=[l / total_length for l in lengths],
            num_images_expected=data_args.num_images_expected,
            max_packed_tokens=data_args.max_packed_tokens,
            max_buffer_size=data_args.max_buffer_size,
            log_freq=data_args.log_freq,
            strict_mode=data_args.strict_mode,
            replacement=data_args.replacement,
            allow_overflow=data_args.allow_overflow,
            allow_deduplicated_ds_name=False,
        )
    elif data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset


def pad_data_collator(features, pad_id=0):

    first = features[0]
    batch = {}

    batch_lens = [feat['input_ids'].shape for feat in features]
    max_item_length = max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]
        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
        feat['input_ids'] = temp_input_ids
        temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
        temp_labels[:feat['labels'].shape[0]] = feat['labels']
        feat['labels'] = temp_labels
        feat['attention_mask'] = feat['input_ids'].ne(pad_id)

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if 'label' in first and first['label'] is not None:
        label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
        dtype = torch.long if isinstance(label, int) else torch.float
        batch['labels'] = torch.tensor([f['label'] for f in features], dtype=dtype)
    elif 'label_ids' in first and first['label_ids'] is not None:
        if isinstance(first['label_ids'], torch.Tensor):
            batch['labels'] = torch.stack([f['label_ids'] for f in features])
        else:
            dtype = torch.long if isinstance(first['label_ids'][0], int) else torch.float
            batch['labels'] = torch.tensor([f['label_ids'] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ('label', 'label_ids') and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
    return batch


def concat_pad_data_collator(features, max_item_length=None, pad_id=0):

    first = features[0]
    batch = {}

    batch_lens = [feat['input_ids'].shape for feat in features]
    max_item_length = max_item_length or max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]
        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
        feat['input_ids'] = temp_input_ids
        temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
        temp_labels[:feat['labels'].shape[0]] = feat['labels']
        feat['labels'] = temp_labels
        feat['attention_mask'] = feat['input_ids'].ne(pad_id)

        if 'position_ids' in feat:
            temp_position_ids = torch.LongTensor([0] * max_item_length)
            temp_position_ids[:feat['position_ids'].shape[0]] = feat['position_ids']
            feat['position_ids'] = temp_position_ids

        if 'loss_weight' in feat:
            temp_loss_weight = torch.FloatTensor([0] * max_item_length)
            temp_loss_weight[:feat['loss_weight'].shape[0]] = feat['loss_weight']
            feat['loss_weight'] = temp_loss_weight

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if 'label' in first and first['label'] is not None:
        label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
        dtype = torch.long if isinstance(label, int) else torch.float
        batch['labels'] = torch.tensor([f['label'] for f in features], dtype=dtype)
    elif 'label_ids' in first and first['label_ids'] is not None:
        if isinstance(first['label_ids'], torch.Tensor):
            batch['labels'] = torch.stack([f['label_ids'] for f in features])
        else:
            dtype = torch.long if isinstance(first['label_ids'][0], int) else torch.float
            batch['labels'] = torch.tensor([f['label_ids'] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ('label', 'label_ids', 'pixel_values', 'image_flags') and \
                v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        if k in ('pixel_values', 'image_flags'):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.concat([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.concat(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.concat([f[k] for f in features])
    return batch


def len2weight(x, loss_reduction):
    if x == 0:
        return x
    if loss_reduction == 'token':
        return 1
    if loss_reduction == 'sample':
        return 1 / x
    if loss_reduction == 'square':
        return 1 / (x ** 0.5)
    raise NotImplementedError(loss_reduction)


def main():
    replace_train_sampler()
    replace_train_dataloader()

    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.use_packed_ds = data_args.use_packed_ds

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('InternV-Chat', model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    processor_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Processor: {processor_path}')
    processor = VLChatProcessor.from_pretrained(processor_path)
    tokenizer = processor.tokenizer
    tokenizer.model_max_length = data_args.max_seq_length
    # token_list = []
    # special_tokens_dict = {"additional_special_tokens": token_list}
    # tokenizer.add_special_tokens(special_tokens_dict)
    tcs_loader = None

    if data_args.use_packed_ds:
        # replace_attention_class()
        raise NotImplementedError

    logger.info('Loading Janus Model...')
    config = MultiModalityConfig.from_pretrained(model_args.model_name_or_path)
    config.vision_config.drop_path_rate = model_args.drop_path_rate

    assert model_args.attn_implementation
    if model_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise ValueError("The 'sdpa' attention implementation requires torch version 2.1.2 or higher.")
    config.language_config._attn_implementation = model_args.attn_implementation  # for LLaMA
    logger.info(f'Using {model_args.attn_implementation} for LLaMA')

    model = MultiModalityCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    model.image_id = processor.image_id
    model.config.output_attentions = model_args.output_attentions
    model.config.output_hidden_states = model_args.output_hidden_states
    model.config.return_dict = model_args.return_dict
    logger.info('Finished!')

    patch_size = model.vision_model.vision_tower_params.patch_size
    logger.info(f'data_args.force_image_size: {data_args.force_image_size}')
    logger.info(f'model.config.vision_config.params.image_size: {model.config.vision_config.params.image_size}')
    if model.config.vision_config.params.image_size != data_args.force_image_size:
        # logger.info(f'Resizing position embedding from '
        #             f'{model.config.vision_config.image_size} '
        #             f'to {data_args.force_image_size}...')
        # model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
        #                                          new_size=data_args.force_image_size,
        #                                          patch_size=patch_size)
        # model.config.vision_config.image_size = data_args.force_image_size
        raise NotImplementedError

    num_new_tokens = len(tokenizer) - model.config.language_config.vocab_size
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.config.language_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = False
    try:
        grad_checkpointing_kwargs = model_args.grad_checkpointing_kwargs
        if not isinstance(grad_checkpointing_kwargs, dict):
            grad_checkpointing_kwargs = json.loads(grad_checkpointing_kwargs)
    except:
        grad_checkpointing_kwargs = None
    if model_args.grad_checkpoint:
        model.language_model.gradient_checkpointing_enable(grad_checkpointing_kwargs)
        if isinstance(model.vision_model.vision_tower, VisionTransformer):
            model.vision_model.vision_tower.set_grad_checkpointing()
        elif isinstance(model.vision_model.vision_tower, CLIPVisionModel):
            model.vision_model.vision_tower.gradient_checkpointing_enable(grad_checkpointing_kwargs)

    data_args.normalize_type = (processor.image_processor.image_mean,
                                processor.image_processor.image_std)

    train_dataset = build_datasets(
        data_args, processor, tcs_loader, model, group_by_length=training_args.group_by_length,
        dynamic_image_size=data_args.dynamic_image_size, use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch,
        normalize_type=data_args.normalize_type)

    def _freeze_params(module, exclude_lists=[]):
        """
        Freeze parameters of a module unless their names include any substring from exclude_lists.

        Args:
        module (torch.nn.Module): The module whose parameters will be processed.
        exclude_lists (list of str): List of substrings to check against parameter names.
                                     Parameters with names including any of these substrings will not be frozen.
        """
        # Iterate over each parameter and its name in the module
        for name, param in module.named_parameters():
            # Check if the parameter name includes any substring from exclude_lists
            if any(exclude in name for exclude in exclude_lists):
                # If parameter name matches an exclude criterion, do not freeze it
                logger.info(f'Unfreezing layer-{name}')
                continue
            # Freeze the parameter by setting requires_grad to False
            param.requires_grad = False

    # Freeze vision model
    if model_args.freeze_vision:
        # model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)
        # Unfreeze aligner
        if not model_args.unfreeze_aligner:
            _freeze_params(model.aligner)

    # Freeze generation vision model
    if model_args.freeze_gen_vision:
        model.gen_vision_model = model.gen_vision_model.eval()
        _freeze_params(model.gen_vision_model)
        # Unfreeze generation aligner
        if not model_args.unfreeze_gen_aligner:
            _freeze_params(model.gen_aligner)
        # Unfreeze generation head
        if not model_args.unfreeze_gen_head:
            _freeze_params(model.gen_head)
        # Unfreeze generation embed
        if not model_args.unfreeze_gen_embed:
            _freeze_params(model.gen_embed)

    # Freeze LLM
    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        exclude = []
        if model_args.unfreeze_ffn:
            exclude.append('mlp.')
        if model_args.unfreeze_gen_ffn:
            exclude.append('mlp_gen.')
        if model_args.unfreeze_attn:
            exclude.append('self_attn.')
        _freeze_params(model.language_model, exclude)
        # Enable input require grads
        if exclude:
            model.language_model.enable_input_require_grads()

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad_(True)

    if model_args.use_vision_lora:
        model.wrap_vision_lora(r=model_args.use_vision_lora, lora_alpha=2 * model_args.use_vision_lora)
        model.config.use_vision_lora = model_args.use_vision_lora

    if model_args.use_llm_lora:
        model.wrap_llm_lora(r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora)
        model.config.use_llm_lora = model_args.use_llm_lora

    if model_args.unfreeze_vit_layers != 0:
        if isinstance(model.vision_model.vision_tower, VisionTransformer):
            layers = model.vision_model.vision_tower.blocks[model_args.unfreeze_vit_layers:]
        elif isinstance(model.vision_model.vision_tower, CLIPVisionModel):
            layers = model.vision_model.vision_tower.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            logger.info(f'Unfreezing ViT layer: {k}')
            v.requires_grad = True

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f'Trainable parameters: {name}')

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # Initialize our Trainer
    if model_args.use_custom_trainer:
        # replace_create_optimizer()
        raise NotImplementedError

    if data_args.use_packed_ds:
        collator = partial(
            packed_collate_fn,
            data_collator=concat_pad_data_collator,
            max_item_length=data_args.max_packed_tokens if data_args.strict_mode else 0,
            micro_num=training_args.train_batch_size,
            len2weight=partial(len2weight, loss_reduction=data_args.loss_reduction),
            loss_reduction_all_gather=data_args.loss_reduction_all_gather,
            pad_id=processor.pad_id
        )
    else:
        collator = partial(concat_pad_data_collator, pad_id=processor.pad_id)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
        processor=processor,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        try:
            metrics['train_samples'] = len(train_dataset)
        except:
            metrics['train_samples'] = -1

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()


if __name__ == '__main__':
    main()
