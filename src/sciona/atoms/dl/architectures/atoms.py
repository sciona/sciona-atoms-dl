"""Opaque neural-network architecture interface atoms.

These atoms do not instantiate PyTorch, timm, transformers, or other model
libraries. They encode the tensor boundary for common pretrained architecture
families and return zero-valued tensors with the documented output shape so
CDG validation can reason about interfaces without decomposing model internals.
Runtime systems should bind these contracts to concrete model implementations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_autoregressive_transformer_decoder,
    witness_densenet_backbone,
    witness_efficientnet_backbone,
    witness_mil_attention_aggregator,
    witness_recurrent_sequence_model,
    witness_resnet_family_backbone,
    witness_slowfast_video_network,
    witness_swin_transformer_backbone,
    witness_unet_1d_sequence,
    witness_unet_2d_segmentation,
    witness_whisper_asr_transformer,
    witness_yolo_object_detector,
)


_EFFICIENTNET_FEATURE_DIMS = {
    "b0": 1280,
    "b1": 1280,
    "b2": 1408,
    "b3": 1536,
    "b4": 1792,
    "b5": 2048,
    "b6": 2304,
    "b7": 2560,
    "tf_efficientnetv2_s": 1280,
    "tf_efficientnetv2_m": 1280,
    "tf_efficientnetv2_l": 1280,
}
_RESNET_FEATURE_DIMS = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
    "resnext50_32x4d": 2048,
    "resnext101_32x8d": 2048,
    "seresnext50_32x4d": 2048,
    "seresnext101_32x8d": 2048,
}
_DENSENET_FEATURE_DIMS = {
    "densenet121": 1024,
    "densenet161": 2208,
    "densenet169": 1664,
    "densenet201": 1920,
}
_SWIN_FEATURE_DIMS = {
    "swin_tiny_patch4_window7_224": 768,
    "swin_small_patch4_window7_224": 1024,
    "swin_base_patch4_window7_224": 1024,
    "swin_large_patch4_window7_224": 1536,
}
_WHISPER_MEL_BINS = {
    "tiny": 80,
    "base": 80,
    "small": 80,
    "medium": 80,
    "large-v2": 80,
    "large-v3": 128,
    "turbo": 128,
}
_YOLOX_VARIANTS = {"yolox_s", "yolox_m", "yolox_l"}
_ULTRALYTICS_FLAGGED_VARIANTS = {"yolov8n", "yolov8s"}


def _is_float32_tensor(array: NDArray[np.float32], rank: int) -> bool:
    tensor = np.asarray(array)
    return bool(tensor.dtype == np.float32 and tensor.ndim == rank and all(dim > 0 for dim in tensor.shape))


def _is_int64_tensor(array: NDArray[np.int64], rank: int) -> bool:
    tensor = np.asarray(array)
    return bool(tensor.dtype == np.int64 and tensor.ndim == rank and all(dim > 0 for dim in tensor.shape))


def _image_batch_valid(images: NDArray[np.float32]) -> bool:
    tensor = np.asarray(images)
    return bool(
        _is_float32_tensor(images, 4)
        and tensor.shape[1] == 3
        and tensor.shape[2] % 32 == 0
        and tensor.shape[3] % 32 == 0
        and np.all(np.isfinite(tensor))
    )


def _variant_dim(mapping: dict[str, int], variant: str) -> int:
    return mapping[variant]


@register_atom(witness_efficientnet_backbone)
@icontract.require(lambda images: _image_batch_valid(images), "images must be a finite float32 (B, 3, H, W) tensor with H and W divisible by 32")
@icontract.require(lambda variant: variant in _EFFICIENTNET_FEATURE_DIMS, "variant must be a supported EfficientNet family member")
@icontract.require(lambda pretrained_weights: len(pretrained_weights) > 0, "pretrained_weights must name the expected weight source")
@icontract.ensure(lambda images, variant, result: result.shape == (images.shape[0], _variant_dim(_EFFICIENTNET_FEATURE_DIMS, variant)), "result must match the variant feature dimension")
@icontract.ensure(lambda result: result.dtype == np.float32, "result must be float32")
def efficientnet_backbone(
    images: NDArray[np.float32],
    variant: str = "b0",
    pretrained_weights: str = "imagenet",
) -> NDArray[np.float32]:
    """Represent EfficientNet or EfficientNetV2 feature extraction.

    The atom accepts ImageNet-normalized image tensors and exposes the pooled
    feature-vector boundary used by timm-style backbones. It returns a zero
    feature tensor with the configured variant width.
    """
    _ = pretrained_weights
    return np.zeros((images.shape[0], _EFFICIENTNET_FEATURE_DIMS[variant]), dtype=np.float32)


@register_atom(witness_resnet_family_backbone)
@icontract.require(lambda images: _image_batch_valid(images), "images must be a finite float32 (B, 3, H, W) tensor with H and W divisible by 32")
@icontract.require(lambda variant: variant in _RESNET_FEATURE_DIMS, "variant must be a supported ResNet, ResNeXt, or SE-ResNeXt member")
@icontract.require(lambda pretrained_weights: len(pretrained_weights) > 0, "pretrained_weights must name the expected weight source")
@icontract.ensure(lambda images, variant, result: result.shape == (images.shape[0], _variant_dim(_RESNET_FEATURE_DIMS, variant)), "result must match the variant feature dimension")
@icontract.ensure(lambda result: result.dtype == np.float32, "result must be float32")
def resnet_family_backbone(
    images: NDArray[np.float32],
    variant: str = "resnet50",
    pretrained_weights: str = "imagenet",
) -> NDArray[np.float32]:
    """Represent ResNet, ResNeXt, or SE-ResNeXt feature extraction.

    The atom preserves the common residual-backbone boundary: normalized image
    batches enter, global-pooled feature vectors leave. It does not include a
    classification head.
    """
    _ = pretrained_weights
    return np.zeros((images.shape[0], _RESNET_FEATURE_DIMS[variant]), dtype=np.float32)


@register_atom(witness_densenet_backbone)
@icontract.require(lambda images: _image_batch_valid(images), "images must be a finite float32 (B, 3, H, W) tensor with H and W divisible by 32")
@icontract.require(lambda variant: variant in _DENSENET_FEATURE_DIMS, "variant must be a supported DenseNet member")
@icontract.require(lambda pretrained_weights: len(pretrained_weights) > 0, "pretrained_weights must name the expected weight source")
@icontract.ensure(lambda images, variant, result: result.shape == (images.shape[0], _variant_dim(_DENSENET_FEATURE_DIMS, variant)), "result must match the variant feature dimension")
@icontract.ensure(lambda result: result.dtype == np.float32, "result must be float32")
def densenet_backbone(
    images: NDArray[np.float32],
    variant: str = "densenet121",
    pretrained_weights: str = "imagenet",
) -> NDArray[np.float32]:
    """Represent DenseNet feature extraction after dense block concatenation.

    The output width follows the irregular DenseNet variant mapping while
    keeping the model internals opaque to the CDG.
    """
    _ = pretrained_weights
    return np.zeros((images.shape[0], _DENSENET_FEATURE_DIMS[variant]), dtype=np.float32)


@register_atom(witness_swin_transformer_backbone)
@icontract.require(lambda images: _image_batch_valid(images), "images must be a finite float32 (B, 3, H, W) tensor with H and W divisible by 32")
@icontract.require(lambda variant: variant in _SWIN_FEATURE_DIMS, "variant must be a supported Swin Transformer member")
@icontract.require(lambda pretrained_weights: len(pretrained_weights) > 0, "pretrained_weights must name the expected weight source")
@icontract.ensure(lambda images, variant, result: result.shape == (images.shape[0], _variant_dim(_SWIN_FEATURE_DIMS, variant)), "result must match the variant feature dimension")
@icontract.ensure(lambda result: result.dtype == np.float32, "result must be float32")
def swin_transformer_backbone(
    images: NDArray[np.float32],
    variant: str = "swin_tiny_patch4_window7_224",
    pretrained_weights: str = "imagenet",
) -> NDArray[np.float32]:
    """Represent pooled Swin Transformer image features.

    The atom models the hierarchical shifted-window backbone boundary and
    returns the pooled embedding width associated with the chosen variant.
    """
    _ = pretrained_weights
    return np.zeros((images.shape[0], _SWIN_FEATURE_DIMS[variant]), dtype=np.float32)


@register_atom(witness_unet_2d_segmentation)
@icontract.require(lambda images: _is_float32_tensor(images, 4), "images must be a float32 (B, C, H, W) tensor")
@icontract.require(lambda images: images.shape[2] % 32 == 0 and images.shape[3] % 32 == 0, "height and width must be divisible by 32")
@icontract.require(lambda images: np.all(np.isfinite(images)), "images must contain finite values")
@icontract.require(lambda num_classes: num_classes >= 1, "num_classes must be positive")
@icontract.require(lambda architecture: architecture in {"unet", "unetplusplus"}, "architecture must be unet or unetplusplus")
@icontract.require(lambda encoder_name: len(encoder_name) > 0, "encoder_name must be non-empty")
@icontract.require(lambda encoder_weights: len(encoder_weights) > 0, "encoder_weights must be non-empty")
@icontract.ensure(lambda images, num_classes, result: result.shape == (images.shape[0], num_classes, images.shape[2], images.shape[3]), "result must preserve spatial dimensions")
@icontract.ensure(lambda result: result.dtype == np.float32, "result must be float32")
def unet_2d_segmentation(
    images: NDArray[np.float32],
    num_classes: int,
    architecture: str = "unet",
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
) -> NDArray[np.float32]:
    """Represent U-Net or U-Net++ dense 2D segmentation logits.

    The atom exposes the segmentation contract from input image tensors to
    per-pixel class logits while leaving encoder and decoder internals opaque.
    """
    _ = architecture, encoder_name, encoder_weights
    return np.zeros((images.shape[0], num_classes, images.shape[2], images.shape[3]), dtype=np.float32)


@register_atom(witness_unet_1d_sequence)
@icontract.require(lambda sequence: _is_float32_tensor(sequence, 3), "sequence must be a float32 (B, C, L) tensor")
@icontract.require(lambda sequence: sequence.shape[2] % 32 == 0, "sequence length must be divisible by 32")
@icontract.require(lambda sequence: np.all(np.isfinite(sequence)), "sequence must contain finite values")
@icontract.require(lambda num_classes: num_classes >= 1, "num_classes must be positive")
@icontract.require(lambda architecture: architecture in {"vanilla_1d", "attention_unet_1d", "residual_unet_1d"}, "architecture must be a supported 1D U-Net variant")
@icontract.ensure(lambda sequence, num_classes, result: result.shape == (sequence.shape[0], num_classes, sequence.shape[2]), "result must preserve sequence length")
@icontract.ensure(lambda result: result.dtype == np.float32, "result must be float32")
def unet_1d_sequence(
    sequence: NDArray[np.float32],
    num_classes: int,
    architecture: str = "vanilla_1d",
) -> NDArray[np.float32]:
    """Represent 1D U-Net sequence segmentation logits.

    The atom maps channel-first temporal tensors to per-step class logits and
    enforces the downsampling divisibility expected by U-Net style decoders.
    """
    _ = architecture
    return np.zeros((sequence.shape[0], num_classes, sequence.shape[2]), dtype=np.float32)


@register_atom(witness_yolo_object_detector)
@icontract.require(lambda images: _image_batch_valid(images), "images must be a finite float32 (B, 3, H, W) tensor with H and W divisible by 32")
@icontract.require(lambda num_classes: num_classes >= 1, "num_classes must be positive")
@icontract.require(lambda num_predictions: num_predictions >= 1, "num_predictions must be positive")
@icontract.require(lambda variant: variant in _YOLOX_VARIANTS | _ULTRALYTICS_FLAGGED_VARIANTS, "variant must be a supported YOLO-style detector")
@icontract.ensure(lambda images, num_classes, num_predictions, result: result.shape == (images.shape[0], 4 + num_classes, num_predictions), "result must be raw detector predictions before NMS")
@icontract.ensure(lambda result: result.dtype == np.float32, "result must be float32")
def yolo_object_detector(
    images: NDArray[np.float32],
    num_classes: int,
    num_predictions: int = 8400,
    variant: str = "yolox_s",
) -> NDArray[np.float32]:
    """Represent raw single-stage YOLO-family detector predictions.

    The default contract is the Apache-2.0 YOLOX family. Ultralytics variants
    are accepted only as flagged interface names because the research notes
    AGPL-3.0 licensing risk for those bindings.
    """
    _ = variant
    return np.zeros((images.shape[0], 4 + num_classes, num_predictions), dtype=np.float32)


@register_atom(witness_whisper_asr_transformer)
@icontract.require(lambda log_mel_spectrograms: _is_float32_tensor(log_mel_spectrograms, 3), "log_mel_spectrograms must be a float32 (B, n_mels, sequence_length) tensor")
@icontract.require(lambda decoder_input_ids: _is_int64_tensor(decoder_input_ids, 2), "decoder_input_ids must be an int64 (B, target_seq_len) tensor")
@icontract.require(lambda log_mel_spectrograms, decoder_input_ids: log_mel_spectrograms.shape[0] == decoder_input_ids.shape[0], "batch sizes must match")
@icontract.require(lambda variant: variant in _WHISPER_MEL_BINS, "variant must be a supported Whisper member")
@icontract.require(lambda log_mel_spectrograms, variant: log_mel_spectrograms.shape[1] == _WHISPER_MEL_BINS[variant], "mel-bin count must match the Whisper variant")
@icontract.require(lambda vocab_size: vocab_size >= 1, "vocab_size must be positive")
@icontract.require(lambda task: task in {"transcribe", "translate"}, "task must be transcribe or translate")
@icontract.ensure(lambda decoder_input_ids, vocab_size, result: result.shape == (decoder_input_ids.shape[0], decoder_input_ids.shape[1], vocab_size), "result must be decoder token logits")
@icontract.ensure(lambda result: result.dtype == np.float32, "result must be float32")
def whisper_asr_transformer(
    log_mel_spectrograms: NDArray[np.float32],
    decoder_input_ids: NDArray[np.int64],
    variant: str = "large-v3",
    vocab_size: int = 51865,
    task: str = "transcribe",
) -> NDArray[np.float32]:
    """Represent Whisper encoder-decoder ASR token logits.

    The atom starts after audio resampling and log-Mel extraction, receives
    decoder prompt tokens, and returns next-token logits without beam search.
    """
    _ = log_mel_spectrograms, variant, task
    return np.zeros((decoder_input_ids.shape[0], decoder_input_ids.shape[1], vocab_size), dtype=np.float32)


@register_atom(witness_autoregressive_transformer_decoder)
@icontract.require(lambda target_tokens: _is_int64_tensor(target_tokens, 2), "target_tokens must be an int64 (B, seq_len) tensor")
@icontract.require(lambda encoder_hidden_states: _is_float32_tensor(encoder_hidden_states, 3), "encoder_hidden_states must be a float32 (B, num_patches, D) tensor")
@icontract.require(lambda target_tokens, encoder_hidden_states: target_tokens.shape[0] == encoder_hidden_states.shape[0], "batch sizes must match")
@icontract.require(lambda vocab_size: vocab_size >= 1, "vocab_size must be positive")
@icontract.require(lambda num_layers: num_layers >= 1, "num_layers must be positive")
@icontract.require(lambda num_heads: num_heads >= 1, "num_heads must be positive")
@icontract.require(lambda hidden_dim: hidden_dim >= 1, "hidden_dim must be positive")
@icontract.ensure(lambda target_tokens, vocab_size, result: result.shape == (target_tokens.shape[0], target_tokens.shape[1], vocab_size), "result must be teacher-forced token logits")
@icontract.ensure(lambda result: result.dtype == np.float32, "result must be float32")
def autoregressive_transformer_decoder(
    target_tokens: NDArray[np.int64],
    encoder_hidden_states: NDArray[np.float32],
    vocab_size: int,
    num_layers: int = 6,
    num_heads: int = 8,
    hidden_dim: int = 512,
) -> NDArray[np.float32]:
    """Represent a teacher-forced autoregressive transformer decoder.

    The atom covers masked self-attention plus cross-attention over encoder
    states, but excludes the greedy or beam-search generation loop.
    """
    _ = encoder_hidden_states, num_layers, num_heads, hidden_dim
    return np.zeros((target_tokens.shape[0], target_tokens.shape[1], vocab_size), dtype=np.float32)


@register_atom(witness_recurrent_sequence_model)
@icontract.require(lambda sequence_features: _is_float32_tensor(sequence_features, 3), "sequence_features must be a float32 (B, L, D_in) tensor")
@icontract.require(lambda sequence_features: np.all(np.isfinite(sequence_features)), "sequence_features must contain finite values")
@icontract.require(lambda hidden_size: hidden_size >= 1, "hidden_size must be positive")
@icontract.require(lambda num_layers: num_layers >= 1, "num_layers must be positive")
@icontract.require(lambda cell_type: cell_type in {"gru", "lstm"}, "cell_type must be gru or lstm")
@icontract.require(lambda dropout: 0.0 <= dropout < 1.0, "dropout must be in [0, 1)")
@icontract.ensure(lambda sequence_features, hidden_size, bidirectional, return_sequences, result: result.shape == ((sequence_features.shape[0], sequence_features.shape[1], hidden_size * (2 if bidirectional else 1)) if return_sequences else (sequence_features.shape[0], hidden_size * (2 if bidirectional else 1))), "result must match the recurrent output mode")
@icontract.ensure(lambda result: result.dtype == np.float32, "result must be float32")
def recurrent_sequence_model(
    sequence_features: NDArray[np.float32],
    hidden_size: int = 128,
    num_layers: int = 1,
    bidirectional: bool = True,
    return_sequences: bool = True,
    cell_type: str = "gru",
    dropout: float = 0.0,
) -> NDArray[np.float32]:
    """Represent stacked GRU or LSTM sequence processing.

    The atom keeps the recurrent cell opaque while enforcing batch-major
    sequence input and the hidden-state width implied by directionality.
    """
    _ = num_layers, cell_type, dropout
    output_dim = hidden_size * (2 if bidirectional else 1)
    if return_sequences:
        return np.zeros((sequence_features.shape[0], sequence_features.shape[1], output_dim), dtype=np.float32)
    return np.zeros((sequence_features.shape[0], output_dim), dtype=np.float32)


@register_atom(witness_slowfast_video_network)
@icontract.require(lambda frames: len(frames) == 2, "frames must contain [slow_frames, fast_frames]")
@icontract.require(lambda frames: _is_float32_tensor(frames[0], 5) and _is_float32_tensor(frames[1], 5), "both pathways must be float32 (B, C, T, H, W) tensors")
@icontract.require(lambda frames: frames[0].shape[0] == frames[1].shape[0] and frames[0].shape[1] == frames[1].shape[1], "pathways must share batch and channel counts")
@icontract.require(lambda frames: frames[0].shape[3] == frames[1].shape[3] and frames[0].shape[4] == frames[1].shape[4], "pathways must share spatial dimensions")
@icontract.require(lambda frames: frames[0].shape[3] % 32 == 0 and frames[0].shape[4] % 32 == 0, "height and width must be divisible by 32")
@icontract.require(lambda frames: frames[1].shape[2] == 8 * frames[0].shape[2], "fast pathway should have eight times the slow temporal length")
@icontract.require(lambda frames: np.all(np.isfinite(frames[0])) and np.all(np.isfinite(frames[1])), "frames must contain finite values")
@icontract.require(lambda num_classes: num_classes >= 1, "num_classes must be positive")
@icontract.require(lambda variant: variant in {"slowfast_r50", "slowfast_r101"}, "variant must be a supported SlowFast member")
@icontract.require(lambda slowfast_channel_reduction_ratio: slowfast_channel_reduction_ratio >= 1, "slowfast_channel_reduction_ratio must be positive")
@icontract.ensure(lambda frames, num_classes, result: result.shape == (frames[0].shape[0], num_classes), "result must be class logits")
@icontract.ensure(lambda result: result.dtype == np.float32, "result must be float32")
def slowfast_video_network(
    frames: list[NDArray[np.float32]],
    num_classes: int = 400,
    variant: str = "slowfast_r50",
    slowfast_channel_reduction_ratio: int = 8,
) -> NDArray[np.float32]:
    """Represent SlowFast dual-pathway action-recognition logits.

    The input is the PyTorchVideo-style pair of slow and fast frame tensors.
    The atom validates pathway alignment and returns class-logit shape only.
    """
    _ = variant, slowfast_channel_reduction_ratio
    return np.zeros((frames[0].shape[0], num_classes), dtype=np.float32)


@register_atom(witness_mil_attention_aggregator)
@icontract.require(lambda patch_embeddings: _is_float32_tensor(patch_embeddings, 3), "patch_embeddings must be a float32 (B, N, D) tensor")
@icontract.require(lambda patch_embeddings: np.all(np.isfinite(patch_embeddings)), "patch_embeddings must contain finite values")
@icontract.require(lambda num_classes: num_classes >= 1, "num_classes must be positive")
@icontract.require(lambda attention_hidden_dim: attention_hidden_dim >= 1, "attention_hidden_dim must be positive")
@icontract.require(lambda variant: variant in {"gated_attention_mil", "standard_attention_mil"}, "variant must be a supported MIL aggregator")
@icontract.ensure(lambda patch_embeddings, num_classes, result: result.shape == (patch_embeddings.shape[0], num_classes), "result must be slide-level logits")
@icontract.ensure(lambda result: result.dtype == np.float32, "result must be float32")
def mil_attention_aggregator(
    patch_embeddings: NDArray[np.float32],
    num_classes: int,
    attention_hidden_dim: int = 256,
    variant: str = "gated_attention_mil",
) -> NDArray[np.float32]:
    """Represent attention pooling for multiple-instance learning.

    The atom receives variable-size bags of patch embeddings and exposes the
    slide-level class-logit boundary without implementing an attention module.
    """
    _ = attention_hidden_dim, variant
    return np.zeros((patch_embeddings.shape[0], num_classes), dtype=np.float32)

