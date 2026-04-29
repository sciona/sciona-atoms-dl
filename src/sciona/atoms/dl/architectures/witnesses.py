"""Shape witnesses for opaque neural-network architecture atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_efficientnet_backbone(
    images: AbstractArray,
    variant: str = "b0",
    pretrained_weights: str = "imagenet",
) -> AbstractArray:
    """Witness EfficientNet image batches becoming feature vectors."""
    return AbstractArray()


def witness_resnet_family_backbone(
    images: AbstractArray,
    variant: str = "resnet50",
    pretrained_weights: str = "imagenet",
) -> AbstractArray:
    """Witness residual-family image batches becoming feature vectors."""
    return AbstractArray()


def witness_densenet_backbone(
    images: AbstractArray,
    variant: str = "densenet121",
    pretrained_weights: str = "imagenet",
) -> AbstractArray:
    """Witness DenseNet image batches becoming feature vectors."""
    return AbstractArray()


def witness_swin_transformer_backbone(
    images: AbstractArray,
    variant: str = "swin_tiny_patch4_window7_224",
    pretrained_weights: str = "imagenet",
) -> AbstractArray:
    """Witness Swin image batches becoming pooled feature vectors."""
    return AbstractArray()


def witness_unet_2d_segmentation(
    images: AbstractArray,
    num_classes: int,
    architecture: str = "unet",
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
) -> AbstractArray:
    """Witness 2D segmentation preserving image height and width."""
    return AbstractArray()


def witness_unet_1d_sequence(
    sequence: AbstractArray,
    num_classes: int,
    architecture: str = "vanilla_1d",
) -> AbstractArray:
    """Witness temporal segmentation preserving sequence length."""
    return AbstractArray()


def witness_yolo_object_detector(
    images: AbstractArray,
    num_classes: int,
    num_predictions: int = 8400,
    variant: str = "yolox_s",
) -> AbstractArray:
    """Witness single-stage detector logits before NMS."""
    return AbstractArray()


def witness_whisper_asr_transformer(
    log_mel_spectrograms: AbstractArray,
    decoder_input_ids: AbstractArray,
    variant: str = "large-v3",
    vocab_size: int = 51865,
    task: str = "transcribe",
) -> AbstractArray:
    """Witness Whisper spectrograms and decoder tokens becoming token logits."""
    return AbstractArray()


def witness_autoregressive_transformer_decoder(
    target_tokens: AbstractArray,
    encoder_hidden_states: AbstractArray,
    vocab_size: int,
    num_layers: int = 6,
    num_heads: int = 8,
    hidden_dim: int = 512,
) -> AbstractArray:
    """Witness teacher-forced decoder tokens becoming vocabulary logits."""
    return AbstractArray()


def witness_recurrent_sequence_model(
    sequence_features: AbstractArray,
    hidden_size: int = 128,
    num_layers: int = 1,
    bidirectional: bool = True,
    return_sequences: bool = True,
    cell_type: str = "gru",
    dropout: float = 0.0,
) -> AbstractArray:
    """Witness recurrent sequence inputs becoming hidden states."""
    return AbstractArray()


def witness_slowfast_video_network(
    frames: list[AbstractArray],
    num_classes: int = 400,
    variant: str = "slowfast_r50",
    slowfast_channel_reduction_ratio: int = 8,
) -> AbstractArray:
    """Witness SlowFast dual-pathway video tensors becoming class logits."""
    return AbstractArray()


def witness_mil_attention_aggregator(
    patch_embeddings: AbstractArray,
    num_classes: int,
    attention_hidden_dim: int = 256,
    variant: str = "gated_attention_mil",
) -> AbstractArray:
    """Witness bags of patch embeddings becoming slide-level logits."""
    return AbstractArray()

