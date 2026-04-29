from __future__ import annotations

import numpy as np
import pytest
import icontract


def test_architecture_backbone_shapes() -> None:
    from sciona.atoms.dl.architectures.atoms import (
        densenet_backbone,
        efficientnet_backbone,
        resnet_family_backbone,
        swin_transformer_backbone,
    )

    images = np.zeros((2, 3, 224, 224), dtype=np.float32)
    assert efficientnet_backbone(images, variant="b4").shape == (2, 1792)
    assert resnet_family_backbone(images, variant="resnet18").shape == (2, 512)
    assert densenet_backbone(images, variant="densenet161").shape == (2, 2208)
    assert swin_transformer_backbone(images).shape == (2, 768)


def test_architecture_dense_prediction_shapes() -> None:
    from sciona.atoms.dl.architectures.atoms import unet_1d_sequence, unet_2d_segmentation

    images = np.zeros((4, 3, 512, 512), dtype=np.float32)
    sequence = np.zeros((8, 6, 3008), dtype=np.float32)
    assert unet_2d_segmentation(images, num_classes=5).shape == (4, 5, 512, 512)
    assert unet_1d_sequence(sequence, num_classes=3).shape == (8, 3, 3008)


def test_architecture_detector_and_sequence_shapes() -> None:
    from sciona.atoms.dl.architectures.atoms import (
        autoregressive_transformer_decoder,
        recurrent_sequence_model,
        whisper_asr_transformer,
        yolo_object_detector,
    )

    images = np.zeros((1, 3, 640, 640), dtype=np.float32)
    assert yolo_object_detector(images, num_classes=80).shape == (1, 84, 8400)

    spectrograms = np.zeros((1, 128, 3000), dtype=np.float32)
    decoder_ids = np.zeros((1, 5), dtype=np.int64)
    assert whisper_asr_transformer(spectrograms, decoder_ids).shape == (1, 5, 51865)

    tokens = np.zeros((2, 25), dtype=np.int64)
    encoder_states = np.zeros((2, 196, 768), dtype=np.float32)
    assert autoregressive_transformer_decoder(tokens, encoder_states, vocab_size=275).shape == (2, 25, 275)

    features = np.zeros((4, 107, 14), dtype=np.float32)
    assert recurrent_sequence_model(features, hidden_size=128, bidirectional=True).shape == (4, 107, 256)
    assert recurrent_sequence_model(features, hidden_size=64, bidirectional=False, return_sequences=False).shape == (4, 64)


def test_architecture_video_and_mil_shapes() -> None:
    from sciona.atoms.dl.architectures.atoms import mil_attention_aggregator, slowfast_video_network

    slow = np.zeros((1, 3, 8, 224, 224), dtype=np.float32)
    fast = np.zeros((1, 3, 64, 224, 224), dtype=np.float32)
    assert slowfast_video_network([slow, fast], num_classes=400).shape == (1, 400)

    patch_embeddings = np.zeros((2, 1000, 512), dtype=np.float32)
    assert mil_attention_aggregator(patch_embeddings, num_classes=6).shape == (2, 6)


def test_architecture_contracts_reject_bad_shapes() -> None:
    from sciona.atoms.dl.architectures.atoms import efficientnet_backbone, whisper_asr_transformer

    bad_images = np.zeros((1, 3, 225, 224), dtype=np.float32)
    with pytest.raises(icontract.ViolationError):
        efficientnet_backbone(bad_images)

    bad_mels = np.zeros((1, 80, 3000), dtype=np.float32)
    decoder_ids = np.zeros((1, 5), dtype=np.int64)
    with pytest.raises(icontract.ViolationError):
        whisper_asr_transformer(bad_mels, decoder_ids, variant="large-v3")
