"""Tests for PyTorch GPU port atoms.

All tests are skipped if torch is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


# ---- miss_penalty_loss_torch ------------------------------------------------


class TestMissPenaltyLossTorch:
    def test_no_misses_returns_zero(self):
        from sciona.atoms.dl.loss.atoms_torch import miss_penalty_loss_torch

        preds = torch.tensor([0.9, 0.8, 0.7])
        labels = torch.tensor([1.0, 1.0, 1.0])
        loss = miss_penalty_loss_torch(preds, labels, threshold=0.03)
        assert loss.item() == pytest.approx(0.0)

    def test_all_misses(self):
        from sciona.atoms.dl.loss.atoms_torch import miss_penalty_loss_torch

        preds = torch.tensor([0.01, 0.02])
        labels = torch.tensor([1.0, 1.0])
        loss = miss_penalty_loss_torch(preds, labels, threshold=0.03)
        assert loss.item() > 0.0

    def test_matches_numpy(self):
        from sciona.atoms.dl.loss.atoms import miss_penalty_loss
        from sciona.atoms.dl.loss.atoms_torch import miss_penalty_loss_torch

        preds_np = np.array([0.01, 0.5, 0.02, 0.9])
        labels_np = np.array([1.0, 1.0, 1.0, 0.0])
        np_loss = miss_penalty_loss(preds_np, labels_np, threshold=0.03)

        preds_t = torch.tensor(preds_np)
        labels_t = torch.tensor(labels_np)
        torch_loss = miss_penalty_loss_torch(preds_t, labels_t, threshold=0.03)
        assert torch_loss.item() == pytest.approx(np_loss, rel=1e-5)

    def test_gradient_flows(self):
        from sciona.atoms.dl.loss.atoms_torch import miss_penalty_loss_torch

        preds = torch.tensor([0.01, 0.02], requires_grad=True)
        labels = torch.tensor([1.0, 1.0])
        loss = miss_penalty_loss_torch(preds, labels, threshold=0.03)
        loss.backward()
        assert preds.grad is not None
        assert preds.grad.shape == preds.shape


# ---- auxiliary_logit_loss_fusion_torch ---------------------------------------


class TestAuxiliaryLogitLossFusionTorch:
    def test_main_only(self):
        from sciona.atoms.dl.adversarial.atoms_torch import auxiliary_logit_loss_fusion_torch

        logits = torch.tensor([[2.0, 1.0, 0.1]])
        labels = torch.tensor([[1.0, 0.0, 0.0]])
        loss = auxiliary_logit_loss_fusion_torch(logits, labels)
        assert loss.item() > 0.0

    def test_with_aux(self):
        from sciona.atoms.dl.adversarial.atoms_torch import auxiliary_logit_loss_fusion_torch

        logits = torch.tensor([[2.0, 1.0, 0.1]])
        labels = torch.tensor([[1.0, 0.0, 0.0]])
        aux = torch.tensor([[1.5, 0.5, 0.1]])
        loss_no_aux = auxiliary_logit_loss_fusion_torch(logits, labels)
        loss_with_aux = auxiliary_logit_loss_fusion_torch(logits, labels, aux_logits=aux)
        assert loss_with_aux.item() > loss_no_aux.item()

    def test_matches_numpy(self):
        from sciona.atoms.dl.adversarial.atoms import auxiliary_logit_loss_fusion
        from sciona.atoms.dl.adversarial.atoms_torch import auxiliary_logit_loss_fusion_torch

        logits_np = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
        labels_np = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        aux_np = np.array([[1.5, 0.5, 0.1], [0.3, 1.8, 0.2]])

        np_loss = auxiliary_logit_loss_fusion(logits_np, labels_np, aux_np, aux_weight=0.4)

        logits_t = torch.tensor(logits_np)
        labels_t = torch.tensor(labels_np)
        aux_t = torch.tensor(aux_np)
        torch_loss = auxiliary_logit_loss_fusion_torch(logits_t, labels_t, aux_t, aux_weight=0.4)
        assert torch_loss.item() == pytest.approx(np_loss, rel=1e-5)


# ---- online_hard_negative_mining_torch ---------------------------------------


class TestOnlineHardNegativeMiningTorch:
    def test_basic(self):
        from sciona.atoms.dl.training.atoms_torch import online_hard_negative_mining_torch

        scores = torch.tensor([0.1, 0.9, 0.5, 0.3])
        indices = online_hard_negative_mining_torch(scores, num_hard=2)
        assert indices.shape == (2,)
        assert set(indices.tolist()) == {1, 2}

    def test_num_hard_exceeds_length(self):
        from sciona.atoms.dl.training.atoms_torch import online_hard_negative_mining_torch

        scores = torch.tensor([0.1, 0.9])
        indices = online_hard_negative_mining_torch(scores, num_hard=10)
        assert indices.shape == (2,)

    def test_matches_numpy_top_indices(self):
        from sciona.atoms.dl.training.atoms import online_hard_negative_mining
        from sciona.atoms.dl.training.atoms_torch import online_hard_negative_mining_torch

        scores_np = np.array([0.1, 0.9, 0.5, 0.3, 0.7])
        np_indices = online_hard_negative_mining(scores_np, num_hard=3)

        scores_t = torch.tensor(scores_np)
        torch_indices = online_hard_negative_mining_torch(scores_t, num_hard=3)
        assert set(torch_indices.tolist()) == set(np_indices.tolist())


# ---- softmax_temperature_proposal_sampling_torch -----------------------------


class TestSoftmaxTemperatureProposalSamplingTorch:
    def test_output_shape(self):
        from sciona.atoms.dl.training.atoms_torch import softmax_temperature_proposal_sampling_torch

        scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        indices = softmax_temperature_proposal_sampling_torch(scores, k=3)
        assert indices.shape == (3,)

    def test_no_replacement(self):
        from sciona.atoms.dl.training.atoms_torch import softmax_temperature_proposal_sampling_torch

        scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        indices = softmax_temperature_proposal_sampling_torch(scores, k=5)
        assert len(set(indices.tolist())) == 5

    def test_temperature_effect(self):
        """High temperature should produce more uniform sampling."""
        from sciona.atoms.dl.training.atoms_torch import softmax_temperature_proposal_sampling_torch

        scores = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0])
        torch.manual_seed(42)
        # Low temperature: nearly always picks index 0
        low_temp_picks = []
        for _ in range(50):
            idx = softmax_temperature_proposal_sampling_torch(scores, k=1, temperature=0.1)
            low_temp_picks.append(idx.item())
        assert low_temp_picks.count(0) > 40  # almost always picks the top score


# ---- center_feature_extraction_3d_torch --------------------------------------


class TestCenterFeatureExtraction3dTorch:
    def test_output_shape(self):
        from sciona.atoms.dl.detection.atoms_torch import center_feature_extraction_3d_torch

        fm = torch.randn(2, 8, 4, 6, 6)
        result = center_feature_extraction_3d_torch(fm)
        assert result.shape == (2, 8)

    def test_matches_numpy(self):
        from sciona.atoms.dl.detection.atoms import center_feature_extraction_3d
        from sciona.atoms.dl.detection.atoms_torch import center_feature_extraction_3d_torch

        np.random.seed(42)
        fm_np = np.random.randn(2, 4, 6, 8, 8).astype(np.float64)
        np_result = center_feature_extraction_3d(fm_np)

        fm_t = torch.tensor(fm_np)
        torch_result = center_feature_extraction_3d_torch(fm_t)
        np.testing.assert_allclose(torch_result.numpy(), np_result, rtol=1e-5)

    def test_minimal_spatial_dims(self):
        from sciona.atoms.dl.detection.atoms_torch import center_feature_extraction_3d_torch

        fm = torch.randn(1, 3, 2, 2, 2)
        result = center_feature_extraction_3d_torch(fm)
        assert result.shape == (1, 3)
