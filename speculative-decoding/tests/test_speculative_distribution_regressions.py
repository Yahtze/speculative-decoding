"""Exercise actual proposal and rejection paths on CPU and available MPS."""

import pytest
import torch

from src.decoding import SpeculativeDecoder
from .decoding_fixtures import ToyLanguageModel


def _devices():
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    return devices


@pytest.mark.parametrize("device", _devices(), ids=lambda device: str(device))
@pytest.mark.parametrize("temperature", [1.0, 1e-30])
def test_filtered_distribution_matches_residual_proposal_law(device, temperature, monkeypatch):
    draft = ToyLanguageModel([0.4, 0.4, 0.15, 0.05])
    target = ToyLanguageModel([0.1, 0.1, 0.3, 0.5])
    for model in (draft, target):
        model.probabilities = model.probabilities.to(device=device, dtype=torch.float32)
    decoder = SpeculativeDecoder(draft, target, k=1, top_k=1, temperature=temperature)
    input_ids = torch.tensor([[0]], device=device)
    sampled_distributions = []

    def capture_sample(probs, num_samples):
        assert num_samples == 1
        sampled_distributions.append(probs.clone())
        return probs.argmax(dim=-1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", capture_sample)
    _, token_prob = decoder._top_k_sample_with_probs(draft, input_ids)
    decoder._resample_rejected(input_ids, [], None)
    proposal, residual = sampled_distributions

    # The proposal threshold must retain both tokens tied at the top-k boundary.
    expected_proposal = torch.tensor([[0.5, 0.5, 0, 0]], device=device)
    expected_residual = torch.tensor(
        [[0, 0, 0.375, 0.625]] if temperature == 1.0 else [[0, 0, 0, 1]],
        device=device, dtype=torch.float32,
    )
    assert proposal.dtype == residual.dtype == torch.float32
    assert proposal.device.type == residual.device.type == device.type
    assert token_prob.item() == pytest.approx(0.5)
    assert torch.isfinite(proposal).all() and torch.isfinite(residual).all()
    assert torch.allclose(proposal, expected_proposal)
    assert torch.allclose(residual, expected_residual)
