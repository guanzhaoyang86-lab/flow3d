from __future__ import annotations

import pytest
import torch

from flow_observation.diffusion import GaussianDiffusion
from flow_observation.models.dit import ConditionalDiT3D, TensorDiT, patchify_3d, unpatchify_3d


@pytest.mark.parametrize("shape,patch", [((8, 12, 4), 2), ((8, 8, 8), 4), ((3, 4, 5), 1)])
def test_volume_patch_round_trip_keeps_every_voxel_and_gradient(shape, patch) -> None:
    values = torch.arange(2 * 3 * shape[0] * shape[1] * shape[2], dtype=torch.float64).reshape(2, 3, *shape).requires_grad_()
    tokens = patchify_3d(values, patch)
    # The first token contains all channel values in the first spatial cube.
    torch.testing.assert_close(tokens[0, 0], values[0, :, :patch, :patch, :patch].reshape(-1))
    reconstructed = unpatchify_3d(tokens, shape, patch)
    torch.testing.assert_close(reconstructed, values, rtol=0, atol=0)
    reconstructed.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values), rtol=0, atol=0)


@pytest.mark.parametrize("rank", [1, 3, 4, 7, 8, 12, 13, 16])
def test_tucker_tokenization_keeps_all_core_and_factor_entries(rank) -> None:
    model = TensorDiT((16, 20, 24), rank=rank, hidden_dim=12, num_heads=3, depth=1, condition_dim=5)
    values = torch.arange(2 * model.packed_dim, dtype=torch.float32).reshape(2, -1).requires_grad_()
    tokens = model.split_tokens(values)
    assert tokens[0].shape[1] <= 512
    assert model.packed_dim == 3 * rank**3 + (16 + 20 + 24) * rank
    assert model.num_tokens == sum(piece.shape[1] for piece in tokens)
    torch.testing.assert_close(tokens[1][0], values[0, model.core_size:model.core_size + 16 * rank].reshape(16, rank))
    recovered = model.merge_tokens(tokens)
    torch.testing.assert_close(recovered, values, atol=0, rtol=0)
    recovered.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values), rtol=0, atol=0)


def _small_model(kind: str):
    options = dict(hidden_dim=24, num_heads=3, depth=2, condition_dim=5)
    if kind == "volume":
        return ConditionalDiT3D((4, 8, 4), patch_size=2, **options), (2, 3, 4, 8, 4)
    model = TensorDiT((4, 6, 8), rank=3, **options)
    return model, (2, model.packed_dim)


@pytest.mark.parametrize("kind", ["volume", "tucker"])
def test_zero_initialized_denoiser_learns_and_uses_condition_and_time(kind) -> None:
    torch.manual_seed(17)
    model, shape = _small_model(kind)
    values = torch.randn(shape)
    steps = torch.tensor([1, 7])
    conditions = torch.randn(2, 5)
    target = torch.randn(shape)
    prediction = model(values, steps, conditions)
    torch.testing.assert_close(prediction, torch.zeros_like(values), rtol=0, atol=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(values, steps, conditions) - target).square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        optimizer.step()
    model.eval()
    baseline = model(values, steps, conditions)
    assert baseline.shape == values.shape
    assert torch.isfinite(baseline).all()
    assert not torch.allclose(baseline, model(values, steps, conditions + 2))
    assert not torch.allclose(baseline, model(values, steps + 10, conditions))
    assert not torch.allclose(baseline, model(values + 1, steps, conditions))
    torch.testing.assert_close(model(values, steps, None), model(values, steps, torch.zeros_like(conditions)))
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.condition_projection.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.blocks[0].attention.parameters())


@pytest.mark.parametrize("kind", ["volume", "tucker"])
def test_float64_and_state_dict_round_trip(kind) -> None:
    torch.manual_seed(29)
    model, shape = _small_model(kind)
    model = model.double().eval()
    # Nonzero heads exercise the full forward path, rather than comparing zeros.
    heads = [model.output_projection] if kind == "volume" else model.output_projections
    for head in heads:
        torch.nn.init.normal_(head.weight, std=0.02)
    values = torch.randn(shape, dtype=torch.float64)
    conditions = torch.randn(2, 5, dtype=torch.float64)
    steps = torch.tensor([2, 3])
    expected = model(values, steps, conditions)
    other, _ = _small_model(kind)
    other = other.double().eval()
    other.load_state_dict(model.state_dict())
    actual = other(values, steps, conditions)
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["volume", "tucker"])
def test_validation_rejects_incompatible_inputs(kind) -> None:
    model, shape = _small_model(kind)
    values = torch.zeros(shape)
    with pytest.raises(ValueError, match="shape"):
        model(values[..., :-1], torch.ones(2), None)
    with pytest.raises(TypeError, match="floating"):
        model(values.long(), torch.ones(2), None)
    with pytest.raises(ValueError, match="timesteps"):
        model(values, torch.ones(2, 1), None)
    with pytest.raises(ValueError, match="non-negative"):
        model(values, torch.tensor([-1, 0]), None)
    with pytest.raises(ValueError, match="finite"):
        model(values, torch.tensor([0, float("nan")]), None)
    with pytest.raises(ValueError, match="condition"):
        model(values, torch.zeros(2), torch.ones(2, 6))
    with pytest.raises(ValueError, match="non-empty"):
        model(values[:0], torch.ones(0), None)


def test_configuration_validation_and_expected_token_counts() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ConditionalDiT3D((8, 9, 8), patch_size=4)
    with pytest.raises(ValueError, match="divisible"):
        ConditionalDiT3D(hidden_dim=32, num_heads=6)
    with pytest.raises(ValueError, match="rank"):
        TensorDiT((4, 4, 4), rank=5)
    with pytest.raises(ValueError, match="rank"):
        TensorDiT(rank=1.5)
    with pytest.raises(ValueError, match="positive integers"):
        TensorDiT((32, 32, 0))
    assert ConditionalDiT3D(hidden_dim=12, num_heads=3, depth=1).num_tokens == 512
    assert TensorDiT(rank=8, hidden_dim=12, num_heads=3, depth=1).num_tokens == 64 + 96
    assert TensorDiT(rank=16, hidden_dim=12, num_heads=3, depth=1).num_tokens == 512 + 96


@pytest.mark.parametrize("kind", ["volume", "tucker"])
def test_diffusion_training_autocast_and_cfg_sampling(kind) -> None:
    torch.manual_seed(91)
    model, shape = _small_model(kind)
    diffusion = GaussianDiffusion(8, clip_x0=4.0)
    clean = torch.randn(shape)
    condition = torch.randn(2, 5, requires_grad=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = diffusion.training_loss(model, clean, condition)
        result["loss"].backward()
        assert torch.isfinite(result["loss"])
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        optimizer.step()
    model.eval()
    noise = torch.randn(shape)
    samples = []
    for _ in range(2):
        samples.append(diffusion.ddim_sample(model, shape, condition.detach(), sampling_steps=3, eta=0, guidance_scale=1.5, initial_noise=noise))
    assert samples[0].shape == clean.shape
    assert torch.isfinite(samples[0]).all()
    torch.testing.assert_close(samples[0], samples[1], atol=0, rtol=0)
