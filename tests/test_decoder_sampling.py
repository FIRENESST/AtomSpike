from __future__ import annotations

import pytest
import torch

from atomspike.models.actions import ActionSpec
from atomspike.models.decoder import ActionDecoder


def _spec() -> ActionSpec:
    return ActionSpec(
        n_keys=1,
        n_key_states=4,
        mouse_bins=5,
        n_mouse_buttons=0,
        key_names=("W",),
        n_slots=3,
    )


@pytest.mark.parametrize("mode", ["gru_ar", "transformer_ar"])
def test_sampled_autoregressive_logits_match_teacher_forcing(mode: str) -> None:
    """The logits stored for learning must describe the executed prefix."""
    torch.manual_seed(7)
    decoder = ActionDecoder(8, _spec(), mode=mode).eval()
    context = torch.randn(3, 8)

    with torch.no_grad():
        sampled_logits, tokens = decoder.sample(
            context,
            temperature=0.7,
            deterministic=False,
        )
        replay_logits = decoder(context, tokens)

    assert tokens.shape == (3, 3)
    for slot, (sampled, replay) in enumerate(zip(sampled_logits, replay_logits)):
        assert int(tokens[:, slot].min()) >= 0
        assert int(tokens[:, slot].max()) < _spec().slot_vocabs[slot]
        torch.testing.assert_close(sampled, replay, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("mode", ["parallel", "gru_ar", "transformer_ar"])
def test_deterministic_sample_is_decoder_argmax(mode: str) -> None:
    torch.manual_seed(11)
    decoder = ActionDecoder(8, _spec(), mode=mode).eval()
    context = torch.randn(2, 8)

    with torch.no_grad():
        logits, tokens = decoder.sample(context, deterministic=True)

    expected = torch.stack([x.argmax(dim=-1) for x in logits], dim=-1)
    torch.testing.assert_close(tokens, expected)
