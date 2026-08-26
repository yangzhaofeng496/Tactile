import torch

from model import FzAwareTactileEncoder, fz_auxiliary_loss


def test_fz_encoder_outputs_and_auxiliary_loss():
    model = FzAwareTactileEncoder()
    history = torch.randn(4, 16, 12)
    output = model(history)
    assert output["feature"].shape == (4, 88)
    assert output["contact_logits"].shape == (4, 4)
    assert output["fz_pred"].shape == (4, 2)
    loss, contact_loss, magnitude_loss = fz_auxiliary_loss(output, history, 0.5)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(contact_loss) and torch.isfinite(magnitude_loss)


def test_contact_labels_have_four_states():
    history = torch.zeros(4, 16, 12)
    history[1, :, 2] = 2.0
    history[2, :, 8] = 2.0
    history[3, :, 2] = 2.0
    history[3, :, 8] = 2.0
    labels = FzAwareTactileEncoder.make_contact_labels(history, 1.0)
    assert labels.tolist() == [0, 1, 2, 3]
