import torch
from transformers import MimiModel


class MimiCodec:
    """Frozen Mimi: waveform <-> (32, T) integer codes. 24 kHz mono in/out."""

    def __init__(self, model_id: str = "kyutai/mimi", device: str = "cpu"):
        self.device = device
        self.model = MimiModel.from_pretrained(model_id).to(device).eval()

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (samples,) float in [-1, 1] at 24 kHz -> (32, frames) long."""
        x = wav.to(self.device).reshape(1, 1, -1)
        codes = self.model.encode(x).audio_codes  # (1, 32, T)
        return codes[0].cpu().long()

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (32, frames) long -> (samples,) float at 24 kHz."""
        x = codes.to(self.device).unsqueeze(0)    # (1, 32, T)
        wav = self.model.decode(x).audio_values   # (1, 1, samples)
        return wav[0, 0].cpu()
