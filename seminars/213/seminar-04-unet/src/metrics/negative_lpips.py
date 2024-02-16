from lpips import LPIPS


class NegativeLPIPS(LPIPS):
    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        return -out.mean()
