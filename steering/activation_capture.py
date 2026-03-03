import torch
import torch.nn as nn


class ActivationCollector:
    def __init__(self):
        self.activations = []

    def record(self, out: torch.Tensor):
        self.activations.append(out.detach().cpu())

    def clear(self):
        self.activations = []


class CaptureActivationWrapper(nn.Module):
    def __init__(self, original_layer: nn.Module, collector: ActivationCollector):
        super().__init__()
        self.original_layer = original_layer
        self.collector = collector

    def forward(self, *args, **kwargs):
        output = self.original_layer(*args, **kwargs)
        self.collector.record(output)
        return output


def wrap_layers(model: nn.Module, steer_location: str, collector: ActivationCollector):
    for b in model.block_chunks:
        for m in b.module:
            if steer_location == "ffn":
                m.ffn = CaptureActivationWrapper(m.ffn, collector)
            elif steer_location == "ca":
                m.ca = CaptureActivationWrapper(m.ca, collector)
            elif steer_location == "ca_block":
                m = CaptureActivationWrapper(m, collector)
            torch.cuda.empty_cache()


def unwrap_layers(model: nn.Module):
    for b in model.block_chunks:
        for m in b.module:
            if isinstance(m.ffn, CaptureActivationWrapper):
                m.ffn = m.ffn.original_layer
            if isinstance(m.ca, CaptureActivationWrapper):
                m.ca = m.ca.original_layer
            if isinstance(m, CaptureActivationWrapper):
                m = m.original_layer
            torch.cuda.empty_cache()
