import threading
import time
from types import SimpleNamespace

import torch
import torch.nn as nn

class DummyBlock(nn.Module):
    def forward(self, x):
        return x + 1

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([DummyBlock() for _ in range(12)])
        self.patch_embed = SimpleNamespace(patch_size=16, proj=SimpleNamespace(stride=(16, 16)))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class BuggyExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DummyModel()
        self._feats = []
        self.hook_handlers = []

    def _get_hook(self):
        def _hook(m, i, o):
            self._feats.append(o)
            time.sleep(0.001)
        return _hook

    def _register_hooks(self, layers):
        for idx, block in enumerate(self.model.blocks):
            if idx in layers:
                self.hook_handlers.append(block.register_forward_hook(self._get_hook()))

    def _unregister_hooks(self):
        for h in self.hook_handlers:
            h.remove()
        self.hook_handlers = []

    def extract_descriptors(self, batch, layer=11):
        self._feats = []
        self._register_hooks([layer])
        _ = self.model(batch)
        self._unregister_hooks()
        x = torch.stack(self._feats, dim=1).unsqueeze(2)
        return x.permute(0, 1, 3, 4, 2).flatten(start_dim=-2, end_dim=-1).unsqueeze(2)

class FixedExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DummyModel()
        self.tl = threading.local()
        self._setup_hooks()

    def _setup_hooks(self):
        def make_hook(idx):
            def _hook(m, i, o):
                tl = self.tl
                if getattr(tl, "active", False) and idx == tl.layer:
                    tl.feats.append(o)
                    time.sleep(0.001)
            return _hook
        for idx, block in enumerate(self.model.blocks):
            block.register_forward_hook(make_hook(idx))

    def extract_descriptors(self, batch, layer=11):
        tl = self.tl
        tl.active = True
        tl.layer = layer
        tl.feats = []
        try:
            _ = self.model(batch)
        finally:
            tl.active = False
        x = torch.stack(tl.feats, dim=1).unsqueeze(2)
        return x.permute(0, 1, 3, 4, 2).flatten(start_dim=-2, end_dim=-1).unsqueeze(2)

class SimpleEmbeddingModel(nn.Module):
    def __init__(self, extractor_cls):
        super().__init__()
        self.extractors = nn.ModuleList([extractor_cls(), extractor_cls()])

    def embed(self, img):
        full = self.extractors[0].extract_descriptors(img).squeeze()
        for ext in self.extractors[1:]:
            feats = ext.extract_descriptors(img).squeeze()
            full = torch.cat((full, feats), dim=-1)
        return full

def _run_threads(model):
    img = torch.zeros(1, 1, 4)
    barrier = threading.Barrier(3)
    errors = []

    def worker():
        try:
            barrier.wait()
            for _ in range(20):
                model.embed(img)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    barrier.wait()
    for t in threads:
        t.join()
    return errors

def test_buggy_extractor_threads_fail():
    model = SimpleEmbeddingModel(BuggyExtractor)
    errors = _run_threads(model)
    assert errors, 'Expected error with buggy extractor'

def test_fixed_extractor_threads_ok():
    model = SimpleEmbeddingModel(FixedExtractor)
    errors = _run_threads(model)
    assert not errors, f'Unexpected errors: {errors}'
