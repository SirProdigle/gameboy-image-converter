"""Reproducible pipeline benchmark. Run: venv/bin/python scripts/bench_pipeline.py"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from PIL import Image
import gb_pipeline

rng = np.random.RandomState(0)

def photo(w, h):
    yy, xx = np.mgrid[0:h, 0:w]
    r = xx / w * 255
    g = yy / h * 255
    b = (np.sin(xx / 9.0) + np.cos(yy / 7.0)) * 60 + 120
    arr = np.stack([r, g, b], axis=2) + rng.normal(0, 14, (h, w, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

if __name__ == "__main__":
    for name, (w, h) in [("160x144", (160, 144)), ("320x288", (320, 288))]:
        img = photo(w, h)
        for preset, kw in [("color_only", dict(tile_budget=192, dither="bayer")),
                           ("mono", dict(dither="bayer"))]:
            t0 = time.perf_counter()
            res = gb_pipeline.convert_for_hardware(img, preset, **kw)
            dt = time.perf_counter() - t0
            print(f"{name} {preset:11s} {dt:6.2f}s  tiles={res.stats['tiles_used']}"
                  f" merges={res.stats['n_merges']} p95={res.stats['p95_delta_e']:.1f}")
