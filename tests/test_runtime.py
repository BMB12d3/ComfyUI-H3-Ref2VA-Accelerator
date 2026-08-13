"""Headless test suite for the H3 Ref2VA Accelerator runtime.

Runs without ComfyUI or a GPU: `comfy` is stubbed, tensors are tiny CPU tensors,
and the block patches are driven the same way ComfyUI's dit patches_replace
mechanism drives them (args dict with "img", extra dict with "original_block").

Usage:  python tests/test_runtime.py
"""

from __future__ import annotations

import importlib.util
import logging
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Stub comfy before importing nodes.py
# ---------------------------------------------------------------------------

comfy_pkg = types.ModuleType("comfy")
model_prefetch = types.ModuleType("comfy.model_prefetch")


def _stub_make_prefetch_queue(queue, device, transformer_options):
    return list(queue)


model_prefetch.make_prefetch_queue = _stub_make_prefetch_queue

patcher_extension = types.ModuleType("comfy.patcher_extension")


class WrappersMP:
    DIFFUSION_MODEL = "diffusion_model"
    OUTER_SAMPLE = "outer_sample"


patcher_extension.WrappersMP = WrappersMP

comfy_pkg.model_prefetch = model_prefetch
comfy_pkg.patcher_extension = patcher_extension
sys.modules["comfy"] = comfy_pkg
sys.modules["comfy.model_prefetch"] = model_prefetch
sys.modules["comfy.patcher_extension"] = patcher_extension

import torch  # noqa: E402

_NODES_PATH = Path(__file__).resolve().parent.parent / "nodes.py"
_spec = importlib.util.spec_from_file_location("h3_ref2va_nodes", _NODES_PATH)
nodes = importlib.util.module_from_spec(_spec)
sys.modules["h3_ref2va_nodes"] = nodes
_spec.loader.exec_module(nodes)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ROWS, DIM = 16, 6
SEGMENTS = [(0, 8, "video"), (8, 12, "audio"), (12, 14, "ref_img"), (14, 16, "ref_audio")]


def make_layout(latent_frames=4):
    return SimpleNamespace(segments=list(SEGMENTS), signature=(1, latent_frames))


def payload(layout=None):
    return {"refs": True, "layout": layout if layout is not None else make_layout()}


def const_blocks(seed=0, dtype=torch.float32):
    """Blocks that add a fixed per-block delta in place (input-independent tail)."""
    g = torch.Generator().manual_seed(seed)
    deltas = [torch.randn(ROWS, DIM, generator=g, dtype=torch.float32).to(dtype) for _ in range(4)]

    def make(i):
        def block(args):
            args["img"].add_(deltas[i])
            return {"img": args["img"]}
        return block

    return [make(i) for i in range(4)], deltas


def doubling_block0_blocks(seed=0):
    """Block 0 doubles the hidden state (residual == original input); blocks 1-3 add constants."""
    g = torch.Generator().manual_seed(seed)
    deltas = [torch.randn(ROWS, DIM, generator=g) for _ in range(4)]

    def block0(args):
        args["img"].add_(args["img"])  # x -> 2x, residual == x
        return {"img": args["img"]}

    def make(i):
        def block(args):
            args["img"].add_(deltas[i])
            return {"img": args["img"]}
        return block

    return [block0, make(1), make(2), make(3)], deltas


def make_runtime(config=None, storage=None, start_sigma=10.0, end_sigma=0.0,
                 blocks=None, debug=False, tail_rescale=False,
                 cpu_tail_compute=None):
    config = config or nodes.PRESETS[nodes.BALANCED]
    storage = storage or nodes.CPU_STORAGE
    blocks = blocks if blocks is not None else [object() for _ in range(4)]
    return nodes.Ref2VAUltraSafeBlockCacheRuntime(
        config=config, start_sigma=start_sigma, end_sigma=end_sigma,
        block_count=4, storage=storage, debug=debug,
        block_modules=blocks, tail_rescale=tail_rescale,
        cpu_tail_compute=cpu_tail_compute or nodes.CPU_TAIL_SAFE,
    )


def native(blocks, x):
    img = x.clone()
    for b in blocks:
        img = b({"img": img})["img"]
    return img


def run_step(runtime, patches, blocks, x, sigma, pl):
    """Drive one model call through the diffusion wrapper + block patches,
    mirroring ComfyUI's calling convention."""
    dwrap = nodes.make_diffusion_wrapper(runtime)
    timestep = torch.tensor([sigma * 1000.0])
    topts = {"uuids": ("u1",), "sigmas": torch.tensor([sigma])}

    def executor(x_in, t_in, control, transformer_options, minimax_payload=None):
        img = x_in.clone()
        for i, p in enumerate(patches):
            img = p({"img": img}, {"original_block": blocks[i]})["img"]
        return img

    return dwrap(executor, x, timestep, None, topts, pl)


def build(runtime):
    return [nodes.make_block_patch(runtime, i, 3) for i in range(4)]


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())

    def __enter__(self):
        logging.getLogger().addHandler(self)
        return self

    def __exit__(self, *exc):
        logging.getLogger().removeHandler(self)
        return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_exact_cache_reconstruction_both_storages():
    """With input-independent blocks 1-3, a cached step must equal native bit-for-bit."""
    for storage in (nodes.CPU_STORAGE, nodes.GPU_STORAGE):
        blocks, _ = const_blocks()
        rt = make_runtime(storage=storage, blocks=blocks)
        patches = build(rt)
        g = torch.Generator().manual_seed(7)
        x1 = torch.randn(ROWS, DIM, generator=g)
        x2 = torch.randn(ROWS, DIM, generator=g)

        out1 = run_step(rt, patches, blocks, x1, 0.9, payload())
        assert torch.equal(out1, native(blocks, x1)), "full step must match native"
        assert rt.full_steps == 1 and rt.cached_steps == 0

        out2 = run_step(rt, patches, blocks, x2, 0.8, payload())
        assert rt.cached_steps == 1, f"expected a cache hit ({storage})"
        # Exact in exact arithmetic; float32 add-order differs, so compare to rounding.
        ref = native(blocks, x2)
        assert torch.allclose(out2, ref, atol=1e-5, rtol=1e-5), (
            f"cached step diverged ({storage}): max abs diff "
            f"{(out2 - ref).abs().max().item():.3e}")


def test_max_consecutive_hits_enforced():
    blocks, _ = const_blocks()
    rt = make_runtime(blocks=blocks)
    patches = build(rt)
    x = torch.randn(ROWS, DIM)
    sigmas = [0.9, 0.8, 0.7, 0.6]
    for s in sigmas:
        run_step(rt, patches, blocks, x.clone(), s, payload())
    # zero-diff every step: hit pattern must alternate full/cache/full/cache
    assert rt.full_steps == 2 and rt.cached_steps == 2
    assert rt.cache_step_numbers == [2, 4]


def test_window_gating():
    blocks, _ = const_blocks()
    rt = make_runtime(blocks=blocks, start_sigma=0.5, end_sigma=0.2)
    patches = build(rt)
    x = torch.randn(ROWS, DIM)
    run_step(rt, patches, blocks, x.clone(), 0.9, payload())
    run_step(rt, patches, blocks, x.clone(), 0.8, payload())  # outside window
    assert rt.cached_steps == 0, "no hits outside the sigma window"
    run_step(rt, patches, blocks, x.clone(), 0.4, payload())  # inside window
    assert rt.cached_steps == 1


def test_sigma_increase_resets_context():
    blocks, _ = const_blocks()
    rt = make_runtime(blocks=blocks)
    patches = build(rt)
    x = torch.randn(ROWS, DIM)
    run_step(rt, patches, blocks, x.clone(), 0.9, payload())
    run_step(rt, patches, blocks, x.clone(), 0.8, payload())
    assert rt.cached_steps == 1
    run_step(rt, patches, blocks, x.clone(), 0.95, payload())  # new run: sigma went up
    assert rt.cached_steps == 1, "reset step must run full"
    assert rt.full_steps == 2


def test_input_signature_change_resets_context():
    blocks, _ = const_blocks()
    rt = make_runtime(blocks=blocks)
    patches = build(rt)
    run_step(rt, patches, blocks, torch.randn(ROWS, DIM), 0.9, payload())
    # Same declining sigma, different shape: must not attempt to compare/cache.
    blocks2, _ = const_blocks()

    def run_shape(rows):
        dwrap = nodes.make_diffusion_wrapper(rt)
        topts = {"uuids": ("u1",), "sigmas": torch.tensor([0.8])}

        def executor(x_in, t_in, control, transformer_options, minimax_payload=None):
            return x_in.clone()

        dwrap(executor, torch.randn(rows, DIM), torch.tensor([800.0]), None, topts, payload())

    run_shape(ROWS + 4)
    ctx = rt.contexts[("u1",)]
    assert ctx.previous_first_residual is None or ctx.previous_first_residual.shape[0] != ROWS
    assert rt.cached_steps == 0


def test_no_refs_idle_and_summary():
    blocks, _ = const_blocks()
    rt = make_runtime(blocks=blocks)
    patches = build(rt)
    x = torch.randn(ROWS, DIM)
    out = run_step(rt, patches, blocks, x, 0.9, {})  # no payload
    assert torch.equal(out, native(blocks, x)), "idle path must be native"
    assert rt.full_steps == 0 and rt.cached_steps == 0
    assert rt.steps_without_refs == 1
    s = rt.summary("test")
    assert "without a Ref2VA payload" in s, s


def test_mixed_idle_summary_line():
    blocks, _ = const_blocks()
    rt = make_runtime(blocks=blocks)
    patches = build(rt)
    x = torch.randn(ROWS, DIM)
    run_step(rt, patches, blocks, x.clone(), 0.9, {})
    run_step(rt, patches, blocks, x.clone(), 0.8, payload())
    run_step(rt, patches, blocks, x.clone(), 0.7, payload())
    s = rt.summary("test")
    assert "steps without Ref2VA payload: 1" in s, s
    assert f"CPU tail compute: {nodes.CPU_TAIL_SAFE}" in s, s
    assert "fast GPU tail path:" not in s, s  # Safe CPU never probes/uses the GPU fast path.



def test_cpu_tail_safe_skips_gpu_headroom_probe():
    blocks, _ = const_blocks()
    rt = make_runtime(blocks=blocks, storage=nodes.CPU_STORAGE,
                      cpu_tail_compute=nodes.CPU_TAIL_SAFE)
    patches = build(rt)
    old = nodes._gpu_headroom_ok

    def should_not_run(*_a, **_k):
        raise AssertionError("Safe CPU must not probe GPU headroom")

    nodes._gpu_headroom_ok = should_not_run
    try:
        run_step(rt, patches, blocks, torch.randn(ROWS, DIM), 0.9, payload())
    finally:
        nodes._gpu_headroom_ok = old
    assert rt.full_steps == 1
    assert rt.gpu_tail_steps == 0


def test_cpu_tail_auto_consults_gpu_headroom_probe():
    blocks, _ = const_blocks()
    rt = make_runtime(blocks=blocks, storage=nodes.CPU_STORAGE,
                      cpu_tail_compute=nodes.CPU_TAIL_AUTO)
    patches = build(rt)
    old = nodes._gpu_headroom_ok
    called = {"value": False}

    def probe(*_a, **_k):
        called["value"] = True
        return False  # headless CPU test intentionally exercises fallback.

    nodes._gpu_headroom_ok = probe
    try:
        run_step(rt, patches, blocks, torch.randn(ROWS, DIM), 0.9, payload())
    finally:
        nodes._gpu_headroom_ok = old
    assert called["value"], "Auto mode must consult the headroom gate"
    summary = rt.summary("test")
    assert f"CPU tail compute: {nodes.CPU_TAIL_AUTO}" in summary
    assert "fast GPU tail path: 0/1 full steps" in summary


def test_production_widget_defaults():
    spec = nodes.ApplyH3Ref2VAUltraSafeBlockCache.INPUT_TYPES()
    mode_choices, mode_opts = spec["required"]["mode"]
    storage_choices, storage_opts = spec["required"]["cache_storage"]
    tail_choices, tail_opts = spec["optional"]["cpu_tail_compute"]
    _rescale_type, rescale_opts = spec["optional"]["tail_rescale"]

    assert mode_choices[0] == nodes.BALANCED
    assert mode_opts["default"] == nodes.BALANCED
    assert storage_choices[0] == nodes.CPU_STORAGE
    assert storage_opts["default"] == nodes.CPU_STORAGE
    assert tail_choices == [nodes.CPU_TAIL_SAFE, nodes.CPU_TAIL_AUTO]
    assert tail_opts["default"] == nodes.CPU_TAIL_SAFE
    assert tail_opts.get("advanced") is True
    assert rescale_opts["default"] is False
    assert rescale_opts.get("advanced") is True


def test_missing_metric_warning_once():
    blocks, _ = const_blocks()
    rt = make_runtime(blocks=blocks)
    patches = build(rt)
    bad_layout = make_layout(latent_frames=5)  # 8 video rows % 5 != 0 -> temporal None
    x = torch.randn(ROWS, DIM)
    with LogCapture() as cap:
        for s in (0.9, 0.8, 0.7):
            run_step(rt, patches, blocks, x.clone(), s, payload(bad_layout))
    warnings = [m for m in cap.records if "required guard metric(s) unavailable" in m]
    assert len(warnings) == 1, f"expected exactly one warning, got {len(warnings)}"
    assert "temporal" in warnings[0]
    assert rt.cached_steps == 0, "caching must stay disabled when a required metric is missing"


def test_fp32_metric_precision():
    torch.manual_seed(11)
    cur = torch.randn(4096, dtype=torch.float32).to(torch.bfloat16)
    prev = (torch.randn(4096, dtype=torch.float32) * 0.9).to(torch.bfloat16)
    got = nodes._ratio(cur, prev)
    ref = float((cur.double() - prev.double()).abs().mean() / prev.double().abs().mean().clamp(min=1e-8))
    rel = abs(got - ref) / max(abs(ref), 1e-12)
    assert rel < 1e-4, f"global ratio drifted from fp64 reference: rel={rel:.2e}"
    # Fast path and ranged path must now agree (both fp32-accumulated).
    ranged = nodes._ratio(cur, prev, [(0, 4096)])
    assert abs(got - ranged) / max(abs(got), 1e-12) < 1e-4, (got, ranged)


def test_temporal_ratio_math():
    layout = make_layout(latent_frames=4)
    prev = torch.ones(ROWS, DIM)
    cur = prev.clone()
    cur[2:4] += 0.5  # frame 1 of the video segment (rows_per_frame == 2)
    got = nodes._temporal_video_ratio(cur, prev, layout)
    assert got is not None and abs(got - 0.5) < 1e-6, got


def test_tail_rescale_math_and_cache_integrity():
    # Block 0 doubles (residual == input), so per-segment residual energy ratios are controllable.
    for storage in (nodes.CPU_STORAGE, nodes.GPU_STORAGE):
        blocks, deltas = doubling_block0_blocks()
        cfg = nodes.PresetConfig(
            global_threshold=0.5, video_threshold=0.5, audio_threshold=0.5,
            visual_ref_threshold=0.5, audio_ref_threshold=0.5, temporal_threshold=0.5,
            start_percent=0.0, end_percent=1.0, max_consecutive_hits=1,
        )
        rt = make_runtime(config=cfg, storage=storage, blocks=blocks, tail_rescale=True)
        patches = build(rt)

        x1 = torch.rand(ROWS, DIM) + 0.5  # strictly positive for clean energy ratios
        factors = {"video": 1.05, "audio": 1.30, "ref_img": 1.04, "ref_audio": 1.03}
        x2 = x1.clone()
        for a, b, kind in SEGMENTS:
            x2[a:b] *= factors[kind]

        run_step(rt, patches, blocks, x1.clone(), 0.9, payload())
        tail_before = rt.contexts[("u1",)].remaining_blocks_residual.clone()
        out2 = run_step(rt, patches, blocks, x2.clone(), 0.8, payload())
        assert rt.cached_steps == 1, f"expected hit ({storage})"

        expected = x2 * 2  # block-0 output
        tail = sum(deltas[1:])
        clamped = {"video": 1.05, "audio": 1.10, "ref_img": 1.04, "ref_audio": 1.03}  # audio clamps
        for a, b, kind in SEGMENTS:
            expected[a:b] += tail[a:b] * clamped[kind]
        assert torch.allclose(out2, expected, atol=1e-4), f"rescaled output mismatch ({storage})"

        # The persistent cached tail must never be mutated by rescaling.
        assert torch.equal(rt.contexts[("u1",)].remaining_blocks_residual, tail_before), storage


def test_observe_only_never_caches_but_measures():
    blocks, _ = const_blocks()
    rt = make_runtime(config=nodes.PRESETS[nodes.OBSERVE], blocks=blocks)
    patches = build(rt)
    x = torch.randn(ROWS, DIM)
    for s in (0.9, 0.8, 0.7):
        out = run_step(rt, patches, blocks, x.clone(), s, payload())
        assert torch.equal(out, native(blocks, x))
    assert rt.cached_steps == 0
    assert len(rt.metric_history) == 2  # first step has nothing to compare against


def test_prefetch_capture_and_neutralize():
    blocks, _ = const_blocks()
    rt = make_runtime(blocks=blocks)
    q = list(range(4))  # stand-in queue entries
    # Wrong identity: not captured.
    out = rt.capture_prefetch_queue(lambda queue, d, t: list(queue), [object()] * 4, None, {})
    assert rt.prefetch_queue is None
    # Matching identity: captured, then neutralized keeping entry 0.
    out = rt.capture_prefetch_queue(lambda queue, d, t: q, blocks, None, {})
    assert rt.prefetch_queue is q
    rt.neutralize_tail_prefetch()
    assert q[0] == 0 and all(v is None for v in q[1:])
    assert rt.prefetch_suppressed_steps == 1


def test_sample_wrapper_without_prefetch_module():
    blocks, _ = const_blocks()
    rt = make_runtime(blocks=blocks)
    old = nodes._PREFETCH_AVAILABLE
    nodes._PREFETCH_AVAILABLE = False
    try:
        wrap = nodes.make_sample_wrapper(rt, nodes.BALANCED)
        with LogCapture() as cap:
            result = wrap(lambda *a, **k: "sentinel")
        assert result == "sentinel"
        assert any("without tail-prefetch suppression" in m for m in cap.records)
    finally:
        nodes._PREFETCH_AVAILABLE = old


def test_sample_wrapper_restores_prefetch_and_summarizes():
    blocks, _ = const_blocks()
    rt = make_runtime(blocks=blocks)
    original = nodes.comfy.model_prefetch.make_prefetch_queue
    wrap = nodes.make_sample_wrapper(rt, nodes.BALANCED)

    def executor(*a, **k):
        assert nodes.comfy.model_prefetch.make_prefetch_queue is not original, "patched during sampling"
        return "ok"

    with LogCapture() as cap:
        assert wrap(executor) == "ok"
    assert nodes.comfy.model_prefetch.make_prefetch_queue is original, "restored after sampling"
    assert any("Accelerator v" in m and "result" in m for m in cap.records)


def _fake_model(block_count=4, existing_dit=None):
    fake_blocks = [object() for _ in range(block_count)]
    H3 = type("MiniMaxH3Model", (), {})
    dm = H3()
    dm.blocks = fake_blocks
    sampling = SimpleNamespace(percent_to_sigma=lambda p: 1.0 - p)

    class FakePatcher:
        def __init__(self):
            self.model_options = {"transformer_options": {"patches_replace": {"dit": dict(existing_dit or {})}}}
            self.patches = {}
            self.wrappers = {}

        def get_model_object(self, name):
            return {"diffusion_model": dm, "model_sampling": sampling}[name]

        def clone(self):
            return self

        def set_model_patch_replace(self, fn, kind, block_kind, index):
            self.patches[(kind, block_kind, index)] = fn

        def add_wrapper_with_key(self, wrapper_type, key, fn):
            self.wrappers[(wrapper_type, key)] = fn

    return FakePatcher()


def test_apply_plumbing_and_conflict_detection():
    node = nodes.ApplyH3Ref2VAUltraSafeBlockCache()
    m = _fake_model()
    (patched,) = node.apply(m, nodes.BALANCED, nodes.CPU_STORAGE, False, tail_rescale=True)
    assert len(patched.patches) == 4
    kinds = {k[0] for k in patched.wrappers}
    assert kinds == {WrappersMP.DIFFUSION_MODEL, WrappersMP.OUTER_SAMPLE}

    conflicted = _fake_model(existing_dit={("double_block", 1): lambda a, e: a})
    try:
        node.apply(conflicted, nodes.BALANCED, nodes.CPU_STORAGE, False)
        raise AssertionError("expected conflict ValueError")
    except ValueError as e:
        assert "existing DiT block replacement" in str(e)

    class NotH3:
        blocks = [object(), object()]
    bad = _fake_model()
    bad.get_model_object = lambda name: {"diffusion_model": NotH3(), "model_sampling": None}[name]
    try:
        node.apply(bad, nodes.BALANCED, nodes.CPU_STORAGE, False)
        raise AssertionError("expected class ValueError")
    except ValueError as e:
        assert "MiniMaxH3Model" in str(e)

    try:
        node.apply(_fake_model(), nodes.CUSTOM, nodes.CPU_STORAGE, False,
                   start_percent=0.9, end_percent=0.5)
        raise AssertionError("expected window ValueError")
    except ValueError as e:
        assert "start_percent" in str(e)


def test_sigma_extraction_prefers_transformer_options():
    rt = nodes.Ref2VAUltraSafeBlockCacheRuntime
    t = torch.tensor([800.0])
    assert abs(rt._extract_sigma(t, {"sigmas": torch.tensor([0.42])}) - 0.42) < 1e-6
    assert abs(rt._extract_sigma(t, {}) - 0.8) < 1e-6  # x1000 fallback
    assert abs(rt._extract_sigma(t, {"sigmas": torch.tensor([])}) - 0.8) < 1e-6


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().handlers = [logging.NullHandler()]
    tests = [(n, fn) for n, fn in sorted(globals().items()) if n.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {e.__class__.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
