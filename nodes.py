from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any

import torch

import comfy.patcher_extension

try:
    import comfy.model_prefetch
    _PREFETCH_AVAILABLE = True
except ImportError:
    # If a future ComfyUI moves/renames the prefetch module, keep the node functional:
    # cache hits still skip block compute, we just cannot suppress tail weight prefetch.
    _PREFETCH_AVAILABLE = False


NODE_VERSION = "0.4.2"

ULTRA_SAFE = "Ref2VA Ultra Safe (quality-first)"
CONSERVATIVE = "Ref2VA Conservative"
BALANCED = "Ref2VA Balanced"
AGGRESSIVE = "Ref2VA Aggressive (experimental)"
OBSERVE = "Observe Only (no caching)"
CUSTOM = "Custom"

CPU_STORAGE = "CPU (VRAM-safe)"
GPU_STORAGE = "GPU (faster, uses VRAM)"

CPU_TAIL_SAFE = "Safe CPU (v0.3 behavior)"
CPU_TAIL_AUTO = "Auto GPU Fast Path"


@dataclass(frozen=True)
class PresetConfig:
    global_threshold: float
    video_threshold: float
    audio_threshold: float
    visual_ref_threshold: float
    audio_ref_threshold: float
    temporal_threshold: float
    start_percent: float
    end_percent: float
    max_consecutive_hits: int
    observe_only: bool = False


# Ref2VA-specific profiles tuned through fixed-seed A/B testing on native ComfyUI H3.
# They are still approximate accelerators, not universal quality guarantees. Balanced is the
# recommended production profile because it produced a substantial speed gain while keeping
# observed trajectory changes small. Aggressive is retained explicitly as an experimental mode.
PRESETS = {
    ULTRA_SAFE: PresetConfig(
        global_threshold=0.070,
        video_threshold=0.070,
        audio_threshold=0.060,
        visual_ref_threshold=0.045,
        audio_ref_threshold=0.045,
        temporal_threshold=0.080,
        start_percent=0.20,
        end_percent=0.85,
        max_consecutive_hits=1,
    ),
    CONSERVATIVE: PresetConfig(
        global_threshold=0.080,
        video_threshold=0.080,
        audio_threshold=0.070,
        visual_ref_threshold=0.055,
        audio_ref_threshold=0.055,
        temporal_threshold=0.095,
        start_percent=0.15,
        end_percent=0.90,
        max_consecutive_hits=1,
    ),
    BALANCED: PresetConfig(
        global_threshold=0.090,
        video_threshold=0.090,
        audio_threshold=0.080,
        visual_ref_threshold=0.065,
        audio_ref_threshold=0.065,
        temporal_threshold=0.110,
        start_percent=0.10,
        end_percent=0.95,
        # Quality-first policy: always keep a native full H3 step between cache hits.
        max_consecutive_hits=1,
    ),
    AGGRESSIVE: PresetConfig(
        global_threshold=0.105,
        video_threshold=0.105,
        audio_threshold=0.095,
        visual_ref_threshold=0.080,
        audio_ref_threshold=0.080,
        temporal_threshold=0.130,
        start_percent=0.05,
        end_percent=0.97,
        # Still forbids consecutive cache hits; the looser guards are the experimental part.
        max_consecutive_hits=1,
    ),
    OBSERVE: PresetConfig(
        global_threshold=0.0,
        video_threshold=0.0,
        audio_threshold=0.0,
        visual_ref_threshold=0.0,
        audio_ref_threshold=0.0,
        temporal_threshold=0.0,
        start_percent=0.0,
        end_percent=1.0,
        max_consecutive_hits=0,
        observe_only=True,
    ),
}


@dataclass
class StepMetrics:
    global_diff: float | None = None
    video_diff: float | None = None
    audio_diff: float | None = None
    visual_ref_diff: float | None = None
    audio_ref_diff: float | None = None
    temporal_diff: float | None = None

    def compact(self) -> str:
        def f(v):
            return "-" if v is None else f"{v:.4f}"
        return (
            f"g={f(self.global_diff)} v={f(self.video_diff)} a={f(self.audio_diff)} "
            f"rv={f(self.visual_ref_diff)} ra={f(self.audio_ref_diff)} t={f(self.temporal_diff)}"
        )


@dataclass
class CacheContext:
    previous_first_residual: torch.Tensor | None = None
    remaining_blocks_residual: torch.Tensor | None = None
    first_block_output: torch.Tensor | None = None
    pending_first_residual: torch.Tensor | None = None
    use_cache: bool = False
    consecutive_hits: int = 0
    previous_sigma: float | None = None
    input_signature: tuple | None = None
    layout: Any = None
    has_refs: bool = False
    metrics: StepMetrics = field(default_factory=StepMetrics)
    tail_scales: list | None = None

    def clear_tensors(self):
        self.previous_first_residual = None
        self.remaining_blocks_residual = None
        self.first_block_output = None
        self.pending_first_residual = None
        self.use_cache = False
        self.consecutive_hits = 0
        self.previous_sigma = None
        self.input_signature = None
        self.layout = None
        self.has_refs = False
        self.metrics = StepMetrics()
        self.tail_scales = None


_PREFETCH_PATCH_LOCK = threading.RLock()


def _nbytes(t: torch.Tensor | None) -> int:
    return 0 if t is None else int(t.numel() * t.element_size())


def _tensor_storage_copy(t: torch.Tensor, storage: str) -> torch.Tensor:
    detached = t.detach()
    if storage == CPU_STORAGE:
        # Preserve dtype exactly. CPU storage is intentionally not pinned by default because
        # the cached hidden tensor can be hundreds of MiB and permanent pinned allocations can
        # interfere with Comfy/Aimdo's own offload pool.
        return detached.to(device="cpu", copy=True)
    return detached.clone()


def _to_device(t: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if t.device == device and t.dtype == dtype:
        return t
    return t.to(device=device, dtype=dtype, non_blocking=False)


def _gpu_headroom_ok(t: torch.Tensor, factor: float = 3.0) -> bool:
    # Gate for the fast tail path: keep block-0 output GPU-resident for one step only when
    # there is comfortable headroom (clone now + tail temp later + slack). Counts both free
    # device memory and the torch allocator's reserved-but-unallocated pool. Any failure or
    # tight headroom falls back to the v0.3 CPU staging path, so this can only help.
    if t.device.type != "cuda":
        return False
    try:
        free, _total = torch.cuda.mem_get_info(t.device)
        reserved = torch.cuda.memory_reserved(t.device)
        allocated = torch.cuda.memory_allocated(t.device)
        available = free + max(reserved - allocated, 0)
    except Exception:
        return False
    return available >= int(t.numel() * t.element_size() * factor)


def _ratio(current: torch.Tensor, previous: torch.Tensor, ranges: list[tuple[int, int]] | None = None) -> float | None:
    if current.shape != previous.shape:
        return None
    if ranges is None:
        # Accumulate in fp32 for parity with the ranged path below. A bf16 reduction quantizes
        # the ratio (~0.4% relative), enough to flip borderline guard decisions against
        # thresholds spaced 0.005 apart.
        numerator = (current - previous).abs().mean(dtype=torch.float32)
        denominator = previous.abs().mean(dtype=torch.float32).clamp(min=1e-8)
        return float((numerator / denominator).item())

    total_num = None
    total_den = None
    count = 0
    for a, b in ranges:
        if b <= a:
            continue
        cur = current[a:b]
        prev = previous[a:b]
        num = (cur - prev).abs().sum(dtype=torch.float32)
        den = prev.abs().sum(dtype=torch.float32)
        total_num = num if total_num is None else total_num + num
        total_den = den if total_den is None else total_den + den
        count += cur.numel()
    if count == 0 or total_num is None or total_den is None:
        return None
    # Mean/mean ratio; count cancels, so sum/sum is equivalent and avoids extra divisions.
    return float((total_num / total_den.clamp(min=1e-8)).item())


def _ranges_for(layout, kinds: set[str]) -> list[tuple[int, int]]:
    if layout is None:
        return []
    return [(a, b) for a, b, kind in layout.segments if kind in kinds]


def _temporal_video_ratio(current: torch.Tensor, previous: torch.Tensor, layout) -> float | None:
    if layout is None or len(getattr(layout, "signature", ())) < 2:
        return None
    video_range = next(((a, b) for a, b, kind in layout.segments if kind == "video"), None)
    if video_range is None:
        return None
    latent_frames = int(layout.signature[1])
    if latent_frames <= 0:
        return None
    a, b = video_range
    cur = current[a:b]
    prev = previous[a:b]
    if cur.shape != prev.shape or cur.shape[0] % latent_frames != 0:
        return None
    rows_per_frame = cur.shape[0] // latent_frames
    cur = cur.reshape(latent_frames, rows_per_frame, -1)
    prev = prev.reshape(latent_frames, rows_per_frame, -1)
    numerator = (cur - prev).abs().mean(dim=(1, 2), dtype=torch.float32)
    denominator = prev.abs().mean(dim=(1, 2), dtype=torch.float32).clamp(min=1e-8)
    return float((numerator / denominator).max().item())


class Ref2VAUltraSafeBlockCacheRuntime:
    def __init__(self, config: PresetConfig, start_sigma: float, end_sigma: float, block_count: int,
                 storage: str, debug: bool, block_modules, tail_rescale: bool = False,
                 cpu_tail_compute: str = CPU_TAIL_SAFE):
        self.config = config
        self.start_sigma = start_sigma
        self.end_sigma = end_sigma
        self.block_count = block_count
        self.storage = storage
        self.debug = debug
        self.tail_rescale = tail_rescale
        self.cpu_tail_compute = cpu_tail_compute
        self.block_ids = tuple(id(b) for b in block_modules)
        self.contexts: dict[tuple, CacheContext] = {}
        self.current: CacheContext | None = None
        self.full_steps = 0
        self.cached_steps = 0
        self.cache_step_numbers: list[int] = []
        self.metric_history: list[StepMetrics] = []
        self.prefetch_queue = None
        self.prefetch_suppressed_steps = 0
        self.steps_without_refs = 0
        self.gpu_tail_steps = 0
        self._layout_warned = False
        self._step_number = 0

    def reset(self):
        for c in self.contexts.values():
            c.clear_tensors()
        self.contexts.clear()
        self.current = None
        self.full_steps = 0
        self.cached_steps = 0
        self.cache_step_numbers.clear()
        self.metric_history.clear()
        self.prefetch_queue = None
        self.prefetch_suppressed_steps = 0
        self.steps_without_refs = 0
        self.gpu_tail_steps = 0
        self._layout_warned = False
        self._step_number = 0

    @staticmethod
    def _input_signature(x):
        tensors = x if isinstance(x, (tuple, list)) else (x,)
        return tuple((tuple(t.shape), t.dtype, str(t.device)) for t in tensors if torch.is_tensor(t))

    @staticmethod
    def _extract_sigma(timestep, transformer_options) -> float:
        # Prefer the sampler-provided sigma when present; it is independent of the model's
        # timestep convention. Fall back to the H3/flow convention (timestep == sigma * 1000).
        sigmas = transformer_options.get("sigmas") if isinstance(transformer_options, dict) else None
        if torch.is_tensor(sigmas) and sigmas.numel() > 0:
            return float(sigmas.flatten()[0].item())
        return float(timestep.flatten()[0].item()) / 1000.0

    def begin_call(self, x, timestep, transformer_options, minimax_payload=None):
        sigma = self._extract_sigma(timestep, transformer_options)
        uuids = transformer_options.get("uuids")
        key = tuple(str(value) for value in uuids) if uuids else ("default",)
        context = self.contexts.setdefault(key, CacheContext())
        signature = self._input_signature(x)
        if context.input_signature != signature or (context.previous_sigma is not None and sigma > context.previous_sigma + 1e-7):
            context.clear_tensors()
        payload = minimax_payload or {}
        context.input_signature = signature
        context.previous_sigma = sigma
        context.layout = payload.get("layout")
        context.has_refs = bool(payload.get("refs")) and context.layout is not None
        context.use_cache = False
        context.metrics = StepMetrics()
        context.tail_scales = None
        if not context.has_refs:
            self.steps_without_refs += 1
        self.current = context
        self.prefetch_queue = None
        self._step_number += 1

    def end_call(self):
        self.current = None
        self.prefetch_queue = None

    def _within_window(self, context: CacheContext) -> bool:
        sigma = context.previous_sigma
        return sigma is not None and self.end_sigma <= sigma <= self.start_sigma

    def capture_prefetch_queue(self, original_make, queue, device, transformer_options):
        out = original_make(queue, device, transformer_options)
        if out is not None and len(queue) == self.block_count:
            try:
                if tuple(id(b) for b in queue) == self.block_ids:
                    self.prefetch_queue = out
            except Exception:
                pass
        return out

    def neutralize_tail_prefetch(self):
        # At block-0 return time the native queue has already prefetched block 0 only.
        # Keep entry 0 so native cleanup still runs, but replace all future block modules
        # with None so skipped blocks do not trigger dynamic weight transfers.
        q = self.prefetch_queue
        if q is None:
            return
        try:
            for i in range(1, len(q)):
                q[i] = None
            self.prefetch_suppressed_steps += 1
        except Exception:
            logging.debug("H3 Ref2VA Block Cache could not neutralize prefetch tail", exc_info=True)

    def _metrics(self, first_residual: torch.Tensor, previous: torch.Tensor, layout) -> StepMetrics:
        video_ranges = _ranges_for(layout, {"video"})
        audio_ranges = _ranges_for(layout, {"audio"})
        visual_ref_ranges = _ranges_for(layout, {"ref_img", "cond"})
        audio_ref_ranges = _ranges_for(layout, {"ref_audio"})
        return StepMetrics(
            global_diff=_ratio(first_residual, previous),
            video_diff=_ratio(first_residual, previous, video_ranges),
            audio_diff=_ratio(first_residual, previous, audio_ranges),
            visual_ref_diff=_ratio(first_residual, previous, visual_ref_ranges),
            audio_ref_diff=_ratio(first_residual, previous, audio_ref_ranges),
            temporal_diff=_temporal_video_ratio(first_residual, previous, layout),
        )

    def _warn_missing_metrics(self, metrics: StepMetrics):
        # A required guard metric coming back None means guards can never pass and the node
        # silently never caches (safe, but confusing). Most likely causes: a native H3/Ref2VA
        # layout change after a ComfyUI update, or a workflow without the expected segments.
        if self._layout_warned:
            return
        missing = [
            name for name, value in (
                ("global", metrics.global_diff),
                ("video", metrics.video_diff),
                ("audio", metrics.audio_diff),
                ("temporal", metrics.temporal_diff),
            ) if value is None
        ]
        if missing:
            self._layout_warned = True
            logging.warning(
                "H3 Ref2VA Accelerator: required guard metric(s) unavailable: %s. Caching stays "
                "disabled for safety. The native H3/Ref2VA layout may have changed or this workflow "
                "lacks those segments; run Observe Only with debug enabled for details.",
                ", ".join(missing),
            )

    @staticmethod
    def _segment_scales(current: torch.Tensor, previous: torch.Tensor, layout,
                        clamp: float = 0.10) -> list[tuple[int, int, float]]:
        # Experimental first-order drift correction: per-segment energy ratio of the current vs
        # cached block-0 residual, clamped to +/-10%. Rows outside any layout segment stay at 1.0.
        scales: list[tuple[int, int, float]] = []
        if layout is None:
            return scales
        lo, hi = 1.0 - clamp, 1.0 + clamp
        for a, b, _kind in layout.segments:
            if b <= a:
                continue
            cur_e = current[a:b].abs().mean(dtype=torch.float32)
            prev_e = previous[a:b].abs().mean(dtype=torch.float32).clamp(min=1e-8)
            s = float((cur_e / prev_e).item())
            if math.isfinite(s):
                scales.append((a, b, min(max(s, lo), hi)))
        return scales

    @staticmethod
    def _finite_leq(value: float | None, threshold: float, required: bool = True) -> bool:
        if value is None:
            return not required
        return math.isfinite(value) and value <= threshold

    def _passes_guards(self, metrics: StepMetrics, layout) -> bool:
        cfg = self.config
        has_visual_ref = bool(_ranges_for(layout, {"ref_img", "cond"}))
        has_audio_ref = bool(_ranges_for(layout, {"ref_audio"}))
        checks = [
            self._finite_leq(metrics.global_diff, cfg.global_threshold),
            self._finite_leq(metrics.video_diff, cfg.video_threshold),
            self._finite_leq(metrics.audio_diff, cfg.audio_threshold),
            self._finite_leq(metrics.temporal_diff, cfg.temporal_threshold),
        ]
        if has_visual_ref:
            checks.append(self._finite_leq(metrics.visual_ref_diff, cfg.visual_ref_threshold))
        if has_audio_ref:
            checks.append(self._finite_leq(metrics.audio_ref_diff, cfg.audio_ref_threshold))
        return all(checks)

    @torch.compiler.disable()
    def decide(self, first_residual: torch.Tensor, first_output: torch.Tensor):
        context = self.current
        if context is None:
            raise RuntimeError("H3 Ref2VA Block Cache called outside active model execution")

        previous_stored = context.previous_first_residual
        tail = context.remaining_blocks_residual
        use_cache = False
        metrics = StepMetrics()

        can_compare = (
            context.has_refs
            and previous_stored is not None
            and tail is not None
            and tuple(previous_stored.shape) == tuple(first_residual.shape)
            and tuple(tail.shape) == tuple(first_output.shape)
        )

        if can_compare:
            previous = _to_device(previous_stored, first_residual.device, first_residual.dtype)
            metrics = self._metrics(first_residual, previous, context.layout)
            self.metric_history.append(metrics)
            self._warn_missing_metrics(metrics)

            if not self.config.observe_only and self._within_window(context):
                use_cache = self._passes_guards(metrics, context.layout)
                use_cache = use_cache and context.consecutive_hits < self.config.max_consecutive_hits

            if use_cache and self.tail_rescale:
                context.tail_scales = self._segment_scales(first_residual, previous, context.layout)
            if previous is not previous_stored:
                del previous

        context.metrics = metrics
        context.use_cache = use_cache

        if use_cache:
            context.consecutive_hits += 1
            self.cache_step_numbers.append(self._step_number)
            self.neutralize_tail_prefetch()
            if self.debug:
                logging.info("H3 Ref2VA Block Cache step %d CACHE sigma=%.5f %s",
                             self._step_number, context.previous_sigma, metrics.compact())
        else:
            context.consecutive_hits = 0
            # Keep the current block-0 output for constructing the exact tail residual after the
            # last block. In CPU cache mode, v0.4.1 defaults to the proven v0.3 behavior: stage
            # block-0 output on CPU immediately. The optional Au…142 tokens truncated…ck_output = _tensor_storage_copy(first_output, self.storage)
            context.pending_first_residual = _tensor_storage_copy(first_residual, self.storage)
            if self.debug:
                reason = "observe/full" if self.config.observe_only else "FULL"
                logging.info("H3 Ref2VA Block Cache step %d %s sigma=%.5f %s",
                             self._step_number, reason, context.previous_sigma, metrics.compact())

    def finish_full_step(self, output: torch.Tensor):
        context = self.current
        if context is None or context.first_block_output is None or context.pending_first_residual is None:
            raise RuntimeError("H3 Ref2VA Block Cache full-step state is incomplete")

        first = context.first_block_output
        if first.device == output.device:
            # Fast path: block-0 output is still on the compute device (GPU storage mode, or the
            # CPU-storage headroom fast path). torch.sub allocates a fresh tensor, so no extra
            # clone is needed; CPU storage then ships the finished tail with one D2H copy.
            tail = torch.sub(output.detach(), first)
            if self.storage == CPU_STORAGE:
                if first.device.type == "cuda":
                    self.gpu_tail_steps += 1
                tail = tail.to(device="cpu")
        else:
            # Legacy low-VRAM path: block-0 output was staged on CPU at decide() time.
            tail = output.detach().to(device="cpu", copy=True)
            tail.sub_(first)
        context.remaining_blocks_residual = tail

        context.previous_first_residual = context.pending_first_residual
        context.first_block_output = None
        context.pending_first_residual = None
        self.full_steps += 1

    def finish_cached_step(self, first_output: torch.Tensor) -> torch.Tensor:
        context = self.current
        if context is None or context.remaining_blocks_residual is None:
            raise RuntimeError("H3 Ref2VA Block Cache has no tail residual")
        tail = _to_device(context.remaining_blocks_residual, first_output.device, first_output.dtype)
        scales = context.tail_scales
        if scales:
            if tail is context.remaining_blocks_residual:
                # GPU storage returns the persistent cache itself; never mutate it in place.
                tail = tail.clone()
            for a, b, s in scales:
                if s != 1.0:
                    tail[a:b].mul_(s)
            if self.debug:
                svals = [s for _a, _b, s in scales]
                logging.info("H3 Ref2VA Block Cache step %d tail rescale: %d segment(s), min/max %.4f/%.4f",
                             self._step_number, len(svals), min(svals), max(svals))
        first_output.add_(tail)
        if tail is not context.remaining_blocks_residual:
            del tail
        self.cached_steps += 1
        return first_output

    def summary(self, label: str) -> str:
        steps = self.full_steps + self.cached_steps
        if steps == 0:
            if self.steps_without_refs:
                return (
                    f"H3 Ref2VA Accelerator v{NODE_VERSION} result: model ran "
                    f"{self.steps_without_refs} step(s) without a Ref2VA payload; "
                    "accelerator stayed idle (no caching applied)"
                )
            return f"H3 Ref2VA Accelerator v{NODE_VERSION} result: no H3 model steps"

        baseline_blocks = steps * self.block_count
        executed_blocks = self.full_steps * self.block_count + self.cached_steps
        theoretical = baseline_blocks / max(executed_blocks, 1)
        work_saved_pct = 100.0 * (1.0 - (executed_blocks / max(baseline_blocks, 1)))
        cached_pct = 100.0 * self.cached_steps / steps

        cache_bytes = 0
        for c in self.contexts.values():
            cache_bytes += _nbytes(c.previous_first_residual) + _nbytes(c.remaining_blocks_residual)
        mib = cache_bytes / (1024 ** 2)
        where = "CPU" if self.storage == CPU_STORAGE else "GPU"

        lines = [
            f"H3 Ref2VA Accelerator v{NODE_VERSION} result",
            f"  preset: {label}",
            f"  steps: {self.full_steps} full + {self.cached_steps} cached = {steps} "
            f"({cached_pct:.1f}% cached)",
            f"  transformer block work: {executed_blocks}/{baseline_blocks} "
            f"(~{work_saved_pct:.1f}% avoided; {theoretical:.2f}x theoretical block-work factor)",
            f"  persistent cache: ~{mib:.1f} MiB on {where}",
        ]
        if self.cache_step_numbers:
            lines.append(f"  cache steps: {self.cache_step_numbers}")
        else:
            lines.append("  cache steps: none")
        if self.prefetch_suppressed_steps:
            lines.append(f"  tail prefetch suppressed: {self.prefetch_suppressed_steps}x")
        if self.storage == CPU_STORAGE and self.full_steps:
            lines.append(f"  CPU tail compute: {self.cpu_tail_compute}")
            if self.cpu_tail_compute == CPU_TAIL_AUTO:
                lines.append(f"  fast GPU tail path: {self.gpu_tail_steps}/{self.full_steps} full steps")
        if self.steps_without_refs:
            lines.append(f"  steps without Ref2VA payload: {self.steps_without_refs} (not accelerated)")

        finite_global = sorted(
            m.global_diff for m in self.metric_history
            if m.global_diff is not None and math.isfinite(m.global_diff)
        )
        finite_ref = sorted(
            m.visual_ref_diff for m in self.metric_history
            if m.visual_ref_diff is not None and math.isfinite(m.visual_ref_diff)
        )
        if finite_global:
            mid = finite_global[len(finite_global)//2]
            lines.append(
                f"  global diff min/med/max: {finite_global[0]:.4f}/{mid:.4f}/{finite_global[-1]:.4f}"
            )
        if finite_ref:
            mid = finite_ref[len(finite_ref)//2]
            lines.append(
                f"  visual-ref diff min/med/max: {finite_ref[0]:.4f}/{mid:.4f}/{finite_ref[-1]:.4f}"
            )
        return "\n".join(lines)



def make_block_patch(runtime: Ref2VAUltraSafeBlockCacheRuntime, index: int, last_index: int):
    # Note: no existing-patch chaining. apply() refuses to install when another DiT
    # double_block replacement is present, so the original block is always the base callee.
    def patch(args, extra):
        context = runtime.current
        if context is None or not context.has_refs:
            return extra["original_block"](args)

        if index == 0:
            # Native H3 blocks mutate their hidden state in place. Clone the input once, then
            # recycle that clone into the first-block residual to avoid a second full-size allocation.
            first_input = args["img"].detach().clone()
            output = extra["original_block"](args)["img"]
            first_input.mul_(-1).add_(output)  # now first_input == output - original_input
            runtime.decide(first_input, output)
            del first_input
            return {"img": output}

        if context.use_cache:
            # Keep blocks 1..48 as identity; inject the cached tail once at block 49 so the
            # native final layer sees a normal full-stack-shaped hidden tensor.
            if index == last_index:
                return {"img": runtime.finish_cached_step(args["img"])}
            return {"img": args["img"]}

        output = extra["original_block"](args)["img"]
        if index == last_index:
            runtime.finish_full_step(output)
        return {"img": output}

    return patch


def make_diffusion_wrapper(runtime: Ref2VAUltraSafeBlockCacheRuntime):
    def wrapper(executor, *args, **kwargs):
        transformer_options = args[3] if len(args) > 3 else kwargs.get("transformer_options", {})
        minimax_payload = args[4] if len(args) > 4 else kwargs.get("minimax_payload")
        runtime.begin_call(args[0], args[1], transformer_options, minimax_payload)
        try:
            return executor(*args, **kwargs)
        finally:
            runtime.end_call()
    return wrapper


def make_sample_wrapper(runtime: Ref2VAUltraSafeBlockCacheRuntime, label: str):
    def wrapper(executor, *args, **kwargs):
        runtime.reset()
        logging.info(
            "H3 Ref2VA Accelerator %s enabled: %s; storage=%s; "
            "g/v/a/rv/ra/t=%.3f/%.3f/%.3f/%.3f/%.3f/%.3f; window=%.2f-%.2f; max_consecutive=%d",
            NODE_VERSION, label, runtime.storage,
            runtime.config.global_threshold, runtime.config.video_threshold, runtime.config.audio_threshold,
            runtime.config.visual_ref_threshold, runtime.config.audio_ref_threshold, runtime.config.temporal_threshold,
            runtime.config.start_percent, runtime.config.end_percent, runtime.config.max_consecutive_hits,
        )
        if label.startswith(AGGRESSIVE):
            logging.warning(
                "H3 Ref2VA Accelerator: Aggressive is experimental. It can produce more visible "
                "trajectory differences in distant subjects, pose/head angle, lips, and fine motion."
            )
        elif label.startswith(BALANCED):
            logging.info(
                "H3 Ref2VA Accelerator: Balanced is the recommended production profile from current Ref2VA testing."
            )
        if runtime.tail_rescale:
            logging.warning(
                "H3 Ref2VA Accelerator: tail_rescale is experimental. Fixed-seed testing showed "
                "slightly different output without a clear quality advantage; leave it OFF for production."
            )
        if runtime.storage == CPU_STORAGE and runtime.cpu_tail_compute == CPU_TAIL_AUTO:
            logging.warning(
                "H3 Ref2VA Accelerator: Auto GPU Fast Path is a benchmark/experimental option. "
                "On the validated RTX 5090 BF16 workflow it saved only about 2 seconds over an ~11 minute sampler run; "
                "Safe CPU is recommended for production and maximum VRAM headroom."
            )
        try:
            if _PREFETCH_AVAILABLE:
                # Native H3 may use Comfy/Aimdo's dynamic block prefetch. On a cache-hit step we
                # must stop blocks 1..49 from being prefetched, otherwise BF16 offload traffic can
                # erase much of the compute saving. Capture only this H3 model's queue and
                # neutralize it after block 0 decides to cache. The monkeypatch is restored in finally.
                with _PREFETCH_PATCH_LOCK:
                    original_make = comfy.model_prefetch.make_prefetch_queue

                    def make_prefetch_queue_wrapper(queue, device, transformer_options):
                        return runtime.capture_prefetch_queue(original_make, queue, device, transformer_options)

                    comfy.model_prefetch.make_prefetch_queue = make_prefetch_queue_wrapper
                    try:
                        return executor(*args, **kwargs)
                    finally:
                        comfy.model_prefetch.make_prefetch_queue = original_make
            else:
                logging.warning(
                    "H3 Ref2VA Accelerator: comfy.model_prefetch not found; running without "
                    "tail-prefetch suppression. Cache hits still skip block compute, but offloaded "
                    "checkpoints may see reduced wall-clock savings."
                )
                return executor(*args, **kwargs)
        finally:
            logging.info("\n%s", runtime.summary(label))
            runtime.reset()
    return wrapper


class ApplyH3Ref2VAUltraSafeBlockCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": ([BALANCED, CONSERVATIVE, ULTRA_SAFE, AGGRESSIVE, OBSERVE, CUSTOM], {"default": BALANCED}),
                "cache_storage": ([CPU_STORAGE, GPU_STORAGE], {"default": CPU_STORAGE, "tooltip": "CPU is recommended for pruned BF16/offloaded H3. GPU can be slightly faster for smaller quantized checkpoints with comfortable VRAM headroom."}),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "tail_rescale": ("BOOLEAN", {"default": False, "advanced": True, "tooltip": "Experimental only. Fixed-seed testing produced slightly different output without a clear quality benefit. Production recommendation: leave OFF."}),
                "global_threshold": ("FLOAT", {"default": 0.090, "min": 0.0, "max": 1.0, "step": 0.005, "advanced": True, "tooltip": "Custom mode only: overall block-0 residual-change limit."}),
                "video_threshold": ("FLOAT", {"default": 0.090, "min": 0.0, "max": 1.0, "step": 0.005, "advanced": True, "tooltip": "Custom mode only: target-video residual-change limit."}),
                "audio_threshold": ("FLOAT", {"default": 0.080, "min": 0.0, "max": 1.0, "step": 0.005, "advanced": True, "tooltip": "Custom mode only: target-audio residual-change limit."}),
                "visual_ref_threshold": ("FLOAT", {"default": 0.065, "min": 0.0, "max": 1.0, "step": 0.005, "advanced": True, "tooltip": "Custom mode only: visual-reference residual-change limit."}),
                "audio_ref_threshold": ("FLOAT", {"default": 0.065, "min": 0.0, "max": 1.0, "step": 0.005, "advanced": True, "tooltip": "Custom mode only: reference-audio residual-change limit."}),
                "temporal_threshold": ("FLOAT", {"default": 0.110, "min": 0.0, "max": 1.0, "step": 0.005, "advanced": True, "tooltip": "Custom mode only: worst target-video frame-change limit."}),
                "start_percent": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True, "tooltip": "Custom mode only: earliest denoising fraction where cache hits are allowed."}),
                "end_percent": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True, "tooltip": "Custom mode only: latest denoising fraction where cache hits are allowed."}),
                "max_consecutive_hits": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1, "advanced": True, "tooltip": "Custom mode only. Quality-first recommendation: leave at 1."}),
                "cpu_tail_compute": ([CPU_TAIL_SAFE, CPU_TAIL_AUTO], {"default": CPU_TAIL_SAFE, "advanced": True, "tooltip": "CPU cache mode only. Production recommendation: Safe CPU. Auto GPU Fast Path is retained for benchmarking; on the validated RTX 5090 BF16 workflow it saved only ~2 seconds over an ~11 minute sampler run."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3/optimization"
    DESCRIPTION = (
        "Production-oriented MiniMax H3 Ref2VA accelerator. Balanced is the recommended quality/speed "
        "profile. Block 0 runs every step; blocks 1-49 are reused only when global, target-video, "
        "target-audio, reference and temporal guards all pass. This is deterministic but approximate. "
        "CPU cache storage is recommended for offloaded pruned BF16 H3."
    )

    def apply(
        self,
        model,
        mode,
        cache_storage,
        debug,
        tail_rescale=False,
        global_threshold=0.090,
        video_threshold=0.090,
        audio_threshold=0.080,
        visual_ref_threshold=0.065,
        audio_ref_threshold=0.065,
        temporal_threshold=0.110,
        start_percent=0.10,
        end_percent=0.95,
        max_consecutive_hits=1,
        cpu_tail_compute=CPU_TAIL_SAFE,
    ):
        if mode == CUSTOM:
            if start_percent >= end_percent:
                raise ValueError("start_percent must be less than end_percent")
            config = PresetConfig(
                global_threshold=float(global_threshold),
                video_threshold=float(video_threshold),
                audio_threshold=float(audio_threshold),
                visual_ref_threshold=float(visual_ref_threshold),
                audio_ref_threshold=float(audio_ref_threshold),
                temporal_threshold=float(temporal_threshold),
                start_percent=float(start_percent),
                end_percent=float(end_percent),
                max_consecutive_hits=int(max_consecutive_hits),
            )
            label = (
                f"Custom g/v/a/rv/ra/t={global_threshold:.3f}/{video_threshold:.3f}/{audio_threshold:.3f}/"
                f"{visual_ref_threshold:.3f}/{audio_ref_threshold:.3f}/{temporal_threshold:.3f}; "
                f"window={start_percent:.2f}-{end_percent:.2f}; max={max_consecutive_hits}"
            )
        else:
            config = PRESETS[mode]
            label = mode
        if tail_rescale:
            label = f"{label}; tail_rescale=ON (experimental)"
        if cpu_tail_compute not in (CPU_TAIL_SAFE, CPU_TAIL_AUTO):
            raise ValueError(f"Unknown cpu_tail_compute mode: {cpu_tail_compute}")
        if cache_storage == CPU_STORAGE and cpu_tail_compute == CPU_TAIL_AUTO:
            label = f"{label}; cpu_tail=Auto GPU Fast Path"

        diffusion_model = model.get_model_object("diffusion_model")
        if diffusion_model.__class__.__name__ != "MiniMaxH3Model" or not hasattr(diffusion_model, "blocks"):
            raise ValueError(
                f"H3 Ref2VA Accelerator only supports native ComfyUI MiniMaxH3Model; got {diffusion_model.__class__.__name__}"
            )
        block_count = len(diffusion_model.blocks)
        if block_count < 2:
            raise ValueError(f"H3 Ref2VA Accelerator needs at least two transformer blocks; got {block_count}")

        # Another DiT block replacement changes the same execution layer. We deliberately refuse
        # rather than silently overwrite it. Object-level attention patches (such as SageAttention)
        # remain compatible because the native block wrapper still invokes the patched block.
        transformer_options = model.model_options.get("transformer_options", {})
        existing_dit = transformer_options.get("patches_replace", {}).get("dit", {})
        conflicts = [("double_block", i) for i in range(block_count) if ("double_block", i) in existing_dit]
        if conflicts:
            raise ValueError(
                "H3 Ref2VA Accelerator found an existing DiT block replacement. Do not combine it with "
                "FirstBlockCache/CacheDiT/T8/Spectrum-style block replacement nodes in the same branch."
            )

        model_sampling = model.get_model_object("model_sampling")
        start_sigma = float(model_sampling.percent_to_sigma(config.start_percent))
        end_sigma = float(model_sampling.percent_to_sigma(config.end_percent))

        patched = model.clone()
        runtime = Ref2VAUltraSafeBlockCacheRuntime(
            config=config,
            start_sigma=start_sigma,
            end_sigma=end_sigma,
            block_count=block_count,
            storage=cache_storage,
            debug=bool(debug),
            block_modules=list(diffusion_model.blocks),
            tail_rescale=bool(tail_rescale),
            cpu_tail_compute=cpu_tail_compute,
        )

        for index in range(block_count):
            patched.set_model_patch_replace(
                make_block_patch(runtime, index, block_count - 1),
                "dit",
                "double_block",
                index,
            )

        key = f"h3_ref2va_ultrasafe_block_cache_{id(runtime)}"
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            key,
            make_diffusion_wrapper(runtime),
        )
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            key,
            make_sample_wrapper(runtime, label),
        )
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "ApplyH3Ref2VAUltraSafeBlockCache": ApplyH3Ref2VAUltraSafeBlockCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyH3Ref2VAUltraSafeBlockCache": "MiniMax H3 Ref2VA Accelerator",
}
