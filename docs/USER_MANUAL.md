# MiniMax H3 Ref2VA Accelerator — User Manual

Version 0.3.0

## 1. What this node does

The MiniMax H3 Ref2VA Accelerator reduces sampling time by conditionally reusing part of the H3 transformer's previous-step result. Transformer block 0 still runs on every denoising step. The residual produced by blocks 1-49 is reused only when all applicable Ref2VA change guards pass.

The guards check:

- overall block-0 residual change;
- target-video change;
- target-audio change;
- visual-reference change;
- reference-audio change; and
- worst target-video latent-frame change.

Every built-in quality preset limits caching to one hit at a time. A real full transformer pass therefore occurs before another cache hit can happen.

## 2. The node at a glance

![MiniMax H3 Ref2VA Accelerator node and controls](images/h3-ref2va-accelerator-node.png)

The screenshot shows the complete node with advanced widgets visible. The displayed threshold values are only applied when `mode` is **Custom**. When a built-in preset is selected, its internal preset values are used.

## 3. Requirements

- A current native ComfyUI installation with MiniMax H3 support.
- A native MiniMax H3 Ref2VA diffusion model loaded as `MiniMaxH3Model`.
- Python 3.10 or newer.
- Sufficient system RAM or VRAM for the selected cache-storage mode.

No extra Python packages are required by this node.

## 4. Installation

### Install with ComfyUI Manager

1. Open ComfyUI Manager.
2. Search for **H3 Ref2VA Accelerator**.
3. Install the custom node.
4. Restart ComfyUI.

### Install manually

1. Open a terminal in `ComfyUI/custom_nodes`.
2. Clone or extract this repository so the directory is named `ComfyUI-H3-Ref2VA-Accelerator`.
3. Restart ComfyUI.
4. Confirm that **MiniMax H3 Ref2VA Accelerator** appears under `MiniMax H3 > optimization`.

## 5. Add the node to a workflow

Place the accelerator after model-loading, attention patching, and H3 sampling/sigma-shift setup, but before the guider and scheduler consume the model.

```text
H3 diffusion model
  -> Memory Efficient Sage Attention (if used)
  -> ModelSamplingMiniMaxH3 / H3 sigma shift
  -> MiniMax H3 Ref2VA Accelerator
  -> Basic Guider + Basic Scheduler
  -> RES Multistep
```

Use the accelerator's patched `MODEL` output everywhere the downstream guider or scheduler path expects that model. If one branch bypasses it, that branch will not be accelerated.

## 6. Recommended first run

1. Set `mode` to **Ref2VA Balanced**.
2. Set `cache_storage` to **CPU (VRAM-safe)** when using a pruned/offloaded BF16 checkpoint.
3. Leave `debug` off.
4. Queue a normal fixed-seed workflow.
5. Read the end-of-run console summary to see the hit count and cache behavior.
6. Compare the result with a native/bypassed run before adopting the node for keeper shots.

## 7. Controls

### model

The native MiniMax H3 model to patch. The node rejects other diffusion-model classes.

### mode

Chooses a tested preset, diagnostics-only operation, or Custom tuning.

### cache_storage

- **CPU (VRAM-safe):** stores the persistent residual cache in system RAM and transfers it to the GPU on a hit. Recommended for pruned BF16/offloaded workflows.
- **GPU (faster, uses VRAM):** keeps the cache resident in VRAM. Use only when your checkpoint leaves ample VRAM headroom.

### debug

Enables detailed per-step diagnostic logging. Leave it off for ordinary production runs; enable it when studying guard behavior or tuning Custom mode.

### Custom-only advanced controls

These widgets are ignored by Balanced, Conservative, Ultra Safe, Aggressive, and Observe Only.

| Control | Meaning | Balanced-style Custom default |
| --- | --- | ---: |
| `global_threshold` | Overall block-0 residual-change limit | 0.090 |
| `video_threshold` | Target-video residual-change limit | 0.090 |
| `audio_threshold` | Target-audio residual-change limit | 0.080 |
| `visual_ref_threshold` | Visual-reference residual-change limit | 0.065 |
| `audio_ref_threshold` | Reference-audio residual-change limit | 0.065 |
| `temporal_threshold` | Worst target-video frame-change limit | 0.110 |
| `start_percent` | Earliest denoising fraction where hits are allowed | 0.10 |
| `end_percent` | Latest denoising fraction where hits are allowed | 0.95 |
| `max_consecutive_hits` | Maximum back-to-back hits | 1 |

Higher thresholds permit more cache hits but can increase trajectory drift. Wider cache windows do the same. For quality-first use, keep `max_consecutive_hits` at 1.

`start_percent` must be lower than `end_percent`.

## 8. Preset reference

| Preset | Global | Video | Audio | Visual ref | Audio ref | Temporal | Window | Max hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ultra Safe | 0.070 | 0.070 | 0.060 | 0.045 | 0.045 | 0.080 | 20%-85% | 1 |
| Conservative | 0.080 | 0.080 | 0.070 | 0.055 | 0.055 | 0.095 | 15%-90% | 1 |
| Balanced | 0.090 | 0.090 | 0.080 | 0.065 | 0.065 | 0.110 | 10%-95% | 1 |
| Aggressive | 0.105 | 0.105 | 0.095 | 0.080 | 0.080 | 0.130 | 5%-97% | 1 |

### Balanced — recommended

Use for normal production. In the primary test it reduced native sampling from about 14:43 to 11:05 and cached 5 of 20 steps. Fixed-seed overlays showed small but real differences, most visible in distant/wide-shot details.

### Conservative

Use when fidelity matters more than maximum speed. The primary test ran around 12:06 and cached 4 of 20 steps, with smaller overlay differences than Balanced.

### Ultra Safe

Use for especially sensitive shots. It has the narrowest built-in thresholds and cache window.

### Aggressive — experimental

Use for previews and testing. It reached about 10:32 and cached 6 of 20 steps in the primary test, but showed more noticeable differences such as distant-subject orientation, head angle, and lip detail.

### Observe Only

Executes every transformer block while collecting the same metrics. Use it for diagnostics and calibration without changing the native computation path through caching.

### Custom

Activates all advanced threshold, cache-window, and consecutive-hit controls. Start from the Balanced defaults and adjust one factor at a time with a fixed seed.

## 9. Quality and validation workflow

This accelerator is deterministic for a fixed configuration but is not native/bit-identical H3.

For a new workflow:

1. Render a short native/bypassed fixed-seed baseline.
2. Render the same seed with Balanced.
3. Compare motion timing, lips, hands, distant subjects, small props, and wide-shot details.
4. Move to Conservative or Ultra Safe if the differences are unacceptable.
5. Use native/bypassed inference for critical keepers that require the exact native trajectory.

## 10. Incompatible accelerators

Do not combine this node in the same model branch with:

- Spectrum MiniMax H3;
- generic MiniMax H3 FirstBlockCache;
- TeaCache/EasyCache-style transformer caching;
- CacheDiT;
- T8 Block Cache; or
- another `double_block`/DiT replacement.

The node deliberately raises an error when an existing H3 DiT block replacement is found. SageAttention-style object-level attention patches remain compatible.

## 11. Troubleshooting

### “Only supports native ComfyUI MiniMaxH3Model”

The connected model is not the supported native H3 model class. Check the loader and ensure no wrapper has replaced the underlying diffusion model with an incompatible class.

### “Found an existing DiT block replacement”

Another trajectory or block-cache accelerator is already active in the same model branch. Remove one of the competing accelerators.

### Out of memory with GPU cache

Switch `cache_storage` to **CPU (VRAM-safe)**. GPU storage can consume multiple GiB depending on the workflow.

### Little or no speed improvement

- Confirm that the patched model output feeds the full guider/scheduler path.
- Check the console summary for cache hits.
- Content with large step-to-step changes may correctly fail the guards.
- Use Observe Only with `debug` enabled to inspect metrics.
- Benchmark sampling time separately from model load, VAE, and other workflow stages.

### Output differs from the native run

This is expected to some degree because cache hits approximate the native trajectory. Try Conservative or Ultra Safe, or bypass the accelerator for exact native behavior.

### Custom values appear to do nothing

Set `mode` to **Custom**. Advanced widgets do not override the built-in presets.

## 12. Tested environment

The v0.3 production profile was tested on a 32 GB RTX 5090 with native H3 pruned BF16 and pruned INT8 ConvRot checkpoints, Sage attention patching, DynamicVRAM/AIMDO, and RES Multistep. Compatibility and performance may change as ComfyUI's native H3 implementation evolves.
