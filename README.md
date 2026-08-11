# ComfyUI MiniMax H3 Ref2VA Accelerator

Quality-first block-cache acceleration for native ComfyUI MiniMax H3 Ref2VA workflows.

![MiniMax H3 Ref2VA Accelerator node](docs/images/h3-ref2va-accelerator-node.png)

Version **0.3.0** turns the experimental Ref2VA cache into a production-oriented node. It preserves the original ComfyUI node class ID, so workflows created with v0.1 or v0.2 upgrade in place.

> [!IMPORTANT]
> This is an approximate accelerator, not bit-identical inference. For exact native behavior, bypass the node or use **Observe Only (no caching)**.

## Highlights

- Built specifically for native ComfyUI `MiniMaxH3Model` Ref2VA layouts.
- Uses Ref2VA-aware global, video, audio, reference, and temporal guards.
- Runs transformer block 0 every step and conditionally reuses the blocks 1-49 residual.
- Prevents consecutive cache hits in every built-in quality preset.
- Offers CPU cache storage for VRAM-constrained BF16 workflows.
- Detects incompatible DiT/block-cache patches instead of silently stacking them.
- Preserves compatibility with object-level attention patches such as SageAttention.

## Tested performance

Primary test: 10-second Ref2VA generation, 32 GB RTX 5090, pruned BF16 MiniMax H3, Sage-style attention, 20-step RES Multistep.

| Mode | Sampling time | Cached steps | Guidance |
| --- | ---: | ---: | --- |
| Native | ~14:43 | 0/20 | Exact native trajectory |
| Ref2VA Conservative | ~12:06 | 4/20 | Smaller observed differences |
| Ref2VA Balanced | ~11:05 | 5/20 | Recommended production compromise |
| Ref2VA Aggressive | ~10:32 | 6/20 | Experimental; more visible drift |

These figures describe one tested workflow, not a universal benchmark. Hardware, checkpoint, video length, resolution, sampler, offloading, and input content all affect results.

## Installation

### ComfyUI Manager

Once the node is listed in ComfyUI Registry/Manager, search for **H3 Ref2VA Accelerator**, install it, and restart ComfyUI.

### Manual installation

From your ComfyUI `custom_nodes` directory:

```bash
git clone https://github.com/BMB12d3/ComfyUI-H3-Ref2VA-Accelerator.git
```

Restart ComfyUI after installation. The node has no additional Python dependencies beyond ComfyUI and PyTorch.

## Quick start

1. Add **MiniMax H3 Ref2VA Accelerator** from `MiniMax H3 > optimization`.
2. Connect the H3 model pipeline to the node's `model` input.
3. Leave `mode` at **Ref2VA Balanced**.
4. Use **CPU (VRAM-safe)** for pruned/offloaded BF16 checkpoints.
5. Connect the patched `MODEL` output to the guider and scheduler/model-sampling path used by the workflow.
6. Queue the workflow and review the console summary after sampling.

Recommended placement:

```text
H3 diffusion model
  -> Memory Efficient Sage Attention (optional)
  -> ModelSamplingMiniMaxH3 / H3 sigma shift
  -> MiniMax H3 Ref2VA Accelerator
  -> Basic Guider + Basic Scheduler
  -> RES Multistep sampler
```

## Modes

| Mode | Best for | Notes |
| --- | --- | --- |
| Ref2VA Balanced | Normal production work | Recommended default |
| Ref2VA Conservative | Keeper shots and higher fidelity | Trades some speed for smaller trajectory changes |
| Ref2VA Ultra Safe | Especially sensitive shots | Narrowest built-in cache profile |
| Ref2VA Aggressive | Previews and experiments | Faster in testing; more visible deviations |
| Observe Only | Diagnostics and calibration | Collects metrics without caching |
| Custom | Advanced tuning | Enables the threshold and window controls |

The advanced threshold values visible on the node are **Custom-mode controls only**. Built-in presets always use their documented internal values, even when different numbers remain visible in the widgets.

## Cache storage

- **CPU (VRAM-safe):** recommended for pruned BF16 or otherwise offloaded H3 checkpoints. It preserves VRAM headroom by holding the persistent tail residual in system RAM.
- **GPU (faster, uses VRAM):** consider only when a smaller/quantized checkpoint leaves comfortable VRAM headroom. Benchmark it for your workflow.

## Do not stack trajectory accelerators

Do not place this node in the same model branch as Spectrum MiniMax H3, generic MiniMax H3 FirstBlockCache, TeaCache/EasyCache-style transformer caching, CacheDiT, T8 Block Cache, or another `double_block`/DiT replacement.

## Compatibility

Designed for native ComfyUI `MiniMaxH3Model` and Ref2VA layouts as of August 2026. Tested with native H3 pruned BF16 and pruned INT8 ConvRot checkpoints, Sage attention patching, DynamicVRAM/AIMDO, and RES Multistep.

## Documentation

See the [User Manual](docs/USER_MANUAL.md) for setup, every control, preset values, troubleshooting, and quality guidance. Release changes are recorded in [CHANGELOG.md](CHANGELOG.md).

## License

No license has been selected yet. Until a license file is added, copyright law reserves all rights to the author.
