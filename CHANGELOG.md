# Changelog

## 0.4.2

- Production-cleanup release; no change to Balanced/Conservative/Ultra Safe/Aggressive cache thresholds or hit policy.
- Locked the documented production recommendation to **Balanced + CPU (VRAM-safe) + Safe CPU + tail_rescale OFF**.
- Moved `cpu_tail_compute` into Advanced controls while preserving its values, default, widget order, and workflow compatibility.
- Kept **Auto GPU Fast Path** as an advanced benchmark option after back-to-back RTX 5090 + pruned BF16 testing showed only about a 2-second sampler improvement (~11:15 -> ~11:13) despite engaging on all 15 full steps.
- Kept `tail_rescale` available but explicitly experimental/off-by-default after fixed-seed testing showed slightly different output without a clear quality advantage.
- Added clear console warnings when either experimental tail feature is enabled.
- Fixed Balanced/Aggressive startup messaging so it still appears when an experimental label suffix is present.
- Updated README and user manual with the validated production defaults and test conclusions.
- Preserved the existing node class ID and input ordering for upgrade compatibility.

## 0.4.1

- Added explicit `cpu_tail_compute` control for CPU cache storage.
- Made **Safe CPU (v0.3 behavior)** the default/recommended tail path for pruned BF16 and heavily offloaded DynamicVRAM/AIMDO workflows.
- Kept the v0.4 **Auto GPU Fast Path** as an opt-in benchmark mode rather than enabling it automatically.
- Added end-of-run reporting for the selected CPU tail mode and fast-path usage when Auto is enabled.
- Preserved all v0.4 fp32 guard metrics, sigma handling, diagnostics, prefetch fallback, experimental `tail_rescale`, and node class/workflow compatibility.
- Appended the new widget after existing v0.4 controls to minimize saved-workflow widget-order disruption.

## 0.4.0

- Formed the tail residual on the GPU when VRAM headroom allows (CPU storage mode): one device-to-host copy per full step instead of two plus a CPU bf16 subtract. Falls back to the v0.3 staging path automatically when headroom is tight; the summary reports fast-path usage.
- Removed a redundant full-size allocation per full step in GPU storage mode.
- Switched every guard metric to fp32 accumulation for parity with the ranged path. A bf16 reduction quantizes ratios by roughly 0.4%, enough to flip borderline guard decisions against thresholds spaced 0.005 apart.
- Read sigma from `transformer_options` when the sampler provides it, keeping the ×1000 flow-timestep fallback.
- Reported steps that ran without a Ref2VA payload instead of claiming "no H3 model steps" after a full run.
- Warned once per run when a required guard metric is unavailable (for example after a native layout change), instead of silently never caching.
- Made `comfy.model_prefetch` optional: if a future ComfyUI removes or renames it, the node loads and runs with tail-prefetch suppression disabled and a clear warning instead of failing to import.
- Added experimental `tail_rescale` (default off, any caching mode): per-segment energy rescaling of the reused tail on cache hits, clamped to ±10%. Fixed-seed A/B before production use.
- Added a headless test suite (`tests/test_runtime.py`, no ComfyUI or GPU required) covering cache reconstruction, guard windows, consecutive-hit limits, context resets, rescale math, cache integrity, idle diagnostics, and wrapper behavior.
- Aligned the pyproject package name with the repository name.
- Preserved the node class ID; existing workflows load unchanged.

## 0.3.0

- Promoted **Ref2VA Balanced** to the recommended/default production preset after fixed-seed BF16 and INT8 testing.
- Added **Ref2VA Aggressive (experimental)** using the tested 0.105/0.105/0.095/0.080/0.080/0.130 guard set and 5%–97% cache window.
- Kept `max_consecutive_hits = 1` in every quality preset, including Aggressive.
- Preserved Ultra Safe and Conservative thresholds.
- Changed Custom Advanced defaults to mirror Balanced, making manual tuning start from the production profile.
- Clarified cache-storage guidance: CPU for offloaded/pruned BF16; GPU only when the checkpoint leaves comfortable VRAM headroom.
- Added a console warning when Aggressive is selected and a production note when Balanced is selected.
- Updated documentation with observed fixed-seed overlay behavior: distant/wide-shot elements were more sensitive to cache trajectory changes than close-up framing.
- Documented and retired the static-reference-cache, kernel-fusion and custom-prefetch experiments because they did not provide worthwhile wall-clock gains.
- Preserved the existing node class ID for v0.1/v0.2 workflow compatibility.

## 0.2.0

- Renamed the display node to **MiniMax H3 Ref2VA Accelerator** while preserving the existing node class ID for workflow compatibility.
- Preserved the tested `Ref2VA Conservative` thresholds exactly.
- Marked Custom-only threshold/window widgets as Advanced.
- Improved end-of-run console reporting.
- Changed `Ref2VA Balanced` to a maximum of one consecutive cache hit.
