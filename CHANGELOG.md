# Changelog

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
