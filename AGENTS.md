# Repository Guidelines

## Project Structure & Module Organization

This repository is a ComfyUI custom node package for FaceFusion-style face swapping. The ComfyUI entry point is `__init__.py`, which installs requirements and exports node mappings from `facefusion_api`. Core source lives in `facefusion_api/`: node definitions are in `facefusion_api/nodes/`, model download/loading helpers are in `facefusion_api/models/`, and detection utilities are in `facefusion_api/detection/`. Content moderation support is isolated in `content_filter/`, with its smoke test in `content_filter/test_filter.py`. Runtime model files belong under `models/`; keep only placeholders such as `.gitkeep` in git. Demo media lives in `assets/`.

## Build, Test, and Development Commands

- `python -m pip install -r requirements.txt` installs runtime dependencies for the current ComfyUI Python environment.
- `python -m flake8 facefusion_api content_filter` runs the configured import/order and Python style checks.
- `python -m mypy facefusion_api content_filter` runs strict type checks from `mypy.ini`.
- `cd content_filter && python test_filter.py` runs the existing content-filter smoke test and may download/load ONNX models.

There is no separate build step; develop this repository inside `ComfyUI/custom_nodes/Facefusion_comfyui`, then restart ComfyUI to reload nodes.

## Coding Style & Naming Conventions

Follow `.editorconfig`: LF line endings, final newline, trimmed trailing whitespace, and tab indentation with width 4. Keep Python modules and functions in `snake_case`, classes in `PascalCase`, and ComfyUI node classes descriptive, for example `AdvancedSwapFaceImage`. Reuse the existing node grouping: image behavior in `image_nodes.py`, video behavior in `video_nodes.py`, visual debug nodes in `visualizer_nodes.py`, and shared node imports in `base.py`.

## Testing Guidelines

Add focused tests or smoke scripts close to the feature they validate until a central test suite exists. Name test files `test_*.py` and include any required model/download assumptions in the file header or PR notes. For node changes, verify both importability and a representative ComfyUI workflow manually.

## Commit & Pull Request Guidelines

Recent history uses short imperative commit subjects such as `Add mask type choosing in faceswapper` and `Fix failed inference ghost models`. Keep commits focused and mention the affected node/model path when useful. Pull requests should describe the user-facing change, list verification commands, note model or dependency impacts, and include screenshots or workflow examples for UI-visible ComfyUI node changes.

## Security & Configuration Tips

Do not commit downloaded ONNX models, API tokens, generated videos, or private test images. Prefer local mode with `api_token=-1` for examples unless documenting API-specific behavior.
