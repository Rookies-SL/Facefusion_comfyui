"""
ONNX Runtime diagnostics shared by local model loaders.
"""
import os
from typing import Any, Sequence

import onnxruntime as ort


ONNX_DEBUG_ENV = 'FACEFUSION_ONNX_DEBUG'
_ENABLED_VALUES = {'1', 'true', 'yes', 'on'}


def is_onnx_debug_enabled() -> bool:
    """Return whether ONNX Runtime provider diagnostics should be printed."""
    return os.environ.get(ONNX_DEBUG_ENV, '').strip().lower() in _ENABLED_VALUES


def log_onnx_session(label: str, requested_providers: Sequence[Any], session: Any) -> None:
    """Print provider diagnostics for one ONNX Runtime session."""
    if not is_onnx_debug_enabled():
        return

    try:
        available_providers = ort.get_available_providers()
    except Exception as e:
        available_providers = [f'unavailable: {e}']

    try:
        actual_providers = session.get_providers()
    except Exception as e:
        actual_providers = [f'unavailable: {e}']

    try:
        provider_options = session.get_provider_options()
    except Exception as e:
        provider_options = {'unavailable': str(e)}

    print(f"[ONNXRuntime][{label}] available providers: {available_providers}")
    print(f"[ONNXRuntime][{label}] requested providers: {requested_providers}")
    print(f"[ONNXRuntime][{label}] actual providers: {actual_providers}")
    print(f"[ONNXRuntime][{label}] provider options: {provider_options}")
