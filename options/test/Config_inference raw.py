"""
Legacy compatibility wrapper.

This file is preserved for older entry points that still reference it.
The active implementation now lives in options.test.Config_inference.
"""

from options.test.Config_inference import Inference_cfg, inference_pa, parse_model_name


if __name__ == "__main__":
    denoise_name, sr_name, v_path, z_zoom = inference_pa()
    print("\n--- GUI Selections ---")
    print(f"Selected Denoise Name: {denoise_name}")
    print(f"Selected SR Name: {sr_name}")
    print(f"Selected Path: {v_path}")
    print(f"Z Zoom Scale: {z_zoom}")
