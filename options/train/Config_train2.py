"""
Legacy compatibility wrapper.

This file is kept only because older local scripts may still import it.
The active implementation now lives in options.train.Config_train.
"""

from options.train.Config_train import chunking_data, tkinter_input


if __name__ == "__main__":
    label, factor, hr_path, lr_path, mr_path, output_file = chunking_data()
    print("--- GUI Selections ---")
    print(f"Constructed Label: {label}")
    print(f"Factor: {factor}")
    print(f"GT Path: {hr_path}")
    print(f"Raw Data Path: {lr_path}")
    print(f"MR Path: {mr_path}")
    print(f"Training Data File: {output_file}")
