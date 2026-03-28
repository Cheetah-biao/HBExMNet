import os
import os.path
from tkinter import *
from tkinter.filedialog import askdirectory
import tkinter as tk
from tkinter import ttk
import numpy as np
import os.path as osp


# 'parse_model_name' function remains the same
def parse_model_name(full_name):
    """
    Parses folder names like 'Mode_Organelle_Task' into a tuple.
    """
    task = None
    rest = full_name

    if full_name.endswith("_Denoise"):
        task = "Denoise"
        rest = full_name[:-len("_Denoise")]
    elif full_name.endswith("_SR"):
        task = "SR"
        rest = full_name[:-len("_SR")]

    if task is None:
        return None

    parts = rest.split('_', 1)
    if len(parts) == 2:
        mode = parts[0]
        organelle = parts[1]
        return (mode, organelle, task, full_name)
    else:
        return None


# -----------------------------------------------------------------
# --- Core Change: Rewriting UI layout for Inference_cfg ---
# -----------------------------------------------------------------
def Inference_cfg():
    win = tk.Tk()
    win.title("Config Inference")
    win.geometry('480x400')  # Keep adjusted height

    # --- Path settings (unchanged) ---
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir, osp.pardir))
    model_root_path = os.path.join(root_path, 'experiments')

    # --- Widget variables ---
    mode_var = tk.StringVar()
    label_tag = tk.StringVar()  # Corresponds to 'Organelle'
    v_path = tk.StringVar()
    denoise_var = tk.BooleanVar(value=True)
    sr_var = tk.BooleanVar(value=True)
    z_zoom_var = tk.StringVar(value='1.0')

    # --- Mapping dictionaries (unchanged) ---
    display_to_full_map = {}
    full_to_display_map = {}

    # --- Scan models on startup (unchanged) ---
    master_model_list = []
    all_modes = set()

    if os.path.exists(model_root_path):
        try:
            for item in os.listdir(model_root_path):
                full_path = os.path.join(model_root_path, item)
                if os.path.isdir(full_path):
                    parsed = parse_model_name(item)
                    if parsed:
                        master_model_list.append(parsed)
                        all_modes.add(parsed[0])
        except Exception as e:
            print(f"Error scanning directory {model_root_path}: {e}")

    sorted_modes = sorted(list(all_modes))

    # ---------------------------------------------------
    # --- 1. Apply Dark Pro Theme Styles ---
    # ---------------------------------------------------

    BG_COLOR = "#2B2B2B"  # Darker gray (window/frame background)
    FG_COLOR = "#D3D3D3"  # Light gray (text)
    ENTRY_BG = "#3C3C3C"  # Medium gray (entry background)
    BTN_BG = "#4A4A4A"  # Button background
    BTN_FG = FG_COLOR  # Button text
    BTN_ACTIVE_BG = "#5A5A5A"  # Button active/hover background
    LBL_FRAME_BG = BG_COLOR  # Same as window background
    SELECT_BG = "#005A9E"  # Accent blue (selection background) - slightly darker blue
    SELECT_FG = "#FFFFFF"  # White (selected text)
    BORDER_COLOR = "#555555"  # Control border
    LBL_FRAME_TITLE_FG = FG_COLOR  # LabelFrame title color
    CHECK_INDICATOR = "#777777"  # Checkbutton indicator color

    # --- A. Set window background ---
    win.configure(bg=BG_COLOR)

    # --- B. Define styles ---
    style = ttk.Style()
    style.theme_use('clam')

    # --- C. Configure all widget styles ---

    style.configure('.',
                    background=BG_COLOR,
                    foreground=FG_COLOR,
                    borderwidth=1,
                    bordercolor=BORDER_COLOR)

    style.configure("TFrame", background=BG_COLOR)

    style.configure("TLabelframe",
                    background=LBL_FRAME_BG,
                    bordercolor=BORDER_COLOR,
                    borderwidth=1)
    style.configure("TLabelframe.Label",
                    font=("Segoe UI", 10, "bold"),
                    background=LBL_FRAME_BG,
                    foreground=LBL_FRAME_TITLE_FG)

    style.configure("TLabel",
                    background=LBL_FRAME_BG,  # Match LabelFrame background
                    foreground=FG_COLOR,
                    font=("Segoe UI", 9))

    # Base button style (TButton)
    style.configure("TButton",
                    font=("Segoe UI", 9),
                    background=BTN_BG,
                    foreground=BTN_FG,
                    bordercolor=BORDER_COLOR,
                    padding=(8, 4))  # Adjusted base padding
    style.map("TButton",
              background=[('pressed', BTN_ACTIVE_BG), ('active', BTN_ACTIVE_BG)],
              foreground=[('pressed', SELECT_FG)],
              relief=[('pressed', 'sunken'), ('!pressed', 'flat')])  # Default flat

    # "Start Running" button style (test.TButton)
    style.configure("test.TButton",
                    font=("Segoe UI", 10, "bold"),
                    padding=8)  # Inherits TButton colors, adjusts font and padding
    # Map inherits TButton

    # "Choose" button style (Choose.TButton)
    style.configure("Choose.TButton",
                    padding=(6, 2),  # Specific padding for Choose
                    relief='raised')  # Raised effect
    # Map inherits TButton

    # Entry style
    style.configure("TEntry",
                    fieldbackground=ENTRY_BG,
                    foreground=FG_COLOR,
                    insertcolor=FG_COLOR,  # Cursor color
                    bordercolor=BORDER_COLOR,
                    borderwidth=1,
                    selectbackground=SELECT_BG,
                    selectforeground=SELECT_FG)

    # Combobox style (removed layout override)
    style.configure("TCombobox",
                    fieldbackground=ENTRY_BG,
                    foreground=FG_COLOR,
                    bordercolor=BORDER_COLOR,  # Inner border
                    arrowcolor=FG_COLOR,
                    selectbackground=SELECT_BG,
                    selectforeground=SELECT_FG)
    style.map('TCombobox',
              fieldbackground=[('readonly', ENTRY_BG)],  # Readonly background
              background=[('active', BTN_ACTIVE_BG)],  # Button part active background
              selectbackground=[('focus', SELECT_BG)],  # Focus selection background
              relief=[('focus', 'flat')])  # No extra border on focus

    # Combobox Listbox style
    win.option_add('*TCombobox*Listbox*Background', ENTRY_BG)
    win.option_add('*TCombobox*Listbox*Foreground', FG_COLOR)
    win.option_add('*TCombobox*Listbox*selectBackground', SELECT_BG)  # Selection background
    win.option_add('*TCombobox*Listbox*selectForeground', SELECT_FG)  # Selection text
    win.option_add('*TCombobox*Listbox*font', ("Segoe UI", 9))
    win.option_add('*TCombobox*Listbox*borderWidth', 0)  # No listbox border

    # Checkbutton style
    style.configure("TCheckbutton",
                    background=LBL_FRAME_BG,  # Match LabelFrame background
                    foreground=FG_COLOR,
                    indicatorcolor=CHECK_INDICATOR,  # Color of the check box itself
                    font=("Segoe UI", 9))
    style.map("TCheckbutton",
              indicatorcolor=[('selected', SELECT_BG), ('!selected', CHECK_INDICATOR)],  # Blue when checked
              background=[('active', LBL_FRAME_BG)])  # Keep background on hover

    # --- End Style Changes ---

    # --- 2. Layout (using ttk.Label, adding justify) ---

    # --- A. Main Frame ---
    main_frame = ttk.Frame(win, padding="10 10 10 10")
    main_frame.pack(fill=BOTH, expand=True)

    # --- B. Group 1: Model Selection ---
    model_frame = ttk.LabelFrame(main_frame, text="Model Selection", padding="10 10 10 10")
    model_frame.pack(fill=X, expand=True, pady=(0, 5))

    ttk.Label(model_frame, text="Mode :").grid(column=0, row=0, sticky=tk.W, pady=5, padx=5)
    mode_combobox = ttk.Combobox(model_frame, width=40, textvariable=mode_var, justify='center')  # Centered
    mode_combobox['values'] = sorted_modes
    if sorted_modes:
        mode_combobox.set(sorted_modes[0])
    else:
        mode_combobox.set("No modes found")
    mode_combobox.grid(column=1, row=0, sticky=tk.EW, pady=5, padx=5)

    ttk.Label(model_frame, text="Organelle :").grid(column=0, row=1, sticky=tk.W, pady=5, padx=5)
    label_tag_entered = ttk.Combobox(model_frame, width=40, textvariable=label_tag, justify='center')  # Centered
    label_tag_entered.grid(column=1, row=1, sticky=tk.EW, pady=5, padx=5)
    label_tag_entered.focus()
    model_frame.columnconfigure(1, weight=1)

    # --- C. Group 2: Task Configuration ---
    task_frame = ttk.LabelFrame(main_frame, text="Task Configuration", padding="10 10 10 10")
    task_frame.pack(fill=X, expand=True, pady=5)

    check_frame = ttk.Frame(task_frame)
    check_frame.pack()
    denoise_check = ttk.Checkbutton(check_frame, text="Denoise", variable=denoise_var,
                                    onvalue=True, offvalue=False, command=lambda: update_model_lists())
    denoise_check.pack(side=LEFT, padx=10, pady=5)
    sr_check = ttk.Checkbutton(check_frame, text="SR", variable=sr_var,
                               onvalue=True, offvalue=False, command=lambda: update_model_lists())
    sr_check.pack(side=LEFT, padx=10, pady=5)

    # --- D. Group 3: Input Data ---
    path_frame = ttk.LabelFrame(main_frame, text="Input Data", padding="10 10 10 10")
    path_frame.pack(fill=X, expand=True, pady=5)

    # -- Validation path (Row 0)
    ttk.Label(path_frame, text="Validation path:").grid(column=0, row=0, sticky=tk.W, pady=5, padx=5)
    v_path_Choose = ttk.Entry(path_frame, width=40, textvariable=v_path, justify='center')  # Centered
    v_path_Choose.grid(column=1, row=0, sticky=tk.EW, pady=5, padx=5)

    def select_LR_Path():
        path_lr = askdirectory(title="Please choose the Validation path")
        if path_lr:
            v_path.set(path_lr)

    lr_path_Choose_button = ttk.Button(path_frame, text="Choose", command=select_LR_Path,
                                       style='Choose.TButton')  # Apply style
    lr_path_Choose_button.grid(column=2, row=0, sticky=tk.E, pady=5, padx=5)

    # -- Z Zoom Scale (Row 1)
    ttk.Label(path_frame, text="Z Zoom Scale:").grid(column=0, row=1, sticky=tk.W, pady=5, padx=5)
    z_zoom_entry = ttk.Entry(path_frame, width=10, textvariable=z_zoom_var, justify='center')  # Centered
    z_zoom_entry.grid(column=1, row=1, sticky=tk.W, pady=5, padx=5)  # sticky=W keeps it left-aligned in the grid cell

    path_frame.columnconfigure(1, weight=1)

    # --- E. Group 4: Actions ---
    action_frame = ttk.Frame(main_frame)
    action_frame.pack(fill=X, expand=True, pady=(10, 0))

    def clickMe():
        win.destroy()

    action = ttk.Button(action_frame, text="Start running", command=clickMe, style="test.TButton")
    action.pack()

    # --- Logic function (unchanged) ---
    def update_model_lists(*args):
        mode_filter = mode_var.get()
        show_denoise = denoise_var.get()
        show_sr = sr_var.get()
        display_to_full_map.clear()
        full_to_display_map.clear()
        candidates_with_time = []
        for mode, organelle, task, full_name in master_model_list:
            if mode != mode_filter:
                continue
            if not show_denoise and not show_sr:
                continue
            if (task == "Denoise" and not show_denoise) or \
                    (task == "SR" and not show_sr):
                continue
            full_path = os.path.join(model_root_path, full_name)
            mod_time = os.path.getmtime(full_path)
            candidates_with_time.append((mod_time, organelle, full_name))
        latest_model_display_name = "No matching models found"
        if candidates_with_time:
            candidates_with_time.sort(key=lambda x: x[0])
            latest_model_full_name = candidates_with_time[-1][2]
            organelle_display_names = set()
            for mod_time, organelle_name, full_name in candidates_with_time:
                organelle_display_names.add(organelle_name)
                display_to_full_map[organelle_name] = full_name
                full_to_display_map[full_name] = organelle_name
            display_list = sorted(list(organelle_display_names))
            label_tag_entered['values'] = display_list
            latest_model_display_name = full_to_display_map.get(latest_model_full_name, "Error")
        else:
            label_tag_entered['values'] = []
        label_tag.set(latest_model_display_name)

    # --- Bind events (unchanged) ---
    mode_combobox.bind('<<ComboboxSelected>>', update_model_lists)

    # --- Startup (unchanged) ---
    update_model_lists()
    win.mainloop()

    # --- Return values (unchanged) ---
    selected_mode = mode_var.get()
    selected_organelle = label_tag.get()
    is_denoise_checked = denoise_var.get()
    is_sr_checked = sr_var.get()
    final_denoise_name = None
    final_sr_name = None
    for mode, organelle, task, full_name in master_model_list:
        if mode == selected_mode and organelle == selected_organelle:
            if task == 'Denoise':
                final_denoise_name = full_name
            elif task == 'SR':
                final_sr_name = full_name
    if not is_denoise_checked:
        final_denoise_name = None
    if not is_sr_checked:
        final_sr_name = None

    # Return 4 values
    return final_denoise_name, final_sr_name, v_path.get(), z_zoom_var.get()


# --- 'inference_pa' function (unchanged, but added float conversion back) ---
def inference_pa():
    # Receive and return 4 values
    denoise_name, sr_name, v_path, z_zoom_str = Inference_cfg()
    try:
        z_zoom_float = float(z_zoom_str)  # Convert to float here
    except ValueError:
        print(f"Warning: Invalid Z Zoom Scale '{z_zoom_str}'. Using 1.0.")
        z_zoom_float = 1.0
    return denoise_name, sr_name, v_path, z_zoom_float


if __name__ == '__main__':
    # Receive 4 values and print
    denoise_name, sr_name, v_path, z_zoom = inference_pa()

    print("\n--- GUI Selections ---")
    print(f"Selected Denoise Name: {denoise_name}")
    print(f"Selected SR Name: {sr_name}")
    print(f"Selected Path: {v_path}")
    print(f"Z Zoom Scale: {z_zoom}")  # Print the float value