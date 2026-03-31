from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askdirectory

from data.chunking import generate_training_data
from utils.project_paths import workspace_path


DEFAULT_TRAIN_ROOT = workspace_path("Train_data")
APP_ICON = Path(__file__).resolve().parents[2] / "assets" / "Fig1.png"

THEME = {
    "bg": "#12161C",
    "panel": "#1C222A",
    "panel_alt": "#141A22",
    "fg": "#E1E8F0",
    "muted": "#96A5B6",
    "entry_bg": "#11161D",
    "entry_border": "#D8E2ED",
    "btn_bg": "#223041",
    "btn_active": "#2E90FF",
    "border": "#394654",
    "select_fg": "#FFFFFF",
    "check": "#5B6978",
}


def _training_data_file_for_task(task: str):
    return workspace_path("data", f"training_data_{task.lower()}.npz")


def _task_defaults(task: str):
    if task == "SR":
        return {"patch_d": "8", "patch_h": "24", "patch_w": "24", "factor": "6"}
    return {"patch_d": "16", "patch_h": "64", "patch_w": "64", "factor": "1"}


def _resolve_denoise_target_path(config):
    if config.get("task") != "Denoise":
        return config.get("hr_path", "")

    if config.get("mr_path"):
        mr_path = Path(config["mr_path"])
        if mr_path.exists():
            return str(mr_path)

    hr_path = Path(config["hr_path"])
    sibling_mr = hr_path.parent / "MR"
    if hr_path.name.upper() == "GT" and sibling_mr.exists():
        return str(sibling_mr)

    return str(hr_path)


def _normalize_config(selection=None):
    selection = selection or {}
    task = selection.get("task", "Denoise")
    defaults = _task_defaults(task)
    if DEFAULT_TRAIN_ROOT.exists() and task == "Denoise" and (DEFAULT_TRAIN_ROOT / "MR").exists():
        default_gt = DEFAULT_TRAIN_ROOT / "MR"
    else:
        default_gt = DEFAULT_TRAIN_ROOT / "GT" if DEFAULT_TRAIN_ROOT.exists() else None
    default_lr = DEFAULT_TRAIN_ROOT / "LR" if DEFAULT_TRAIN_ROOT.exists() else None
    config = {
        "mode": selection.get("mode", "TIM"),
        "organelle": selection.get("organelle", "Tub"),
        "task": task,
        "hr_path": str(selection.get("hr_path", default_gt if default_gt is not None and default_gt.exists() else "")),
        "lr_path": str(selection.get("lr_path", default_lr if default_lr is not None and default_lr.exists() else "")),
        "mr_path": "",
        "patch_d": str(selection.get("patch_d", defaults["patch_d"])),
        "patch_h": str(selection.get("patch_h", defaults["patch_h"])),
        "patch_w": str(selection.get("patch_w", defaults["patch_w"])),
        "factor": str(selection.get("factor", defaults["factor"])),
    }
    return config


def _validate_config(config):
    hr_path = Path(config["hr_path"])
    lr_path = Path(config["lr_path"])
    if not config["mode"] or not config["organelle"] or not config["task"]:
        raise ValueError("Mode, organelle, and task must be selected.")
    if not hr_path.exists():
        raise FileNotFoundError(f"GT path does not exist: {hr_path}")
    if not lr_path.exists():
        raise FileNotFoundError(f"Raw data path does not exist: {lr_path}")

    for key in ("patch_d", "patch_h", "patch_w", "factor"):
        value = int(config[key])
        if value <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    return config


def _apply_dark_theme(win, style):
    win.configure(bg=THEME["bg"])
    style.theme_use("clam")
    style.configure(".", background=THEME["bg"], foreground=THEME["fg"])
    style.configure("TFrame", background=THEME["bg"])
    style.configure("Card.TLabelframe", background=THEME["panel"], bordercolor=THEME["border"], borderwidth=1, relief="solid")
    style.configure("Card.TLabelframe.Label", font=("Segoe UI Semibold", 10), background=THEME["panel"], foreground=THEME["fg"])
    style.configure("TLabel", background=THEME["bg"], foreground=THEME["fg"], font=("Segoe UI", 10))
    style.configure("Card.TLabel", background=THEME["panel"], foreground=THEME["fg"], font=("Segoe UI", 10))
    style.configure("PanelMuted.TLabel", background=THEME["panel"], foreground=THEME["muted"], font=("Segoe UI", 9))
    style.configure(
        "TButton",
        font=("Segoe UI Semibold", 9),
        background=THEME["btn_bg"],
        foreground=THEME["fg"],
        bordercolor=THEME["border"],
        padding=(12, 6),
    )
    style.map(
        "TButton",
        background=[("pressed", THEME["btn_active"]), ("active", THEME["btn_active"])],
        foreground=[("pressed", THEME["select_fg"]), ("active", THEME["select_fg"])],
        relief=[("pressed", "sunken"), ("!pressed", "flat")],
    )
    style.configure("Choose.TButton", padding=(18, 6))
    style.configure("Start.TButton", font=("Segoe UI Semibold", 10), padding=(18, 10))
    style.configure(
        "TCombobox",
        fieldbackground=THEME["entry_bg"],
        foreground=THEME["fg"],
        bordercolor=THEME["entry_border"],
        arrowcolor=THEME["fg"],
        selectbackground=THEME["btn_active"],
        selectforeground=THEME["select_fg"],
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", THEME["entry_bg"])],
        background=[("active", THEME["btn_active"])],
        selectbackground=[("focus", THEME["btn_active"])],
        selectforeground=[("focus", THEME["select_fg"])],
    )


def _apply_window_icon(win):
    if not APP_ICON.exists():
        return
    try:
        icon_image = tk.PhotoImage(file=str(APP_ICON))
        win.iconphoto(True, icon_image)
        win._hbexmnet_icon = icon_image
    except tk.TclError:
        pass


def _make_hero(parent, title, subtitle):
    hero = tk.Frame(parent, bg=THEME["panel_alt"], highlightbackground=THEME["border"], highlightthickness=1)
    hero.grid_columnconfigure(0, weight=1)
    tk.Label(hero, text=title, bg=THEME["panel_alt"], fg=THEME["fg"], font=("Segoe UI Semibold", 18)).grid(
        row=0, column=0, sticky="w", padx=18, pady=(16, 4)
    )
    tk.Label(
        hero,
        text=subtitle,
        bg=THEME["panel_alt"],
        fg=THEME["muted"],
        font=("Segoe UI", 10),
        justify=tk.LEFT,
        wraplength=820,
    ).grid(row=1, column=0, sticky="w", padx=18, pady=(0, 16))
    return hero


def _make_framed_entry(parent, textvariable, width=None, justify="left"):
    shell = tk.Frame(parent, bg=THEME["entry_border"], highlightthickness=0)
    entry = tk.Entry(
        shell,
        textvariable=textvariable,
        bd=0,
        relief=tk.FLAT,
        bg=THEME["entry_bg"],
        fg=THEME["fg"],
        insertbackground=THEME["fg"],
        font=("Segoe UI", 11),
        justify=justify,
        width=width,
    )
    entry.pack(fill=tk.BOTH, expand=True, padx=2, pady=2, ipady=7)
    return shell, entry


def _make_text_panel(parent, textvariable):
    panel = tk.Frame(parent, bg=THEME["panel"], highlightbackground=THEME["border"], highlightthickness=1)
    label = tk.Label(
        panel,
        textvariable=textvariable,
        bg=THEME["panel"],
        fg=THEME["fg"],
        font=("Segoe UI", 10),
        justify=tk.LEFT,
        anchor="nw",
        wraplength=860,
        padx=12,
        pady=10,
    )
    label.pack(fill=tk.BOTH, expand=True)
    return panel


def _launch_gui():
    result = {}
    win = tk.Tk()
    win.title("HBExMNet Training")
    win.geometry("980x780")
    win.minsize(930, 720)
    _apply_window_icon(win)

    style = ttk.Style()
    _apply_dark_theme(win, style)

    defaults = _normalize_config()

    mode_var = tk.StringVar(value=defaults["mode"])
    organelle_var = tk.StringVar(value=defaults["organelle"])
    task_var = tk.StringVar(value=defaults["task"])
    hr_var = tk.StringVar(value=defaults["hr_path"])
    lr_var = tk.StringVar(value=defaults["lr_path"])
    target_label_var = tk.StringVar(value="Target path" if defaults["task"] == "Denoise" else "GT path")
    patch_d_var = tk.StringVar(value=defaults["patch_d"])
    patch_h_var = tk.StringVar(value=defaults["patch_h"])
    patch_w_var = tk.StringVar(value=defaults["patch_w"])
    factor_var = tk.StringVar(value=defaults["factor"])
    status_var = tk.StringVar(value="Waiting for configuration...")
    summary_var = tk.StringVar()

    def apply_task_defaults(*_args):
        task_defaults = _task_defaults(task_var.get())
        patch_d_var.set(task_defaults["patch_d"])
        patch_h_var.set(task_defaults["patch_h"])
        patch_w_var.set(task_defaults["patch_w"])
        factor_var.set(task_defaults["factor"])
        target_label_var.set("Target path" if task_var.get() == "Denoise" else "GT path")
        refresh_status()

    def choose_hr_path():
        title = "Select the denoise target directory" if task_var.get() == "Denoise" else "Select the GT directory"
        path = askdirectory(title=title)
        if path:
            hr_var.set(path)
            refresh_status()

    def choose_lr_path():
        path = askdirectory(title="Select the raw data directory")
        if path:
            lr_var.set(path)
            refresh_status()

    def refresh_status(*_args):
        output_file = _training_data_file_for_task(task_var.get())
        summary_lines = [
            f"Training root : {DEFAULT_TRAIN_ROOT}",
            f"Output file   : {output_file}",
            f"Task          : {task_var.get()}",
        ]
        if task_var.get() == "Denoise":
            summary_lines.append(
                f"Supervision   : LR -> {Path(_resolve_denoise_target_path(_normalize_config({'task': task_var.get(), 'hr_path': hr_var.get()}))).name}"
            )
        else:
            summary_lines.append("Supervision   : LR + GT-downsampled mid -> GT")
        summary_var.set("\n".join(summary_lines))

        try:
            _validate_config(
                _normalize_config(
                    {
                        "mode": mode_var.get(),
                        "organelle": organelle_var.get(),
                        "task": task_var.get(),
                        "hr_path": hr_var.get(),
                        "lr_path": lr_var.get(),
                        "patch_d": patch_d_var.get(),
                        "patch_h": patch_h_var.get(),
                        "patch_w": patch_w_var.get(),
                        "factor": factor_var.get(),
                    }
                )
            )
            status_var.set(f"Ready to build paired patches into {output_file} and start the selected training stage.")
            start_button.state(["!disabled"])
        except Exception as exc:
            status_var.set(str(exc))
            start_button.state(["disabled"])

    def start():
        result.update(
            _normalize_config(
                {
                    "mode": mode_var.get(),
                    "organelle": organelle_var.get(),
                    "task": task_var.get(),
                    "hr_path": hr_var.get(),
                    "lr_path": lr_var.get(),
                    "patch_d": patch_d_var.get(),
                    "patch_h": patch_h_var.get(),
                    "patch_w": patch_w_var.get(),
                    "factor": factor_var.get(),
                }
            )
        )
        win.destroy()

    def on_enter(_event=None):
        if start_button.instate(["!disabled"]):
            start()

    main = ttk.Frame(win, padding=16)
    main.pack(fill=tk.BOTH, expand=True)
    main.columnconfigure(0, weight=1)
    main.columnconfigure(1, weight=1)

    hero = _make_hero(
        main,
        "HBExMNet Training",
        "Build paired training patches from your LR and GT folders, then launch the selected network stage with matching defaults.",
    )
    hero.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))

    exp_frame = ttk.LabelFrame(main, text="Experiment Profile", padding=12, style="Card.TLabelframe")
    exp_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
    exp_frame.columnconfigure(1, weight=1)
    exp_frame.columnconfigure(3, weight=1)
    exp_frame.columnconfigure(5, weight=1)
    ttk.Label(exp_frame, text="Mode", style="Card.TLabel").grid(row=0, column=0, sticky="w", padx=4, pady=6)
    ttk.Combobox(exp_frame, textvariable=mode_var, values=["TIM", "SDCM"], state="readonly", justify="center").grid(row=0, column=1, sticky="ew", padx=4, pady=6)
    ttk.Label(exp_frame, text="Organelle", style="Card.TLabel").grid(row=0, column=2, sticky="w", padx=4, pady=6)
    ttk.Combobox(exp_frame, textvariable=organelle_var, values=["Tub", "Rab7", "Tomm20", "ER"], justify="center").grid(row=0, column=3, sticky="ew", padx=4, pady=6)
    ttk.Label(exp_frame, text="Task", style="Card.TLabel").grid(row=0, column=4, sticky="w", padx=4, pady=6)
    ttk.Combobox(exp_frame, textvariable=task_var, values=["Denoise", "SR"], state="readonly", justify="center").grid(row=0, column=5, sticky="ew", padx=4, pady=6)

    path_frame = ttk.LabelFrame(main, text="Training Data", padding=12, style="Card.TLabelframe")
    path_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
    path_frame.columnconfigure(0, weight=1)
    ttk.Label(
        path_frame,
        text="Provide LR and GT folder paths. For Denoise, the launcher uses stage-1 supervision automatically. For SR, the middle supervision is generated implicitly by downsampling GT to the LR size.",
        style="PanelMuted.TLabel",
        wraplength=860,
        justify=tk.LEFT,
    ).grid(row=0, column=0, sticky="w", padx=4, pady=(0, 10))

    grid = ttk.Frame(path_frame)
    grid.grid(row=1, column=0, sticky="ew")
    grid.columnconfigure(1, weight=1)

    ttk.Label(grid, textvariable=target_label_var, style="Card.TLabel").grid(row=0, column=0, sticky="w", padx=4, pady=6)
    hr_shell, _ = _make_framed_entry(grid, hr_var, justify="left")
    hr_shell.grid(row=0, column=1, sticky="ew", padx=(4, 10), pady=6)
    ttk.Button(grid, text="Choose", command=choose_hr_path, style="Choose.TButton").grid(row=0, column=2, pady=6)

    ttk.Label(grid, text="Raw data path", style="Card.TLabel").grid(row=1, column=0, sticky="w", padx=4, pady=6)
    lr_shell, _ = _make_framed_entry(grid, lr_var, justify="left")
    lr_shell.grid(row=1, column=1, sticky="ew", padx=(4, 10), pady=6)
    ttk.Button(grid, text="Choose", command=choose_lr_path, style="Choose.TButton").grid(row=1, column=2, pady=6)

    params_frame = ttk.LabelFrame(main, text="Patch Parameters", padding=12, style="Card.TLabelframe")
    params_frame.grid(row=3, column=0, sticky="nsew", padx=(0, 8), pady=(0, 10))
    for idx in range(4):
        params_frame.columnconfigure(idx, weight=1)
    ttk.Label(params_frame, text="Patch D", style="Card.TLabel").grid(row=0, column=0, sticky="w", padx=4, pady=6)
    ttk.Label(params_frame, text="Patch H", style="Card.TLabel").grid(row=0, column=1, sticky="w", padx=4, pady=6)
    ttk.Label(params_frame, text="Patch W", style="Card.TLabel").grid(row=0, column=2, sticky="w", padx=4, pady=6)
    ttk.Label(params_frame, text="Scale factor", style="Card.TLabel").grid(row=0, column=3, sticky="w", padx=4, pady=6)
    pd_shell, _ = _make_framed_entry(params_frame, patch_d_var, justify="center")
    ph_shell, _ = _make_framed_entry(params_frame, patch_h_var, justify="center")
    pw_shell, _ = _make_framed_entry(params_frame, patch_w_var, justify="center")
    sf_shell, _ = _make_framed_entry(params_frame, factor_var, justify="center")
    pd_shell.grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 4))
    ph_shell.grid(row=1, column=1, sticky="ew", padx=4, pady=(0, 4))
    pw_shell.grid(row=1, column=2, sticky="ew", padx=4, pady=(0, 4))
    sf_shell.grid(row=1, column=3, sticky="ew", padx=4, pady=(0, 4))
    ttk.Label(
        params_frame,
        text="Task defaults are applied automatically, but you can override them before building the dataset.",
        style="PanelMuted.TLabel",
        wraplength=400,
        justify=tk.LEFT,
    ).grid(row=2, column=0, columnspan=4, sticky="w", padx=4, pady=(8, 0))

    summary_frame = ttk.LabelFrame(main, text="Resolved Output", padding=12, style="Card.TLabelframe")
    summary_frame.grid(row=3, column=1, sticky="nsew", padx=(8, 0), pady=(0, 10))
    summary_frame.columnconfigure(0, weight=1)
    summary_panel = _make_text_panel(summary_frame, summary_var)
    summary_panel.grid(row=0, column=0, sticky="ew")

    status_frame = ttk.LabelFrame(main, text="Run Status", padding=12, style="Card.TLabelframe")
    status_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")
    status_frame.columnconfigure(0, weight=1)
    status_panel = _make_text_panel(status_frame, status_var)
    status_panel.grid(row=0, column=0, sticky="ew")
    start_button = ttk.Button(status_frame, text="Start running", style="Start.TButton", command=start)
    start_button.grid(row=0, column=1, padx=(18, 0), sticky="e")

    task_var.trace_add("write", apply_task_defaults)
    for variable in (mode_var, organelle_var, hr_var, lr_var, patch_d_var, patch_h_var, patch_w_var, factor_var):
        variable.trace_add("write", refresh_status)

    win.bind("<Return>", on_enter)
    win.bind("<Escape>", lambda _event: win.destroy())
    refresh_status()
    win.mainloop()
    return result


def tkinter_input(selection=None):
    if selection is not None:
        return _validate_config(_normalize_config(selection))

    config = _launch_gui()
    if not config:
        raise RuntimeError("Training configuration was cancelled.")
    return _validate_config(config)


def chunking_data(selection=None):
    config = tkinter_input(selection)
    patch_size = [int(config["patch_d"]), int(config["patch_h"]), int(config["patch_w"])]
    factor = int(config["factor"])
    label_tag = f"{config['mode']}_{config['organelle']}_{config['task']}"
    output_file = _training_data_file_for_task(config["task"])
    primary_target_path = config["hr_path"]
    mr_path = config["mr_path"] or None

    if config["task"] == "Denoise":
        primary_target_path = _resolve_denoise_target_path(config)
        mr_path = None

    generate_training_data(
        hr_path=primary_target_path,
        lr_path=config["lr_path"],
        mr_path=mr_path,
        output_file=str(output_file),
        patch_size=patch_size,
        factor=factor,
        overlap=0.5,
    )

    return label_tag, factor, primary_target_path, config["lr_path"], config["mr_path"], str(output_file)


if __name__ == "__main__":
    label, factor, hr_path, lr_path, _mr_path, output_file = chunking_data()
    print("--- GUI Selections ---")
    print(f"Constructed Label: {label}")
    print(f"Factor: {factor}")
    print(f"GT Path: {hr_path}")
    print(f"Raw Data Path: {lr_path}")
    print(f"Training Data File: {output_file}")
