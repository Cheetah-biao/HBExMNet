from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askdirectory

from utils.project_paths import workspace_path


EXPERIMENTS_ROOT = workspace_path("experiments")
DEFAULT_DATA_ROOT = workspace_path("data")
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
    "accent": "#2E90FF",
    "border": "#394654",
    "select_fg": "#FFFFFF",
    "check": "#5B6978",
}


@dataclass
class ModelEntry:
    mode: str
    organelle: str
    task: str
    folder_name: str
    folder_path: Path
    modified_time: float


def parse_model_name(full_name):
    task = None
    stem = full_name
    if full_name.endswith("_Denoise"):
        task = "Denoise"
        stem = full_name[: -len("_Denoise")]
    elif full_name.endswith("_SR"):
        task = "SR"
        stem = full_name[: -len("_SR")]

    if task is None:
        return None

    parts = stem.split("_", 1)
    if len(parts) != 2:
        return None
    return parts[0], parts[1], task, full_name


def _scan_models():
    models = []
    if not EXPERIMENTS_ROOT.exists():
        return models

    for item in sorted(EXPERIMENTS_ROOT.iterdir(), key=lambda path: path.name.lower()):
        if not item.is_dir():
            continue
        parsed = parse_model_name(item.name)
        if not parsed:
            continue
        mode, organelle, task, folder_name = parsed
        models.append(
            ModelEntry(
                mode=mode,
                organelle=organelle,
                task=task,
                folder_name=folder_name,
                folder_path=item,
                modified_time=item.stat().st_mtime,
            )
        )
    return models


def _latest_matching_model(models, mode, organelle, task):
    candidates = [entry for entry in models if entry.mode == mode and entry.organelle == organelle and entry.task == task]
    if not candidates:
        return None
    return max(candidates, key=lambda entry: entry.modified_time)


def _normalize_selection(selection=None):
    selection = selection or {}
    xy_nm = selection.get("xy_nm")
    z_nm = selection.get("z_nm")
    if xy_nm is None:
        xy_nm = 65.0
    if z_nm is None:
        z_nm = 65.0
    z_zoom = selection.get("z_zoom")
    if z_zoom is None:
        z_zoom = float(z_nm) / float(xy_nm) if selection.get("sr", True) else 1.0
    return {
        "mode": selection.get("mode", ""),
        "organelle": selection.get("organelle", ""),
        "denoise": bool(selection.get("denoise", False)),
        "sr": bool(selection.get("sr", True)),
        "input_path": str(selection.get("input_path", DEFAULT_DATA_ROOT if DEFAULT_DATA_ROOT.exists() else "")),
        "xy_nm": str(xy_nm),
        "z_nm": str(z_nm),
        "z_zoom": str(z_zoom),
    }


def _resolve_selection(selection=None):
    models = _scan_models()
    normalized = _normalize_selection(selection)

    if not normalized["denoise"] and not normalized["sr"]:
        raise ValueError("At least one task must be selected for inference.")

    if not normalized["mode"]:
        available_modes = sorted({entry.mode for entry in models})
        if available_modes:
            normalized["mode"] = available_modes[0]
    if not normalized["organelle"]:
        matching_organelles = sorted({entry.organelle for entry in models if entry.mode == normalized["mode"]})
        if matching_organelles:
            normalized["organelle"] = matching_organelles[0]

    input_path = Path(normalized["input_path"])
    if not input_path.exists():
        raise FileNotFoundError(f"Inference input path does not exist: {input_path}")

    denoise_name = None
    sr_name = None
    if normalized["denoise"]:
        entry = _latest_matching_model(models, normalized["mode"], normalized["organelle"], "Denoise")
        if entry is None:
            raise FileNotFoundError(
                f"No Denoise experiment was found for {normalized['mode']}_{normalized['organelle']} under {EXPERIMENTS_ROOT}."
            )
        denoise_name = entry.folder_name
    if normalized["sr"]:
        entry = _latest_matching_model(models, normalized["mode"], normalized["organelle"], "SR")
        if entry is None:
            raise FileNotFoundError(
                f"No SR experiment was found for {normalized['mode']}_{normalized['organelle']} under {EXPERIMENTS_ROOT}."
            )
        sr_name = entry.folder_name

    xy_nm = float(normalized["xy_nm"])
    z_nm = float(normalized["z_nm"])
    if xy_nm <= 0 or z_nm <= 0:
        raise ValueError("XY and Z pixel sizes must be positive numbers.")

    z_zoom = float(normalized["z_zoom"])
    if normalized["sr"]:
        z_zoom = z_nm / xy_nm
    else:
        z_zoom = 1.0

    return denoise_name, sr_name, str(input_path), float(z_zoom)


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
    style.configure("Muted.TLabel", background=THEME["bg"], foreground=THEME["muted"], font=("Segoe UI", 9))
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
    style.configure("TCheckbutton", background=THEME["bg"], foreground=THEME["fg"], indicatorcolor=THEME["check"], font=("Segoe UI", 10))
    style.map("TCheckbutton", indicatorcolor=[("selected", THEME["btn_active"]), ("!selected", THEME["check"])], background=[("active", THEME["bg"])])
    win.option_add("*TCombobox*Listbox*Background", THEME["entry_bg"])
    win.option_add("*TCombobox*Listbox*Foreground", THEME["fg"])
    win.option_add("*TCombobox*Listbox*selectBackground", THEME["btn_active"])
    win.option_add("*TCombobox*Listbox*selectForeground", THEME["select_fg"])
    win.option_add("*TCombobox*Listbox*font", ("Segoe UI", 10))


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
        wraplength=760,
    ).grid(row=1, column=0, sticky="w", padx=18, pady=(0, 16))
    return hero


def _make_framed_entry(parent, textvariable, width=None, readonly=False, justify="left"):
    shell = tk.Frame(parent, bg=THEME["entry_border"], highlightthickness=0)
    state = "readonly" if readonly else "normal"
    entry = tk.Entry(
        shell,
        textvariable=textvariable,
        bd=0,
        relief=tk.FLAT,
        bg=THEME["entry_bg"],
        fg=THEME["fg"],
        insertbackground=THEME["fg"],
        readonlybackground=THEME["entry_bg"],
        disabledbackground=THEME["entry_bg"],
        disabledforeground=THEME["fg"],
        font=("Segoe UI", 11),
        justify=justify,
        state=state,
        width=width,
    )
    entry.pack(fill=tk.BOTH, expand=True, padx=2, pady=2, ipady=7)
    return shell, entry


def _make_text_panel(parent, textvariable, height=4, mono=False):
    panel = tk.Frame(parent, bg=THEME["panel"], highlightbackground=THEME["border"], highlightthickness=1)
    label = tk.Label(
        panel,
        textvariable=textvariable,
        bg=THEME["panel"],
        fg=THEME["fg"],
        font=("Consolas", 10) if mono else ("Segoe UI", 10),
        justify=tk.LEFT,
        anchor="nw",
        wraplength=860,
        padx=12,
        pady=10,
    )
    label.pack(fill=tk.BOTH, expand=True)
    return panel


def _launch_gui():
    models = _scan_models()
    if not models:
        raise FileNotFoundError(f"No experiment folders were found under {EXPERIMENTS_ROOT}.")

    result = {}
    win = tk.Tk()
    win.title("HBExMNet Inference")
    win.geometry("940x720")
    win.minsize(900, 680)
    _apply_window_icon(win)

    style = ttk.Style()
    _apply_dark_theme(win, style)

    available_modes = sorted({entry.mode for entry in models})

    mode_var = tk.StringVar(value=available_modes[0] if available_modes else "")
    organelle_var = tk.StringVar()
    denoise_var = tk.BooleanVar(value=False)
    sr_var = tk.BooleanVar(value=True)
    input_path_var = tk.StringVar(value=str(DEFAULT_DATA_ROOT) if DEFAULT_DATA_ROOT.exists() else "")
    xy_nm_var = tk.StringVar(value="65")
    z_nm_var = tk.StringVar(value="65")
    z_zoom_var = tk.StringVar(value=f"{65 / 65:.4f}")
    status_var = tk.StringVar(value="Waiting for configuration...")
    model_hint_var = tk.StringVar()
    asset_note_var = tk.StringVar()

    def organelle_options():
        organelles = []
        for organelle in sorted({entry.organelle for entry in models if entry.mode == mode_var.get()}):
            if denoise_var.get() and _latest_matching_model(models, mode_var.get(), organelle, "Denoise") is None:
                continue
            if sr_var.get() and _latest_matching_model(models, mode_var.get(), organelle, "SR") is None:
                continue
            organelles.append(organelle)
        return organelles

    def browse_input_path():
        path = askdirectory(title="Select the inference input directory")
        if path:
            input_path_var.set(path)
            refresh_status()

    def selected_model_summary():
        organelle = organelle_var.get()
        labels = []
        if denoise_var.get():
            denoise_entry = _latest_matching_model(models, mode_var.get(), organelle, "Denoise")
            labels.append(f"Denoise : {denoise_entry.folder_name if denoise_entry else 'missing'}")
        if sr_var.get():
            sr_entry = _latest_matching_model(models, mode_var.get(), organelle, "SR")
            labels.append(f"SR      : {sr_entry.folder_name if sr_entry else 'missing'}")
        if not labels:
            return "No tasks selected."
        return "\n".join(labels)

    def refresh_organelle_options(*_args):
        values = organelle_options()
        organelle_combo["values"] = values
        if organelle_var.get() not in values:
            organelle_var.set(values[0] if values else "")
        refresh_status()

    def refresh_status(*_args):
        try:
            xy_nm = float(xy_nm_var.get())
            z_nm = float(z_nm_var.get())
            if xy_nm <= 0 or z_nm <= 0:
                raise ValueError
            computed_zoom = z_nm / xy_nm if sr_var.get() else 1.0
            z_zoom_var.set(f"{computed_zoom:.4f}")
        except ValueError:
            z_zoom_var.set("invalid")

        model_hint_var.set(selected_model_summary())
        asset_note_var.set(
            f"Experiment root: {EXPERIMENTS_ROOT}\n"
            f"Bundled profiles: {len(sorted({(entry.mode, entry.organelle) for entry in models}))}\n"
            f"Inference engine: Python / PyTorch"
        )
        try:
            _resolve_selection(
                {
                    "mode": mode_var.get(),
                    "organelle": organelle_var.get(),
                    "denoise": denoise_var.get(),
                    "sr": sr_var.get(),
                    "input_path": input_path_var.get(),
                    "xy_nm": xy_nm_var.get(),
                    "z_nm": z_nm_var.get(),
                }
            )
            status_var.set("Ready to run. The selected experiment folders and input path are valid.")
            start_button.state(["!disabled"])
        except Exception as exc:
            status_var.set(str(exc))
            start_button.state(["disabled"])

    def start():
        denoise_name, sr_name, input_path, z_zoom = _resolve_selection(
            {
                "mode": mode_var.get(),
                "organelle": organelle_var.get(),
                "denoise": denoise_var.get(),
                "sr": sr_var.get(),
                "input_path": input_path_var.get(),
                "xy_nm": xy_nm_var.get(),
                "z_nm": z_nm_var.get(),
            }
        )
        result.update(
            {
                "denoise_name": denoise_name,
                "sr_name": sr_name,
                "input_path": input_path,
                "z_zoom": z_zoom,
                "xy_nm": float(xy_nm_var.get()),
                "z_nm": float(z_nm_var.get()),
            }
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
        "HBExMNet Inference",
        "Choose one TIFF folder to restore, select the model profile, and keep voxel sizes in nanometers.",
    )
    hero.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))

    input_frame = ttk.LabelFrame(main, text="Input Data", padding=12, style="Card.TLabelframe")
    input_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
    input_frame.columnconfigure(0, weight=1)

    ttk.Label(input_frame, text="Input TIFF folder", style="Card.TLabel").grid(row=0, column=0, sticky="w", padx=4, pady=(0, 6))
    path_row = ttk.Frame(input_frame)
    path_row.grid(row=1, column=0, sticky="ew")
    path_row.columnconfigure(0, weight=1)
    path_shell, _ = _make_framed_entry(path_row, input_path_var, justify="left")
    path_shell.grid(row=0, column=0, sticky="ew", padx=(4, 10))
    ttk.Button(path_row, text="Choose", command=browse_input_path, style="Choose.TButton").grid(row=0, column=1, sticky="e")
    ttk.Label(
        input_frame,
        text="Select one folder containing one or more .tif / .tiff stacks. A timestamped result folder will be created beside it.",
        style="PanelMuted.TLabel",
        wraplength=850,
        justify=tk.LEFT,
    ).grid(row=2, column=0, sticky="w", padx=4, pady=(10, 0))

    profile_frame = ttk.LabelFrame(main, text="Model Profile", padding=12, style="Card.TLabelframe")
    profile_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 8), pady=(0, 10))
    profile_frame.columnconfigure(1, weight=1)
    profile_frame.columnconfigure(3, weight=1)

    ttk.Label(profile_frame, text="Mode", style="Card.TLabel").grid(row=0, column=0, sticky="w", padx=4, pady=6)
    mode_combo = ttk.Combobox(profile_frame, textvariable=mode_var, values=available_modes, state="readonly", justify="center")
    mode_combo.grid(row=0, column=1, sticky="ew", padx=4, pady=6)
    ttk.Label(profile_frame, text="Organelle", style="Card.TLabel").grid(row=0, column=2, sticky="w", padx=4, pady=6)
    organelle_combo = ttk.Combobox(profile_frame, textvariable=organelle_var, state="readonly", justify="center")
    organelle_combo.grid(row=0, column=3, sticky="ew", padx=4, pady=6)

    checks = ttk.Frame(profile_frame)
    checks.grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 4))
    ttk.Checkbutton(checks, text="Denoise", variable=denoise_var).pack(side=tk.LEFT, padx=(4, 18))
    ttk.Checkbutton(checks, text="Super-resolution", variable=sr_var).pack(side=tk.LEFT, padx=4)

    ttk.Label(profile_frame, text="Resolved models", style="Card.TLabel").grid(row=2, column=0, sticky="nw", padx=4, pady=(10, 4))
    model_panel = _make_text_panel(profile_frame, model_hint_var, mono=True)
    model_panel.grid(row=2, column=1, columnspan=3, sticky="ew", padx=4, pady=(10, 0))

    sampling_frame = ttk.LabelFrame(main, text="Sampling And Output", padding=12, style="Card.TLabelframe")
    sampling_frame.grid(row=2, column=1, sticky="nsew", padx=(8, 0), pady=(0, 10))
    for idx in range(3):
        sampling_frame.columnconfigure(idx, weight=1)

    ttk.Label(sampling_frame, text="Input XY pixel size (nm)", style="Card.TLabel").grid(row=0, column=0, sticky="w", padx=4, pady=6)
    ttk.Label(sampling_frame, text="Input Z pixel size (nm)", style="Card.TLabel").grid(row=0, column=1, sticky="w", padx=4, pady=6)
    ttk.Label(sampling_frame, text="Z / XY", style="Card.TLabel").grid(row=0, column=2, sticky="w", padx=4, pady=6)

    xy_shell, _ = _make_framed_entry(sampling_frame, xy_nm_var, justify="center")
    xy_shell.grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 4))
    z_shell, _ = _make_framed_entry(sampling_frame, z_nm_var, justify="center")
    z_shell.grid(row=1, column=1, sticky="ew", padx=4, pady=(0, 4))
    zoom_shell, _ = _make_framed_entry(sampling_frame, z_zoom_var, readonly=True, justify="center")
    zoom_shell.grid(row=1, column=2, sticky="ew", padx=4, pady=(0, 4))

    ttk.Label(
        sampling_frame,
        text="SR output remains fixed at 6x in X / Y / Z, with no final downsampling.",
        style="PanelMuted.TLabel",
        wraplength=360,
        justify=tk.LEFT,
    ).grid(row=2, column=0, columnspan=3, sticky="w", padx=4, pady=(8, 0))

    asset_frame = ttk.LabelFrame(main, text="Resolved Assets", padding=12, style="Card.TLabelframe")
    asset_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
    asset_frame.columnconfigure(0, weight=1)
    asset_panel = _make_text_panel(asset_frame, asset_note_var, mono=False)
    asset_panel.grid(row=0, column=0, sticky="ew")

    status_frame = ttk.LabelFrame(main, text="Run Status", padding=12, style="Card.TLabelframe")
    status_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")
    status_frame.columnconfigure(0, weight=1)
    status_panel = _make_text_panel(status_frame, status_var, mono=False)
    status_panel.grid(row=0, column=0, sticky="ew")
    start_button = ttk.Button(status_frame, text="Start running", style="Start.TButton", command=start)
    start_button.grid(row=0, column=1, padx=(18, 0), sticky="e")

    mode_var.trace_add("write", refresh_organelle_options)
    for variable in (organelle_var, denoise_var, sr_var, input_path_var, xy_nm_var, z_nm_var):
        variable.trace_add("write", refresh_status)

    win.bind("<Return>", on_enter)
    win.bind("<Escape>", lambda _event: win.destroy())
    refresh_organelle_options()
    win.mainloop()
    return result


def Inference_cfg(selection=None):
    if selection is not None:
        return _resolve_selection(selection)

    config = _launch_gui()
    if not config:
        raise RuntimeError("Inference configuration was cancelled.")
    return config["denoise_name"], config["sr_name"], config["input_path"], config["z_zoom"]


def inference_pa(selection=None):
    denoise_name, sr_name, v_path, z_zoom = Inference_cfg(selection=selection)
    return denoise_name, sr_name, v_path, float(z_zoom)


if __name__ == "__main__":
    denoise_name, sr_name, v_path, z_zoom = inference_pa()
    print("\n--- GUI Selections ---")
    print(f"Selected Denoise Name: {denoise_name}")
    print(f"Selected SR Name: {sr_name}")
    print(f"Selected Path: {v_path}")
    print(f"Z Zoom Scale: {z_zoom}")
