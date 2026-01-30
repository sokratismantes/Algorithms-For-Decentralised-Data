# =========================
# GUI (Tabs + Cards) for experiments.py in Colab
# =========================

import os, subprocess, zipfile, textwrap
from pathlib import Path
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML, Image, Markdown

CONTENT_DIR = Path("/content")

# ---------- CSS ----------
CSS = """
<style>
:root { --bg:#0b1220; --card:#111a2b; --card2:#0f1726; --text:#e6edf7; --muted:#9bb0d0; --accent:#6aa6ff; --warn:#ffcc66; --ok:#74f0a6; --err:#ff6a6a; }
.colab-dht * { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji"; }
.colab-dht .wrap { background: var(--bg); border-radius: 16px; padding: 16px; color: var(--text); }
.colab-dht h2 { margin: 0 0 8px 0; font-size: 20px; font-weight: 700; letter-spacing: .2px; }
.colab-dht .sub { color: var(--muted); margin-bottom: 14px; }
.colab-dht .row { display: flex; gap: 12px; flex-wrap: wrap; }
.colab-dht .card { background: linear-gradient(180deg, var(--card), var(--card2)); border: 1px solid rgba(255,255,255,.08); border-radius: 14px; padding: 14px; flex: 1; min-width: 280px; }
.colab-dht .card h3 { margin: 0 0 10px 0; font-size: 14px; font-weight: 800; color: var(--text); }
.colab-dht .hint { color: var(--muted); font-size: 12px; line-height: 1.35; }
.colab-dht .pill { display:inline-block; padding: 4px 10px; border-radius: 999px; font-size: 12px; border: 1px solid rgba(255,255,255,.12); background: rgba(255,255,255,.04); margin-right: 6px; }
.colab-dht .pill.ok { border-color: rgba(116,240,166,.35); color: var(--ok); }
.colab-dht .pill.warn { border-color: rgba(255,204,102,.35); color: var(--warn); }
.colab-dht .pill.err { border-color: rgba(255,106,106,.35); color: var(--err); }
.colab-dht .sp { height: 10px; }
.colab-dht .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 12px; color: var(--muted); }
.colab-dht .footer { color: var(--muted); font-size: 12px; margin-top: 10px; }
</style>
"""

display(HTML(CSS))

# ---------- Helpers ----------
def list_data_files():
    files = []
    for ext in ("*.csv", "*.xlsx", "*.xls"):
        files += [p.name for p in CONTENT_DIR.glob(ext)]
    return sorted(files)

def ensure_csv_from_xlsx(xlsx_path: Path, csv_name: str, max_rows: int, seed: int):
    df = pd.read_excel(xlsx_path)
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    csv_path = CONTENT_DIR / csv_name
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path, len(df)

def run_cmd(cmd_args):
    proc = subprocess.run(cmd_args, cwd=str(CONTENT_DIR), capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr

def safe_read_csv(path: Path, n=20):
    try:
        df = pd.read_csv(path)
        return df.head(n), df
    except Exception as e:
        return None, str(e)

def make_zip(outdir: Path, zip_path: Path, include_dataset: Path = None):
    with zipfile.ZipFile(zip_path, "w") as z:
        # results csv + plots
        if (outdir / "results.csv").exists():
            z.write(outdir / "results.csv", arcname="results/results.csv")
        for op in ["build","insert","update","lookup","delete","join","leave"]:  # CHANGED
            p = outdir / f"{op}.png"
            if p.exists():
                z.write(p, arcname=f"results/{op}.png")
        # dataset
        if include_dataset is not None and include_dataset.exists():
            z.write(include_dataset, arcname=include_dataset.name)

# ---------- UI state ----------
data_files = list_data_files()
file_dropdown = widgets.Dropdown(
    options=data_files,
    value=(data_files[0] if data_files else None),
    description="File",
    layout=widgets.Layout(width="420px")
)
refresh_btn = widgets.Button(description="↻ Refresh", button_style="", layout=widgets.Layout(width="130px"))
convert_chk = widgets.Checkbox(value=False, description="If XLSX → convert to CSV (sample)")
csv_name_text = widgets.Text(value="data_movies_clean.csv", description="CSV name", layout=widgets.Layout(width="420px"))

nodes_text = widgets.Text(value="16,32,64", description="nodes", layout=widgets.Layout(width="220px"))
m_int = widgets.IntText(value=40, description="m", layout=widgets.Layout(width="160px"))
seed_int = widgets.IntText(value=7, description="seed", layout=widgets.Layout(width="160px"))
max_rows_int = widgets.IntText(value=50000, description="max_rows", layout=widgets.Layout(width="220px"))
outdir_text = widgets.Text(value="results", description="outdir", layout=widgets.Layout(width="220px"))

records_int = widgets.IntText(value=20000, description="records", layout=widgets.Layout(width="220px"))
updates_int = widgets.IntText(value=2000, description="updates", layout=widgets.Layout(width="220px"))  # NEW
queries_int = widgets.IntText(value=5000, description="queries", layout=widgets.Layout(width="220px"))
deletes_int = widgets.IntText(value=2000, description="deletes", layout=widgets.Layout(width="220px"))
joins_int = widgets.IntText(value=10, description="joins", layout=widgets.Layout(width="220px"))
leaves_int = widgets.IntText(value=10, description="leaves", layout=widgets.Layout(width="220px"))
K_int = widgets.IntText(value=10, description="K demo", layout=widgets.Layout(width="220px"))

force_records_chk = widgets.Checkbox(value=True, description="Auto-cap records to max_rows")
show_preview_chk = widgets.Checkbox(value=True, description="Show dataset preview")
show_plots_chk = widgets.Checkbox(value=True, description="Show plots inline")

run_btn = widgets.Button(description="▶ Run", button_style="success", layout=widgets.Layout(width="140px"))
show_btn = widgets.Button(description="📊 Show results", button_style="info", layout=widgets.Layout(width="160px"))
zip_btn = widgets.Button(description="⬇ Download ZIP", button_style="", layout=widgets.Layout(width="170px"))

status_html = widgets.HTML()
run_out = widgets.Output(layout=widgets.Layout(border="1px solid rgba(0,0,0,.15)", padding="10px"))
results_out = widgets.Output(layout=widgets.Layout(border="1px solid rgba(0,0,0,.15)", padding="10px"))

# ---------- Header ----------
header = widgets.HTML("""
<div class="colab-dht">
  <div class="wrap">
    <h2>Chord vs Pastry — Experiments Console</h2>
    <div class="sub">Configure parameters, run experiments, and view plots & CSV results directly in Colab.</div>
    <div id="statusline"></div>
  </div>
</div>
""")

def set_status(msg, kind="ok"):
    # kind: ok|warn|err
    pill = f'<span class="pill {kind}">{kind.upper()}</span>'
    status_html.value = f"""
    <div class="colab-dht"><div class="wrap">
      {pill} <span class="mono">{msg}</span>
    </div></div>
    """

def validate_params():
    msgs = []
    kind = "ok"
    if file_dropdown.value is None:
        return "No dataset file found in /content. Upload a .csv or .xlsx first.", "err"
    if records_int.value < 0 or updates_int.value < 0 or queries_int.value < 0 or deletes_int.value < 0 or joins_int.value < 0 or leaves_int.value < 0:
        return "Counts must be non-negative.", "err"
    if max_rows_int.value <= 0:
        return "max_rows must be > 0", "err"
    if records_int.value > max_rows_int.value:
        msgs.append(f"records ({records_int.value}) > max_rows ({max_rows_int.value}) → inserts will be capped.")
        kind = "warn"
    if m_int.value <= 0:
        return "m must be > 0", "err"
    return (" | ".join(msgs) if msgs else "Ready."), kind

def refresh_files(_=None):
    files = list_data_files()
    file_dropdown.options = files
    if files and file_dropdown.value not in files:
        file_dropdown.value = files[0]
    msg, kind = validate_params()
    set_status(msg, kind)

refresh_btn.on_click(refresh_files)

for w in [file_dropdown, records_int, max_rows_int, m_int]:
    w.observe(lambda ch: set_status(*validate_params()), names="value")

refresh_files()

# ---------- Tabs content ----------
# Dataset tab
dataset_card = widgets.VBox([
    widgets.HTML("""
    <div class="colab-dht"><div class="wrap">
      <div class="row">
        <div class="card">
          <h3>Dataset</h3>
          <div class="hint">Choose a dataset from <span class="mono">/content</span>. If you select XLSX, you can convert it to CSV with sampling.</div>
          <div class="sp"></div>
        </div>
      </div>
    </div></div>
    """),
    widgets.HBox([file_dropdown, refresh_btn]),
    widgets.HBox([convert_chk, show_preview_chk]),
    csv_name_text,
    max_rows_int,
])

preview_out = widgets.Output(layout=widgets.Layout(border="1px solid rgba(0,0,0,.15)", padding="10px"))

def show_dataset_preview():
    with preview_out:
        preview_out.clear_output()
        if not show_preview_chk.value:
            display(Markdown("Dataset preview is disabled."))
            return
        if file_dropdown.value is None:
            display(Markdown("No dataset selected."))
            return
        p = CONTENT_DIR / file_dropdown.value
        if p.suffix.lower() in (".xlsx", ".xls"):
            # show only first rows by reading first sheet head
            try:
                df = pd.read_excel(p, nrows=15)
                display(Markdown(f"**Preview of `{p.name}` (first 15 rows)**"))
                display(df)
            except Exception as e:
                display(Markdown(f"⚠️ Could not read xlsx preview: `{e}`"))
        else:
            head, full = safe_read_csv(p, n=15)
            if head is None:
                display(Markdown(f"⚠️ Could not read CSV: `{full}`"))
            else:
                display(Markdown(f"**Preview of `{p.name}` (first 15 rows)** — rows: **{len(full)}**"))
                display(head)

show_preview_chk.observe(lambda ch: show_dataset_preview(), names="value")
file_dropdown.observe(lambda ch: show_dataset_preview(), names="value")

# Parameters tab
params_card = widgets.VBox([
    widgets.HTML("""
    <div class="colab-dht"><div class="wrap">
      <div class="row">
        <div class="card">
          <h3>Core parameters</h3>
          <div class="hint">nodes: comma list (e.g. 16,32,64). m: keyspace bits. seed: determinism.</div>
        </div>
        <div class="card">
          <h3>Workload</h3>
          <div class="hint">records, queries, deletes, joins, leaves. Use max_rows ≥ records.</div>
        </div>
      </div>
    </div></div>
    """),
    widgets.HBox([nodes_text, m_int, seed_int]),
    widgets.HBox([outdir_text, K_int]),
    widgets.HBox([records_int, updates_int, queries_int]),
    widgets.HBox([deletes_int, joins_int, leaves_int]),
    widgets.HBox([force_records_chk, show_plots_chk]),
])

# Run tab
run_controls = widgets.VBox([
    widgets.HTML("""
    <div class="colab-dht"><div class="wrap">
      <div class="row">
        <div class="card">
          <h3>Run</h3>
          <div class="hint">Press Run to execute <span class="mono">experiments.py</span>. Use Show results to render tables & plots.</div>
        </div>
      </div>
    </div></div>
    """),
    status_html,
    widgets.HBox([run_btn, show_btn, zip_btn]),
    widgets.HTML("<div class='colab-dht'><div class='wrap'><div class='footer'>Tip: If records > max_rows, inserts are capped (or auto-capped if enabled).</div></div></div>"),
    run_out,
])

# Results tab
results_panel = widgets.VBox([
    widgets.HTML("""
    <div class="colab-dht"><div class="wrap">
      <div class="row">
        <div class="card">
          <h3>Results</h3>
          <div class="hint">Displays <span class="mono">results/results.csv</span> and plots. Use Download ZIP to export everything.</div>
        </div>
      </div>
    </div></div>
    """),
    results_out
])

tabs = widgets.Tab(children=[
    widgets.VBox([dataset_card, preview_out]),
    params_card,
    run_controls,
    results_panel
])
tabs.set_title(0, "Dataset")
tabs.set_title(1, "Parameters")
tabs.set_title(2, "Run")
tabs.set_title(3, "Results")

display(header, tabs)

# ---------- Actions ----------
def on_run(_):
    with run_out:
        run_out.clear_output()

    msg, kind = validate_params()
    set_status(msg, kind)
    if kind == "err":
        with run_out:
            print("Fix errors and try again.")
        return

    dataset_name = file_dropdown.value
    ds_path = CONTENT_DIR / dataset_name
    final_path = ds_path

    # Auto cap
    max_rows = int(max_rows_int.value)
    records = int(records_int.value)
    if force_records_chk.value and records > max_rows:
        records = max_rows

    updates = int(updates_int.value)
    if force_records_chk.value and updates > records:
        updates = records

    # Convert if requested and XLSX
    if convert_chk.value and ds_path.suffix.lower() in (".xlsx", ".xls"):
        final_path, used_rows = ensure_csv_from_xlsx(
            ds_path,
            csv_name=csv_name_text.value.strip() or "data_movies_clean.csv",
            max_rows=max_rows,
            seed=int(seed_int.value)
        )
        set_status(f"Converted XLSX → CSV: {final_path.name} (rows: {used_rows})", "ok")

    cmd = [
        "python", "experiments.py",
        "--file", str(final_path),
        "--nodes", nodes_text.value,
        "--max_rows", str(max_rows),
        "--m", str(int(m_int.value)),
        "--records", str(records),
        "--updates", str(updates),
        "--queries", str(int(queries_int.value)),
        "--deletes", str(int(deletes_int.value)),
        "--joins", str(int(joins_int.value)),
        "--leaves", str(int(leaves_int.value)),
        "--K", str(int(K_int.value)),
        "--seed", str(int(seed_int.value)),
        "--outdir", outdir_text.value.strip() or "results",
    ]

    with run_out:
        print("Running:\n", " ".join(cmd), "\n")
        rc, stdout, stderr = run_cmd(cmd)
        print(stdout)
        if rc != 0:
            print("ERROR:\n", stderr)
            set_status("Run failed. Check output for traceback.", "err")
        else:
            set_status("Run completed successfully.", "ok")

def on_show(_):
    outdir = CONTENT_DIR / (outdir_text.value.strip() or "results")
    csv_path = outdir / "results.csv"

    with results_out:
        results_out.clear_output()
        if not csv_path.exists():
            display(Markdown(f"⚠️ Δεν βρέθηκε `{csv_path}`. Τρέξε πρώτα Run."))
            return

        res = pd.read_csv(csv_path)
        display(Markdown(f"### ✅ `{csv_path}` — rows: **{len(res)}**"))
        display(Markdown("#### Preview (first 20 rows)"))
        display(res.head(20))

        display(Markdown("#### Summary: mean hops by (operation, protocol, N)"))
        summary = (res.groupby(["operation","protocol","N"])["hops"]
                   .agg(["count","mean","median","std","min","max"])
                   .reset_index()
                   .sort_values(["operation","N","protocol"]))
        display(summary)

        if {"locate_hops","move_hops","moved_keys"}.issubset(res.columns):
            display(Markdown("#### Join/Leave Overhead"))
            jl = res[res["operation"].isin(["join","leave"])].copy()
            overhead = (jl.groupby(["operation","phase","protocol","N"])
                        .agg(events=("hops","count"),
                             avg_locate_hops=("locate_hops","mean"),
                             avg_move_hops=("move_hops","mean"),
                             avg_moved_keys=("moved_keys","mean"),
                             max_moved_keys=("moved_keys","max"))
                        .reset_index()
                        .sort_values(["operation","phase","N","protocol"]))
            display(overhead)

        if show_plots_chk.value:
            display(Markdown("#### Plots"))
            for op in ["build","insert","update","lookup","delete","join","leave"]:  # CHANGED
                img = outdir / f"{op}.png"
                if img.exists():
                    display(Markdown(f"**{op}.png**"))
                    display(Image(filename=str(img)))
                else:
                    display(Markdown(f"⚠️ Missing plot: `{img}`"))

def on_zip(_):
    outdir = CONTENT_DIR / (outdir_text.value.strip() or "results")
    zip_path = CONTENT_DIR / "results_bundle.zip"
    dataset_name = file_dropdown.value
    ds_path = CONTENT_DIR / dataset_name if dataset_name else None

    # If we converted XLSX to CSV, include the produced CSV if exists
    include = None
    if convert_chk.value and ds_path and ds_path.suffix.lower() in (".xlsx", ".xls"):
        maybe_csv = CONTENT_DIR / (csv_name_text.value.strip() or "data_movies_clean.csv")
        include = maybe_csv if maybe_csv.exists() else None
    else:
        include = ds_path if (ds_path and ds_path.exists()) else None

    make_zip(outdir, zip_path, include_dataset=include)

    with run_out:
        print("\nCreated ZIP:", zip_path)

    try:
        from google.colab import files
        files.download(str(zip_path))
    except Exception as e:
        with run_out:
            print("Could not auto-download (non-Colab env?):", e)

run_btn.on_click(on_run)
show_btn.on_click(on_show)
zip_btn.on_click(on_zip)

# initial dataset preview
show_dataset_preview()
