from datetime import datetime
from pathlib import Path

import marimo as mo

app = mo.App()


@app.cell
def __(mo):
    mo.md(
        """
        # 🧠 Dashboard Browser
        Interactive viewer for all your exported **marimo HTML dashboards**.
        """
    )
    return


@app.cell
def __(mo, Path):
    # Discover all exported dashboards
    here = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    html_files = sorted([p for p in here.glob("*.html") if p.name != "index.html"])

    def meta(p):
        s = p.stat()
        return {
            "label": f"{p.stem.replace('_', ' ').title()}  ·  {round(s.st_size / 1024, 1)} KB  ·  {datetime.fromtimestamp(s.st_mtime).strftime('%b %d')}",
            "file": p.name,
            "path": str(p),
        }

    options = [meta(p) for p in html_files]
    return here, html_files, options


@app.cell
def __(mo, options):
    # Nice selector
    choice = mo.ui.radio(
        options={o["label"]: o for o in options},
        label="Choose a dashboard",
        value=options[0]["label"] if options else None,
    )
    mo.vstack([choice])
    return (choice,)


@app.cell
def __(choice, mo):
    if not choice.value:
        mo.stop(True, mo.md("No dashboards found."))

    selected = choice.value
    file_name = selected["file"]
    rel_path = selected["file"]

    # Two action buttons
    open_tab = mo.ui.button(
        label="↗ Open in new browser tab",
        kind="success",
        on_click=lambda: mo.output.clear(),
    )

    mo.md(f"### {selected['label'].split('·')[0].strip()}")

    # The main preview — use a large iframe with sandbox that allows everything the dashboards need
    iframe = mo.Html(
        f"""
        <iframe
            src="{rel_path}"
            width="100%"
            height="720px"
            style="
                border: 1px solid #27272a;
                border-radius: 16px;
                background: white;
            "
            sandbox="allow-scripts allow-same-origin allow-popups allow-forms allow-modals"
        ></iframe>
        """
    )

    # Show the iframe + helpful text
    mo.vstack(
        [
            mo.hstack([open_tab], justify="end"),
            iframe,
            mo.md(
                f"""
                <div style="font-size:0.8rem; color:#71717a; margin-top:8px;">
                    📁 File: <code>{file_name}</code> &nbsp;•&nbsp;
                    Static marimo HTML export (outputs rendered). Widgets are read-only.
                    For full reactivity run the .py with `marimo edit` or export as html-wasm.
                </div>
                """
            ),
        ]
    )
    return file_name, iframe, open_tab, rel_path, selected


@app.cell
def __(mo):
    mo.md(
        """
        ---
        ### Tips
        - Regular `marimo export html` gives a **static** snapshot (no live widgets).
        - For fully interactive browser-only versions use:
          `marimo export html-wasm notebook.py -o out_dir --mode run`
        - This dashboard auto-discovers new .html files when you re-run the selector cell.
        """
    )
    return


if __name__ == "__main__":
    app.run()
