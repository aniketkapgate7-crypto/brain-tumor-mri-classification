"""
Execute and Save Capstone Notebook Outputs.
Runs all cells of Brain_Tumor_Classification_Capstone.ipynb headlessly,
captures all terminal text outputs, DataFrame displays, statistical test results,
and base64-encoded matplotlib/seaborn plots, and saves them directly into the .ipynb file.
"""

import base64
import contextlib
import io
import json
import logging
import os
import sys
from pathlib import Path

# Ensure headless execution
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NOTEBOOK_PATH = Path("notebooks/Brain_Tumor_Classification_Capstone.ipynb")


def figure_to_base64(fig) -> str:
    """Converts a matplotlib figure to base64 encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return img_b64


def execute_capstone_notebook(notebook_path: Path = NOTEBOOK_PATH):
    logger.info(f"Loading notebook from: {notebook_path.resolve()}")
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Initialize execution environment
    exec_env = {
        "__name__": "__main__",
        "__file__": str(notebook_path.resolve()),
    }

    # Custom display handler to capture DataFrames and objects
    captured_displays = []

    def custom_display(*args):
        for arg in args:
            if isinstance(arg, (pd.DataFrame, pd.Series)):
                captured_displays.append({
                    "output_type": "execute_result",
                    "data": {
                        "text/plain": [arg.to_string() + "\n"],
                        "text/html": [arg.to_html() + "\n"] if hasattr(arg, "to_html") else [arg.to_string() + "\n"]
                    },
                    "metadata": {}
                })
            else:
                captured_displays.append({
                    "output_type": "display_data",
                    "data": {
                        "text/plain": [str(arg) + "\n"]
                    },
                    "metadata": {}
                })

    exec_env["display"] = custom_display

    # Hook plt.show to capture open figures
    captured_plots = []

    def custom_plt_show(*args, **kwargs):
        for fignum in plt.get_fignums():
            fig = plt.figure(fignum)
            b64_str = figure_to_base64(fig)
            captured_plots.append({
                "output_type": "display_data",
                "data": {
                    "image/png": b64_str,
                    "text/plain": ["<Figure size ...>\n"]
                },
                "metadata": {}
            })
        plt.close("all")

    exec_env["plt_show_hook"] = custom_plt_show
    plt.show = custom_plt_show

    execution_counter = 1
    total_code_cells = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    logger.info(f"Total code cells to execute: {total_code_cells}")

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue

        source_code = "".join(cell["source"]).strip()
        if not source_code:
            cell["outputs"] = []
            cell["execution_count"] = None
            continue

        logger.info(f"Executing Code Cell #{execution_counter} (Cell index {idx})...")
        cell_outputs = []
        captured_plots.clear()
        captured_displays.clear()

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                import ast
                tree = ast.parse(source_code)
                eval_val = None
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last_expr = tree.body.pop()
                    if tree.body:
                        exec(compile(tree, f"cell_{idx}", "exec"), exec_env)
                    eval_val = eval(compile(ast.Expression(last_expr.value), f"cell_{idx}", "eval"), exec_env)
                else:
                    exec(compile(tree, f"cell_{idx}", "exec"), exec_env)

                # If eval returned a DataFrame, Series, or displayable object, capture it
                if eval_val is not None:
                    if isinstance(eval_val, (pd.DataFrame, pd.Series)):
                        captured_displays.append({
                            "output_type": "execute_result",
                            "data": {
                                "text/plain": [eval_val.to_string() + "\n"],
                                "text/html": [eval_val.to_html() + "\n"] if hasattr(eval_val, "to_html") else [eval_val.to_string() + "\n"]
                            },
                            "metadata": {}
                        })
                    elif not isinstance(eval_val, (str,)) or eval_val.strip():
                        captured_displays.append({
                            "output_type": "execute_result",
                            "data": {
                                "text/plain": [repr(eval_val) + "\n"]
                            },
                            "metadata": {}
                        })

                # If there are any lingering unclosed figures, capture them
                for fignum in plt.get_fignums():
                    fig = plt.figure(fignum)
                    b64_str = figure_to_base64(fig)
                    captured_plots.append({
                        "output_type": "display_data",
                        "data": {
                            "image/png": b64_str,
                            "text/plain": ["<Figure size ...>\n"]
                        },
                        "metadata": {}
                    })
                plt.close("all")

            # Collect stdout
            stdout_text = stdout_buf.getvalue()
            if stdout_text.strip():
                # Format as lines
                lines = [line + "\n" for line in stdout_text.splitlines()]
                cell_outputs.append({
                    "output_type": "stream",
                    "name": "stdout",
                    "text": lines
                })

            # Collect custom displays (e.g. DataFrames)
            if captured_displays:
                for disp in captured_displays:
                    if disp["output_type"] == "execute_result":
                        disp["execution_count"] = execution_counter
                    cell_outputs.append(disp)

            # Collect plots
            if captured_plots:
                for plot in captured_plots:
                    cell_outputs.append(plot)
            # Done collecting outputs

        except Exception as e:
            logger.error(f"Error during cell {idx} execution: {e}", exc_info=True)
            stderr_text = stderr_buf.getvalue() + f"\nTraceback:\n{str(e)}"
            cell_outputs.append({
                "output_type": "error",
                "ename": type(e).__name__,
                "evalue": str(e),
                "traceback": [stderr_text]
            })
            cell["outputs"] = cell_outputs
            cell["execution_count"] = execution_counter
            # Re-raise or save partial
            with open(notebook_path, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=1)
            raise

        cell["outputs"] = cell_outputs
        cell["execution_count"] = execution_counter
        execution_counter += 1

    # Save finalized notebook with populated outputs
    logger.info(f"Saving fully executed notebook with embedded outputs to: {notebook_path}")
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    logger.info("Successfully executed all code cells and saved outputs!")


if __name__ == "__main__":
    execute_capstone_notebook()
