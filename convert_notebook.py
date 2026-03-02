import json
from pathlib import Path

def convert_py_to_ipynb(py_path, ipynb_path):
    with open(py_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cells = []
    current_cell = []
    cell_type = "code"

    for line in lines:
        if line.startswith("# %% [markdown]"):
            if current_cell:
                cells.append({
                    "cell_type": cell_type,
                    "metadata": {},
                    "source": current_cell
                })
            current_cell = []
            cell_type = "markdown"
        elif line.startswith("# %% [code]"):
            if current_cell:
                cells.append({
                    "cell_type": cell_type,
                    "metadata": {},
                    "source": current_cell
                })
            current_cell = []
            cell_type = "code"
        else:
            # Skip the marker lines themselves if they are the generic # %%
            if line.strip() == "# %%":
                continue
            current_cell.append(line)

    # Add the last cell
    if current_cell:
        cells.append({
            "cell_type": cell_type,
            "metadata": {},
            "source": current_cell
        })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(ipynb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    print(f"Converted {py_path} to {ipynb_path}")

if __name__ == "__main__":
    convert_py_to_ipynb(
        'c:/Users/dell/OneDrive/Desktop/SEM 6/XAI/notebooks/kaggle_training.py',
        'c:/Users/dell/OneDrive/Desktop/SEM 6/XAI/notebooks/kaggle_training.ipynb'
    )
