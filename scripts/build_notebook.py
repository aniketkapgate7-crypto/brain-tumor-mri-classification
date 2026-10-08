"""
Capstone Notebook Generation Script.
Builds the complete, error-free Brain_Tumor_Classification_Capstone.ipynb
based on the AlmaBetter template guidelines without altering any markdown headers.
"""

import json
from pathlib import Path

TEMPLATE_PATH = Path(r"C:\Users\HP\Downloads\Sample_ML_Submission_Template-2.ipynb")
OUTPUT_PATH = Path(r"d:\Internship-1Aug\brain-tumor-mri-classification\notebooks\Brain_Tumor_Classification_Capstone.ipynb")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
    template_nb = json.load(f)

print(f"Loaded template with {len(template_nb['cells'])} cells.")
