"""
Main Entry Point: Confusion Matrix Generator for 6-Class ML Selector Model

Executes src.analysis.generate_confusion_matrix
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analysis.generate_confusion_matrix import generate_6class_confusion_matrix

if __name__ == "__main__":
    generate_6class_confusion_matrix()
