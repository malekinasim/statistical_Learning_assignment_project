import pandas as pd
from utils.FileUtil import read_csv_file

class DataLoader:
    @staticmethod
    def load_data(file_path):
        """Load data from a file and return a DataFrame."""
        if file_path.endswith('.csv'):
            return read_csv_file(file_path)
        elif file_path.endswith(".xlsx"):
            return pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file type")