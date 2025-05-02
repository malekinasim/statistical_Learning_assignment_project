# utils/DataLoader.py
import pandas as pd
from utils.FileUtil import read_csv_file
from dataclasses import dataclass,asdict

class DataReader:
    @staticmethod
    def load_data(file_path):
        if file_path.endswith('.csv'):
            return read_csv_file(file_path)
        elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
            return pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file type")

    @staticmethod
    def to_dataFrame(object_list):
        data = [obj.to_dict() for obj in object_list]
        return pd.DataFrame(data)
