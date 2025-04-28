import os
from pathlib import Path
import pandas as pd

def read_excel_file(file_path,sheet_name=None):
    """Read an Excel file and return a pandas DataFrame."""
    if not os.path.exists(file_path):
        raise ValueError(f"The file {file_path} does not exist")
    try:
        # Use pandas to read the Excel file
        if sheet_name is None:
            df = pd.read_excel(file_path,sheet_name=0)  # Read the first sheet by default
        else:
            df = pd.read_excel(file_path,sheet_name=sheet_name)  # Read the specified sheet
        return df
    except Exception as e:
        raise ValueError(f"An error occurred when reading the Excel file {file_path}: {e}")

def read_csv_file(file_path,sep=',',encoding='utf-8'):
    """Read an Excel file and return a pandas DataFrame."""
    if not os.path.exists(file_path):
        raise ValueError(f"The file {file_path} does not exist")
    try:
        # Use pandas to read the Excel file
        df = pd.read_csv(file_path,sep=sep,encoding=encoding)
        return df
    except Exception as e:
        raise ValueError(f"An error occurred when reading the Excel file {file_path}: {e}")
    

def write_df_to_excel(df, folder_path,file_name,sheet_name='Sheet1'):
    """Write a DataFrame to an Excel file."""
    if not os.path.exists(folder_path):
        raise ValueError(f"the file {folder_path}  does not exist")  
    try:
        # Write the DataFrame to an Excel file
        file_path=os.path.join(folder_path,file_name)
        df.to_excel(file_path,sheet_name=sheet_name, index=False)  # index=False to avoid writing row numbers
        print(f"Data successfully written to {file_path}")
    except Exception as e:
        raise ValueError(f"An error occurred when writing to the Excel file {file_path}: {e}")   
   
def write_df_to_csv(df, folder_path,file_name,sep=',',encoding='utf-8'):
    """Write a DataFrame to a CSV file."""
    if not os.path.exists(folder_path):
        raise ValueError(f"the file {folder_path}  does not exist")  
    try:
        # Write the DataFrame to a CSV file
        file_path=os.path.join(folder_path,file_name)
        df.to_csv(file_path,sep=sep,encoding=encoding, index=False)  # index=False to avoid writing row numbers
        print(f"Data successfully written to {file_path}")
    except Exception as e:
        raise ValueError(f"An error occurred when writing to the CSV file {file_path}: {e}")