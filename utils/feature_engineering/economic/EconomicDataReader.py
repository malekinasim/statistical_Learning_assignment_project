# feature_engineering/economic/EconomicDataLoader.py
from utils.DataReader import DataReader
from utils.feature_engineering.economic.EconomicFeatures import EconomicFeatures

class EconomicDataReader(DataReader):    
    @staticmethod
    def load_economic_data_main(file_path):
        data = DataReader.load_data(file_path)
        data['Municipality'].ffill(inplace=True)
        data[['Code', 'Municipality']] = data['Municipality'].str.split(' ', expand=True) 
        features_list = []

        for row in  data.to_dict(orient='records'):
            municipality = row['Municipality']
            year =row['Year'] 
            income_inequality=row['Income Inequality']
            average_income=row['Average_income']
            median_income=row['Median_income']
            # Create an instance of EconomicFeatures for each row
            economic = EconomicFeatures(year,municipality,income_inequality, average_income,median_income)
            features_list.append(economic)
       
        return DataReader.to_dataFrame(features_list)