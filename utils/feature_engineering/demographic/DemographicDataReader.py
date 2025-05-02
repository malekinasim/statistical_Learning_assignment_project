
#from feature_engineering.demographic
from utils.DataReader import DataReader
from utils.feature_engineering.demographic.DemographicFeatures import DemographicFeatures
class DemographicDataReader(DataReader):    
    @staticmethod
    def load_demographic_data_main(file_path):
        data = DataReader.load_data(file_path)
        data['Municipality'].ffill(inplace=True)
        data['pre_population'] = data['Population'].shift(1) 
        data[['Code', 'Municipality']] = data['Municipality'].str.split(' ', expand=True) 
        features_list = []
        
        for row in  data.to_dict(orient='records'):
            municipality = row['Municipality']
            year =row['Year'] 
            population=row['Population']
            pre_population=row['pre_population']
            # Create an instance of DemographicFeatures for each row
            population_info = DemographicFeatures(year,municipality,pre_population, population)
            features_list.append(population_info)
        return DataReader.to_dataFrame( features_list)