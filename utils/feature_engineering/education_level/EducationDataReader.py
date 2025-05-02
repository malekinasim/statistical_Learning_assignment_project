#feature_engineering/education_level/EducationDataLoader.py
from utils.DataReader import DataReader
from utils.feature_engineering.education_level.EducationFeatures import EducationFeatures

class EducationDataReader(DataReader):    
    @staticmethod
    def load_education_data_main(file_path):
        data = DataReader.load_data(file_path)
        data['Municipality'].ffill(inplace=True)
        data[['Code', 'Municipality']] = data['Municipality'].str.split(' ', expand=True) 
        features_list = []

        for row in  data.to_dict(orient='records'):
            municipality = row['Municipality']
            year =row['Year'] 
            lower_education_count=row['post-secondary education, less than 3 years (ISCED97 4+5B)']
            higher_education_count=row['post-secondary education 3 years or more (ISCED97 5A)']
            population=row['Population']
            # Create an instance of EconomicFeatures for each row
            education = EducationFeatures(year,municipality,lower_education_count, higher_education_count,population)
            features_list.append(education)
            
        return DataReader.to_dataFrame(features_list)