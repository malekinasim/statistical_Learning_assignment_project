from utils.feature_engineering.economic.EconomicDataReader import EconomicDataReader
from utils.feature_engineering.politic.PoliticDataReader import PoliticDataReader
from utils.feature_engineering.education_level.EducationDataReader import EducationDataReader
from utils.feature_engineering.demographic.DemographicDataReader import DemographicDataReader
from utils.feature_engineering.math_grade_results.MathGradeRateDataReader import MathGradeRateDataReader


def load_data(file_paths):
    politic_data = PoliticDataReader.load_political_data_main(file_paths['politic'])
    economic_data = EconomicDataReader.load_economic_data_main(file_paths['economic'])
    education_data = EducationDataReader.load_education_data_main(file_paths['education'])
    demographic_data = DemographicDataReader.load_demographic_data_main(file_paths['demographic'])
    grade_rate_result_data = MathGradeRateDataReader.load_math_grade_result_data_main(file_paths['grade_results'])
    
    return politic_data, economic_data, education_data, demographic_data, grade_rate_result_data
