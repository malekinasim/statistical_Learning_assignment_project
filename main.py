from feature_engineering.economic.EconomicDataLoader import EconomicDataLoader
from feature_engineering.politic.PoliticDataLoader import PoliticDataLoader
from feature_engineering.education_level.EducationDataLoader import EducationDataLoader
from feature_engineering.demographic.DemographicDataLoader import DemographicDataLoader
from feature_engineering.math_grade_results.MathGradeResultsDataloader import MathGradeResultLoader



def main():
    file_path = 'data/Dalarna_MunicipalElectionResults.xlsx'  
    politic_data = PoliticDataLoader.load_political_data_main(file_path)
    
    file_path = 'data/Dalarna_AverageIncome.xlsx'
    economic_data = EconomicDataLoader.load_economic_data_main(file_path)
    
    file_path = 'data/Dalarna_Population_HigherEducation.xlsx'
    education_data = EducationDataLoader.load_education_data_main(file_path)
    
    file_path = 'data/Dalarna_Population_HigherEducation.xlsx'
    demographic_data = DemographicDataLoader.load_demographic_data_main(file_path)
    
    # print(demographic_data)
    file_path = 'data/Result_AK9_Dalarna_Kum.xlsx'
    grade_rate_result_data = MathGradeResultLoader.load_math_grade_result_data_main(file_path)
    
    
    data=grade_rate_result_data.merge(politic_data, on=['Municipality', 'Year'], how='left')    
    data=data.merge(economic_data, on=['Municipality', 'Year'], how='left')
    data=data.merge(education_data, on=['Municipality', 'Year'], how='left')  
    data=data.merge(demographic_data, on=['Municipality', 'Year'], how='left')  

    data.to_csv('data/processed_data.csv', index=False)
main()       
