#feature_engineering/education_level/EducationFeatures.py
import numpy as np
class EducationFeatures:
    def __init__(self,year:int,municipality:str,lower_education_count:int,higher_education_count:int,total_population:int):
        self.__year=year
        self.__municipality=municipality
        self.__total_population = total_population
        self.__lower_education_count= lower_education_count
        self.__higher_education_count = higher_education_count
        self.__lower_education_percentage = self.__calculate_lower_education_percentage()
        self.__higher_education_percentage = self.__calculate_higher_education_percentage()
        self.__higher_to_lower_education_ratio=self.__calculate_higher_to_lower_education_ratio()
        self.__education_share=self.__calculate_education_share(total_population)
        
    def __calculate_lower_education_percentage(self):
        if self.__total_population == 0:
           return np.nan # Avoid division by zero
        return round( self.__lower_education_count / self.__total_population * 100,2)
    
    def __calculate_higher_education_percentage(self):
        if self.__total_population == 0:
           return np.nan # Avoid division by zero
        return round( self.__higher_education_count / self.__total_population * 100,2)

    def __calculate_higher_to_lower_education_ratio(self):
        if self.__lower_education_count == 0:
            return np.nan
        return round(self.__higher_education_count / self.__lower_education_count,4)
    
    def __calculate_education_share(self,total_population):
        return round(self.__higher_education_count+ self.__lower_education_count/total_population ,2)
    def get_lower_education_count(self):
        return self.__lower_education_count
    
    def get_higher_education_count(self):
        return self.__higher_education_count
    
    def get_total_population(self):
        return self.__total_population
    
    def get_lower_education_percentage(self):
        return self.__lower_education_percentage
    
    def get_higher_education_percentage(self):
        return self.__higher_education_percentage
    
    def get_higher_to_lower_education_ratio(self):
        return self.__higher_to_lower_education_ratio   
    
    def get_year(self):
        return self.__year  
    def get_municipality(self):
        return self.__municipality  
    
    def get_education_share(self):
        return self.__education_share
    def to_dict(self):
        return {
            "Year":self.__year,
            "Municipality":self.__municipality,
            "Lower_education_percentage":self.__lower_education_percentage,
            "Higher_education_percentage":self.__higher_education_percentage,
            "Higher_to_lower_education_ratio":self.__higher_to_lower_education_ratio,
            "education_share":self.__education_share
        }