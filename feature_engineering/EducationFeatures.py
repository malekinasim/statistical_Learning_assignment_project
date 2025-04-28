class EducationFeatures:
    def __init__(self,lower_education_count,higher_education_count,total_population):
        self.__total_population = total_population
        self.__lower_education_count= lower_education_count
        self.__higher_education_count = higher_education_count
        self.__lower_education_percentage = self.calculate_lower_education_percentage()
        self.__higher_education_percentage = self.calculate_higher_education_percentage()
        self.__higher_to_lower_education_ratio=self.__calculate_higher_to_lower_education_ratio()
        
    def __calculate_lower_education_percentage(self):
        if self.__total_population == 0:
           return float('inf')  # Avoid division by zero
        return round( self.__lower_education_count / self.__total_population * 100,2)
    
    def __calculate_higher_education_percentage(self):
        if self.__total_population == 0:
           return float('inf')  # Avoid division by zero
        return round( self.__higher_education_count / self.__total_population * 100,2)

    def __calculate_higher_to_lower_education_ratio(self):
        if self.lower_education_count == 0:
            return float('inf') 
        return round(self.higher_education_count / self.lower_education_count,4)
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