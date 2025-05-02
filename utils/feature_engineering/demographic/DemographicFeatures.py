
#from feature_engineering.demographic
class DemographicFeatures:
    def __init__(self,year:int,municipality:str,population:int,previous_population:int=None):
        self.__population = population
        self.__previous_population = previous_population
        self.__population_growth_rate = self.__calculate_population_growth_rate()
        self.__population_growth = self.__calculate_population_growth()
        self.__year=year
        self.__municipality=municipality


    def __calculate_population_growth_rate(self):
        if self.__previous_population is None or self.__population is None:
            return 0
        return (self.__population - self.__previous_population) / self.__previous_population

    def __calculate_population_growth(self):
        if self.__previous_population is None: 
            return 0  
        growth_rate = self.__calculate_population_growth_rate()
        return growth_rate * self.__previous_population
    
    def get_population(self):
        return self.__population
    
    def get_previous_population(self):
        return self.__previous_population
    
    def get_population_growth_rate(self):
        return self.__population_growth_rate
    
    def get_population_growth(self):
        return self.__population_growth
    
    def get_year(self):
        return self.__year  
    
    def get_municipality(self):
        return self.__municipality
    def to_dict(self):
        return {
            'Year': self.__year,
            'Municipality': self.__municipality,
            'Population': self.__population,
            'Population_growth_rate': self.__population_growth_rate
        }