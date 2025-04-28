class DemographicFeatures:
    def __init__(self,population,previous_population=None):
        self.__population = population
        self.__previous_population = previous_population
        self.__population_growth_rate = self.__calculate_population_growth_rate()
        self.__population_growth = self.__calculate_population_growth()


    def __calculate_population_growth_rate(self):
        if self.__population_previous is None or self.__population is None:
            return 0
        return (self.__population - self.__previous_population) / self.__previous_population

    def __calculate_population_growth(self):
        if self.__population_end is None: 
            return 0  
        growth_rate = self.population_growth_rate()
        return growth_rate * self.__population_end
    
    def get_population(self):
        return self.__population
    
    def get_previous_population(self):
        return self.__previous_population
    
    def get_population_growth_rate(self):
        return self.__population_growth_rate
    
    def get_population_growth(self):
        return self.__population_growth


# # وقتی فایل رو می‌خونی و رکوردها رو بررسی می‌کنی:
# def process_population_data(data):
#     for i in range(len(data) - 1):  # بررسی داده‌ها برای دو سال مختلف
#         population_start = data[i]['population']
#         population_end = data[i + 1]['population'] if i + 1 < len(data) else None
#         demographic = DemographicFeatures(population_start, population_end)
        
#         growth_rate = demographic.population_growth_rate()
#         population_growth = demographic.population_growth()
#         print(f"Growth Rate: {growth_rate}, Population Growth: {population_growth}")
