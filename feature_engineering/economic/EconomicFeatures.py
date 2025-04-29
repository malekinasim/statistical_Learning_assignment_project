
class EconomicFeatures:
    def __init__(self,year:int,municipality:str, average_income: float, income_inequality:float, median_income: float):
        self.__average_income = average_income
        self.__income_inequality = income_inequality
        self.__median_income = median_income
        self.__year = year
        self.__municipality = municipality

    def get_average_income(self):
        return self.__average_income

    def get_income_inequality(self):
        return self.__income_inequality
    def get_median_income(self):
        return self.__median_income
    def get_year(self):
        return self.__year  
    def get_municipality(self):
        return self.__municipality
    def to_dict(self):
        return {
            'Year': self.__year,
            'Municipality': self.__municipality,
            'Average_income': self.__average_income,
            'Income_inequality': self.__income_inequality,
            'Median_income': self.__median_income
        }