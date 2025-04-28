class EconomicFeatures:
    def __init__(self, average_income, income_inequality, median_income):
        self.__average_income = average_income
        self.__income_inequality = income_inequality
        self.__median_income = median_income

    def get_average_income(self):
        return self.__average_income

    def get_income_inequality(self):
        return self.__income_inequality
    def get_median_income(self):
        return self.__median_income