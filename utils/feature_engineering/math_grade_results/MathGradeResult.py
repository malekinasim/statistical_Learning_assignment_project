#feature_engineering/math_grade_results/MathGradeResult.py
class MathGradeResult:
    def __init__(self, year, municipality,math_grade_rate_result):
        self.__year = year
        self.__municipality = municipality
        self.__math_grade_rate_result = math_grade_rate_result
    def get_year(self):
        return self.__year  
    def get_municipality(self):
        return self.__municipality  
    def get_math_grade_rate_result(self):
        return self.__math_grade_rate_result
    def to_dict(self):
        return {
            'Year': self.__year,
            'Municipality': self.__municipality,
            'Math_grade_rate_result': self.__math_grade_rate_result
        }
