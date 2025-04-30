from utils.DataLoader import DataLoader
from feature_engineering.math_grade_results.MathGradeResult import MathGradeResult

class MathGradeResultLoader(DataLoader):    
    @staticmethod
    def __fix_year(y):
        y = str(y)
        if len(y) == 2:
            return int('20' + y)
        return int(y)


    @staticmethod
    def load_math_grade_result_data_main(file_path):
        data = DataLoader.load_data(file_path)
        data = data.reindex().melt(id_vars=['Municipality','Code','County','Subject'], var_name='Year', value_name='MathGradeRateResult')
        data['Year'] = data['Year'].apply(MathGradeResultLoader.__fix_year)
        data['Year'] = data['Year'].astype(int)
        features_list=[]
        for row in data.to_dict(orient='records'):
            if(row['Year']>2023):
                continue
            grade_rate = MathGradeResult(
                year=row['Year'],
                municipality=row['Municipality'],
                math_grade_rate_result=row['MathGradeRateResult']
            )
            features_list.append(grade_rate)

        return DataLoader.to_dataFrame(features_list)
    