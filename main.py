from feature_engineering.economic.EconomicDataLoader import EconomicDataLoader
from feature_engineering.politic.PoliticDataLoader import PoliticDataLoader
from feature_engineering.education_level.EducationDataLoader import EducationDataLoader
from feature_engineering.demographic.DemographicDataLoader import DemographicDataLoader
from feature_engineering.math_grade_results.MathGradeResultsDataloader import MathGradeResultLoader
from utils.FileUtil import write_df_to_csv
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn
import numpy as np

def Merge_data(folder_path,file_name):
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
    write_df_to_csv(data,folder_path, file_name,encoding='utf-8-sig')
    
    return data
def check_data_VIF(df):
    X = df.drop('Math_grade_rate_result', axis=1)
  
    X = X.select_dtypes(include=['int64', 'float64', 'bool'])
    X = sm.add_constant(X)

    print(X.dtypes) 
    print(X.head())  

    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    print(vif_data)


def show_correlation_matrix(df, image_path):
    plt.figure(figsize=(10, 8))
    corr = df.corr().round(2)
    seaborn.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation Matrix", fontsize=16)
    plt.tight_layout()
    plt.savefig(image_path, dpi=300)
    plt.close()

def math_grade_rate_plot(df):
    graph = seaborn.FacetGrid(df, col="Municipality", hue="Municipality", col_wrap=2, height=4, aspect=1.5)
    graph.map(seaborn.lineplot, "Year", "Math_grade_rate_result", marker="o")
    plt.savefig("data/math_grade_rate_result.png", dpi=300)
    plt.close()

def show_correlation_pair_plot(df, image_path):
    seaborn.pairplot(df, kind='reg')
    plt.suptitle("Pairwise Relationships with Regression Diagnostics", y=1.02)
    plt.savefig(image_path, dpi=300)
    plt.close()
    
    
def drop_highly_correlated_features(df, threshold=0.7):
    corr = df.corr().round(2).abs()
    dropped = set()
    for i in corr.columns:
        for j in corr.columns:
            if(len(dropped)>len(corr.columns)-3):
                break
            if i!="Math_grade_rate_result" and j!="Math_grade_rate_result" and  i != j and corr.loc[i, j] > threshold:
                dropped.add(j)
    return df.drop(columns=dropped), list(dropped)

def features_correlation_checking(df):
    income_vars = ["Math_grade_rate_result", "Average_income", "Income_inequality"]
    pop_vars = ["Math_grade_rate_result","Population_growth_rate", "Population"]
    edu_vars = ["Math_grade_rate_result", "Higher_education_percentage", "Lower_education_percentage", "Higher_to_lower_education_ratio"]

    show_correlation_matrix(df[income_vars], "data/correlation_matrix_income.png")
    show_correlation_pair_plot(df[income_vars], "data/correlation_plot_income.png")
    
    show_correlation_matrix(df[pop_vars], "data/correlation_matrix_population.png")
    show_correlation_pair_plot(df[pop_vars], "data/correlation_plot_population.png")
    
    show_correlation_matrix(df[edu_vars], "data/correlation_matrix_education.png")
    show_correlation_pair_plot(df[edu_vars], "data/correlation_plot_education.png")
    
    income_df, income_dropped = drop_highly_correlated_features(df[income_vars])
    pop_df, pop_dropped = drop_highly_correlated_features(df[pop_vars])
    edu_df, edu_dropped = drop_highly_correlated_features(df[edu_vars])
    selected_features=[col for col in  df.columns if col not in income_dropped and col not in pop_dropped and col not in edu_dropped ]
    return selected_features

def main():
    df=Merge_data('data/','merged_data.csv')
    math_grade_rate_plot(df)
    selected_features=features_correlation_checking(df)
    not_correlated_df=df[selected_features]
    dummy_df= pd.get_dummies(not_correlated_df, columns=['Municipality','Year'], drop_first=True)
    #check_data_VIF(dummy_df) 


main()