
from utils.FileUtil import read_csv_file
from regression_models import OLS_backward_elimination,model_diagnostics,plot_actual_vs_predicted,fixed_effects_panel_backward_elimination,random_effects_panel_backward_elimination
from data_cleaning_processing import load_clean_data
import os
import pandas as pd


    
def main():
    # 1. Load data
    if(not os.path.exists("results/data/cleaned_data.csv")):
       load_clean_data()
    df=read_csv_file("results/data/cleaned_data.csv")
  
    #Model analysis 
    exclude_features = ['Year', 'Municipality', 'Math_grade_rate_result']
    features = [col for col in df.columns if col not in exclude_features ]
    X=df[features].astype(float)
    y=df['Math_grade_rate_result'].astype(float)
    OLS_model = OLS_backward_elimination(X, y)
    print("OLS model with none correlated features :")
    print(OLS_model.summary())
    model_diagnostics(OLS_model,X,'OLS_model')
    plot_actual_vs_predicted(OLS_model,X,y,'OLS_model')
    print("-----------------------------------------------------------------------------------------------------")
    
    

    dummy_df = pd.get_dummies(df, columns=['Municipality','Year'], drop_first=True)
    X = dummy_df[[col for col in dummy_df.columns if col != 'Math_grade_rate_result']].astype(float)
    y = dummy_df['Math_grade_rate_result'].astype(float)
    
    OLS_model = OLS_backward_elimination(X, y)
    print("OLS model with none correlated features and dummy variables :")
    print(OLS_model.summary())
    model_diagnostics(OLS_model,X,'dummy_variables_OLS_model')
    plot_actual_vs_predicted(OLS_model,X,y,'dummy_variables_OLS_model')
    print("-----------------------------------------------------------------------------------------------------")
    
    
    exclude_features = ['Year', 'Municipality', 'Math_grade_rate_result']
    features = [col for col in df.columns if col not in exclude_features]
    
    fix_effective_panel_model = fixed_effects_panel_backward_elimination(
        df,
        'Math_grade_rate_result',
        features,
        'Municipality',
        'Year'
    )
    print("Fixed effects panel model with non correlated features based on year and municipality fixed effect entities:")
    print(fix_effective_panel_model.summary)
    model_diagnostics(fix_effective_panel_model, df[features],  'Fixed_effective_panel_model')
    plot_actual_vs_predicted(fix_effective_panel_model, df[features], df['Math_grade_rate_result'], 'Fixed_effective_panel_model')

    print("-----------------------------------------------------------------------------------------------------")
    
    
    exclude_features = ['Year', 'Municipality', 'Math_grade_rate_result']
    features = [col for col in df.columns if col not in exclude_features]
    
    random_effective_panel_model = random_effects_panel_backward_elimination(
        df,
        'Math_grade_rate_result',
        features,
        'Municipality',
        'Year'
    )
    print("Random effects panel model with non correlated features based on year and municipality fixed effect entities:")
    print(random_effective_panel_model.summary)
    model_diagnostics(random_effective_panel_model, df[features],  'Random_effective_panel_model')
    plot_actual_vs_predicted(random_effective_panel_model, df[features], df['Math_grade_rate_result'], 'Random_effective_panel_model')

    print("-----------------------------------------------------------------------------------------------------")
    
    
main()
