from feature_engineering.economic.EconomicDataLoader import EconomicDataLoader
from feature_engineering.politic.PoliticDataLoader import PoliticDataLoader
from feature_engineering.education_level.EducationDataLoader import EducationDataLoader
from feature_engineering.demographic.DemographicDataLoader import DemographicDataLoader
from feature_engineering.math_grade_results.MathGradeResultsDataloader import MathGradeResultLoader
from utils.FileUtil import write_df_to_csv
import itertools
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn
import numpy as np
from linearmodels.panel import PanelOLS,RandomEffects

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
      
    # data=grade_rate_result_data.merge(economic_data, on=['Municipality', 'Year'], how='left')
    # data=data.merge(education_data, on=['Municipality', 'Year'], how='left')  
    # data=data.merge(demographic_data, on=['Municipality', 'Year'], how='left') 
    write_df_to_csv(data,folder_path, file_name,encoding='utf-8-sig')
    data.columns = [col.replace(" ", "_") for col in data.columns]
    return data
def show_correlation_matrix(df, image_path):
    plt.figure(figsize=(10, 8))
    corr = df.corr().round(2)
    seaborn.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation Matrix", fontsize=16)
    plt.tight_layout()
    plt.savefig(image_path, dpi=300)
    plt.close()
    return corr
def show_correlation_pair_plot(df, image_path):
    seaborn.pairplot(df, kind='reg')
    plt.suptitle("Pairwise Relationships with Regression Diagnostics", y=1.02)
    plt.savefig(image_path, dpi=300)
    plt.close() 
def math_grade_rate_plot(df):
    graph = seaborn.FacetGrid(df, col="Municipality", hue="Municipality", col_wrap=2, height=4, aspect=1.5)
    graph.map(seaborn.lineplot, "Year", "Math_grade_rate_result", marker="o")
    plt.savefig("data/math_grade_rate_result.png", dpi=500)
    plt.close()
def drop_highly_correlated_features(df, threshold=0.7):
    corr_matrix = df.corr().abs()
    print(corr_matrix)
    to_drop = set()
    for (row, col) in itertools.combinations(corr_matrix.columns, 2):
        if corr_matrix.loc[row, col] > threshold:
            if row != "Math_grade_rate_result" and col != "Math_grade_rate_result":
                if col not in to_drop:
                    print(f"Dropping {col} because of high correlation with {row} : {corr_matrix.loc[row, col]}")
                    to_drop.add(col)
    kept = [col for col in df.columns if col not in to_drop]
    
    return kept, list(to_drop)
def math_grade_rate_mean_plot(df):
    summary_df = df.groupby('Year', as_index=False)['Math_grade_rate_result'].mean()
    seaborn.lineplot(data=summary_df, x='Year', y='Math_grade_rate_result', marker='o', color='steelblue')
    plt.title("average math grade rate result over time (2015-2023)", fontsize=14)
    plt.xlabel("year")
    plt.ylabel("Math grade rate result")
    plt.savefig("data/average_math_grade_rate_result.png", dpi=300)
    plt.close()
def features_correlation_checking(df):
    income_vars = ["Math_grade_rate_result", "Average_income", "Income_inequality"]
    pop_vars = ["Math_grade_rate_result","Population_growth_rate", "Population"]
    edu_vars = ["Math_grade_rate_result", "education_share","Higher_education_percentage", "Lower_education_percentage", "Higher_to_lower_education_ratio"]

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
def check_data_VIF(df, thresh=5.0):
    X = df.dropna()  
    X = X.select_dtypes(include=[np.number]).astype(float)  
    drop_features = [] 

    while True:
        X_const = sm.add_constant(X) 
        vif_data = pd.DataFrame()  
        vif_data["feature"] = X_const.columns 
        vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]  # محاسبه VIF
        print(vif_data)
        vif_no_const = vif_data[vif_data["feature"] != "const"] 
        max_vif_value = vif_no_const["VIF"].max()  

        if max_vif_value > thresh: 
            drop_feature = vif_no_const.sort_values("VIF", ascending=False)["feature"].iloc[0]  
            print(f"Dropping {drop_feature} with VIF = {max_vif_value:.2f}")
            X = X.drop(columns=[drop_feature])  
            drop_features.append(drop_feature) 
        else:
            break 
    print("Dropped features:", drop_features)
    
    return drop_features  
def OLS_backward_elimination(X, y, significance_level=0.25):
    X = sm.add_constant(X)  
    model = sm.OLS(y, X).fit()
    while True:
        pvalues = model.pvalues.drop("const", errors='ignore')
        max_p_value = pvalues.max()
        if max_p_value < significance_level:
            break
        
        excluded_feature = pvalues.idxmax()
        X_temp= X.drop(columns=[excluded_feature])
        new_model = sm.OLS(y, X_temp).fit()
        if(new_model.rsquared_adj >= model.rsquared_adj):
            print(f"Dropping {excluded_feature} with p-value = {max_p_value:.4f} because Adj R²: {model.rsquared_adj:.4f} → {new_model.rsquared_adj:.4f}")
            model = new_model
            X= X_temp
        else:
            print(f"Keeping {excluded_feature} with p-value = {max_p_value:.4f} because Adj R²: {model.rsquared_adj:.4f}")
            break
    return model
def Fixed_effects_panel_backward_elimination(df, dep_var, features, entity, time, significance_level=0.25):
    df = df.set_index([entity, time])
    observation_size = len(df)  
    remaining_features = features.copy()
    prev_adj_r2 = -np.inf  
    while True:
        formula = f"{dep_var} ~ {' + '.join(remaining_features)} + EntityEffects + TimeEffects"
   

        #formula = f"{dep_var} ~ {' + '.join([f'\"{feat}\"' for feat in remaining_features])} + EntityEffects + TimeEffects"

        model = PanelOLS.from_formula(formula, data=df).fit()
        pvalues = model.pvalues[remaining_features]
        max_pval = pvalues.max()
        excluded_feature = pvalues.idxmax()

        if max_pval < significance_level:
            break
        

        parameter_size= len(model.params) 
        adj_r2 = 1 - ((1 - model.rsquared) * ( observation_size- 1)) / (observation_size - (parameter_size+1))
        if adj_r2>= prev_adj_r2:
            remaining_features.remove(excluded_feature)
            print(f"Dropping {excluded_feature} with p = {max_pval:.4f} because Adj R² {prev_adj_r2:.4f}-> {adj_r2:.4f}")
            prev_adj_r2 = adj_r2
        else:
            print(f"Keeping {excluded_feature} with p = {max_pval:.4f} because  Adj R² decreased stopped")
            break
        
 
    final_formula = f"{dep_var} ~ {' + '.join(remaining_features)} + EntityEffects + TimeEffects"
    final_model = PanelOLS.from_formula(final_formula, data=df).fit()
    return final_model
def Random_effects_panel_backward_elimination(df, dep_var, features, entity, time, significance_level=0.25):
    import numpy as np
    from linearmodels.panel import RandomEffects

    df = df.set_index([entity, time])
    observation_size = len(df)
    num_entities = df.index.get_level_values(entity).nunique()
    remaining_features = features.copy()
    prev_adj_r2 = -np.inf

    while True:

        if len(remaining_features) >= num_entities:
            print("Stopping: Too many predictors for available entities.")
            break

        formula = f"{dep_var} ~ {' + '.join(remaining_features)}"
        try:
            model = RandomEffects.from_formula(formula, data=df).fit()
        except ZeroDivisionError as e:
            print("Model fitting failed due to ZeroDivisionError:", e)
            break

        pvalues = model.pvalues[remaining_features]
        max_pval = pvalues.max()
        excluded_feature = pvalues.idxmax()

        if max_pval < significance_level:
            print("All remaining features are significant.")
            break

        parameter_size = len(model.params)
        adj_r2 = 1 - ((1 - model.rsquared) * (observation_size - 1)) / (observation_size - (parameter_size + 1))

        if adj_r2 >= prev_adj_r2:
            remaining_features.remove(excluded_feature)
            print(f"Dropping {excluded_feature} with p = {max_pval:.4f} because Adj R² {prev_adj_r2:.4f} -> {adj_r2:.4f}")
            prev_adj_r2 = adj_r2
        else:
            print(f"Keeping {excluded_feature} with p = {max_pval:.4f} because Adj R² decreased. Stopping.")
            break

    
    if remaining_features:
        final_formula = f"{dep_var} ~ {' + '.join(remaining_features)}"
        final_model = RandomEffects.from_formula(final_formula, data=df).fit()
        return final_model
    else:
        print("No features left to build a final model.")
        return None
def main():
    df=Merge_data('data/','merged_data.csv')
    # math_grade_rate_plot(df)
    # math_grade_rate_mean_plot(df)

    # selected_features=features_correlation_checking(df)
    # not_correlated_df=df[selected_features]
    # print(selected_features)

#     corr=show_correlation_matrix(df[[col for col in df.columns if col not in ['Year', 'Municipality']]], "data/correlation_matrix.png")
#    # show_correlation_pair_plot(df, "data/correlation_plot.png")
#     print(corr)
    
    keep_features, dropped_features = drop_highly_correlated_features(
        df[[col for col in df.columns if col not in ['Year', 'Municipality']]],0.6)
    

    dropped_features.extend(check_data_VIF(df[[col for col  in keep_features if col not in ['Year', 'Municipality','Math_grade_rate_result']]]) )
    #dropped_features=check_data_VIF(df[[col for col  in df.columns if col not in ['Year', 'Municipality','Math_grade_rate_result']]]) 
    selected_features=[col for col in df.columns if (col not in  dropped_features)  ]
    write_df_to_csv(df[selected_features], "data","process_data.csv")
  
    dropped_features.extend(['Municipality', 'Year'])
    features = [col for col in df.columns if col not in dropped_features and col != 'Math_grade_rate_result']
    
    

    # all_feature_OLS_model = OLS_backward_elimination(df[[col for col in df.columns if  col not in  ['Year', 'Municipality','Math_grade_rate_result']]],  
    #                                                 df.loc[df[[col for col in df.columns if  col not in  ['Year', 'Municipality']]].index, 'Math_grade_rate_result'])
    # print("OLS model with all features and Math_grade_rate_result as dependent variable results are as follows:")
    # print(all_feature_OLS_model.summary())
    # print("-----------------------------------------------------------------------------------------------------")
    
    # dummy_df = pd.get_dummies(df, columns=['Municipality','Year'], drop_first=True)
    # X = dummy_df[[col for col in dummy_df.columns if col != 'Math_grade_rate_result']].astype(float)
    # y = dummy_df['Math_grade_rate_result'].astype(float)

    # all_feature_OLS_model = OLS_backward_elimination(X,y)
    # print("OLS model with all features and Math_grade_rate_result and dummy for year and city as dependent variable results are as follows:")
    # print(all_feature_OLS_model.summary())
    # print("-----------------------------------------------------------------------------------------------------")
    
    
    # all_feature_fix_effective_panel_model = Fixed_effects_panel_backward_elimination(df, 'Math_grade_rate_result',
    #                                                 [col for col in df.columns if  col not in  ['Year', 'Municipality','Math_grade_rate_result']],
    #                                                 'Municipality', 'Year')
    # print("Fixed effects panel model with all features and Math_grade_rate_result as dependent variable results are as follows:")
    # print(all_feature_fix_effective_panel_model.summary)
    # print("-----------------------------------------------------------------------------------------------------")
 
    # all_feature_random_effective_panel_model = Random_effects_panel_backward_elimination(df, 'Math_grade_rate_result',
    #                                                 [col for col in df.columns if  col not in  ['Year', 'Municipality','Math_grade_rate_result']],
    #                                                 'Municipality', 'Year')
    # print("random effects panel model with all features and Math_grade_rate_result as dependent variable results are as follows:")
    # print(all_feature_random_effective_panel_model.summary)
    # print("-----------------------------------------------------------------------------------------------------")
 
 
    # OLS_model = OLS_backward_elimination(df[features],  df.loc[df[features].index, 'Math_grade_rate_result'])
    # print("OLS model with selected features based on  correlation checking and VIF and Math_grade_rate_result as dependent variable results are as followsVIF and Math_grade_rate_result as dependent variable results are as follows:")
    # print(OLS_model.summary())
    # print("-----------------------------------------------------------------------------------------------------")
    
    
    features_with_city_year = [col for col in df.columns if col not in dropped_features ]
    features_with_city_year.append('Municipality')
    features_with_city_year.append('Year')
    dummy_df = pd.get_dummies(df[features_with_city_year], columns=['Municipality','Year'], drop_first=True)
    X = dummy_df[[col for col in dummy_df.columns if col != 'Math_grade_rate_result']].astype(float)
    y = dummy_df['Math_grade_rate_result'].astype(float)
    
    OLS_model = OLS_backward_elimination(X, y)
    print("OLS model with selected features based on  correlation checking and VIF and dummy variables for year and city and Math_grade_rate_result as dependent variable results are as followsVIF and Math_grade_rate_result as dependent variable results are as follows:")
    print(OLS_model.summary())
    print("-----------------------------------------------------------------------------------------------------")
    
    
    
    # fix_effective_panel_model  = Fixed_effects_panel_backward_elimination(df[[col for col in df.columns if 
    #                                                     col not in dropped_features  
    #                                                     or col=='Municipality'
    #                                                     or col== 'Year']], 'Math_grade_rate_result',
    #                                                     features,
    #                                                     'Municipality', 'Year')
    
    # print("Fixed effects panel model with selected features based on  correlation checking and VIF and Math_grade_rate_result as dependent variable results are as followsVIF and Math_grade_rate_result as dependent variable results are as follows:")
    # print(fix_effective_panel_model.summary)
    # print("-----------------------------------------------------------------------------------------------------")
    
    
    # random_effective_panel_model = Random_effects_panel_backward_elimination(df[[col for col in df.columns if 
    #                                                     col not in dropped_features  
    #                                                     or col=='Municipality'
    #                                                     or col== 'Year']], 'Math_grade_rate_result',
    #                                                     features,
    #                                                     'Municipality', 'Year')
    
    # print("random effects panel model with selected features based on  correlation checking and VIF and Math_grade_rate_result as dependent variable results are as followsVIF and Math_grade_rate_result as dependent variable results are as follows:")
    # print(random_effective_panel_model.summary)
    # print("-----------------------------------------------------------------------------------------------------")
    
    
    
    
   
    
    # print("Done")



main()
