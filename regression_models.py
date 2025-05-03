import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
from linearmodels.panel import RandomEffects, PanelOLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from utils.FileUtil import write_df_to_csv

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data
def model_diagnostics(model,  plot_name):
    # Residuals
    try:
        residuals = model.resid
        fitted_values = model.fittedvalues
    except AttributeError:
        residuals = model.resids
        fitted_values = model.fitted_values
    
    # Residuals distribution
    sns.histplot(residuals, kde=True)
    plt.title(f'Residuals Distribution for {plot_name}')
    plt.savefig(f"results/plot/{plot_name}_residuals_distribution.png", dpi=500) 
    plt.close()

    # Residuals vs Fitted Values plot
    plt.scatter(fitted_values, residuals)
    plt.title(f'Residuals vs Fitted Values for {plot_name}')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.savefig(f"results/plot/{plot_name}_Residuals vs Fitted Values.png", dpi=500) 
    plt.close()
        
    # Residuals vs leverage
    if hasattr(model, "get_influence"): 
        fig, ax = plt.subplots(figsize=(8, 6))
        sm.graphics.influence_plot(model, ax=ax, criterion="cooks")
        plt.title(f"Residuals vs Leverage for {plot_name}")
        plt.savefig(f"data/{plot_name}_Residuals vs leverage.png", dpi=500)
        plt.close()
    else:   
        plt.scatter(fitted_values, residuals)
        plt.title(f"Residuals vs Fitted Values (Panel Model) for {plot_name}")
        plt.xlabel("Fitted Values")
        plt.ylabel("Residuals")
        plt.axhline(0, color='red', linestyle='--')
        plt.savefig(f"results/plot/{plot_name}_residuals_vs_fitted_panel.png", dpi=500)
        plt.close()
    X = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
    
   
    vif_data=calculate_vif(X)
   
    #write_df_to_csv(vif_data, f"results/plot/{plot_name}_VIF.csv")
 
    print(vif_data)    
    
    
    
    return residuals, fitted_values


def OLS_full_model(X, y):
    # Add constant to the data
    X = sm.add_constant(X)
    
    # Fit OLS regression model
    model = sm.OLS(y, X).fit()    
    return model


# def OLS_backward_elimination(X, y, significance_level=0.25, vif_threshold=10.0):
#     X = sm.add_constant(X)  
#     model = sm.OLS(y, X).fit()
    
#     while True:
#         # Step 1: P-value elimination
#         pvalues = model.pvalues.drop("const", errors='ignore')
#         max_p_value = pvalues.max()

#         if max_p_value >= significance_level:
#             excluded_feature = pvalues.idxmax()
#             X_temp = X.drop(columns=[excluded_feature])
#             new_model = sm.OLS(y, X_temp).fit()

#             if new_model.rsquared_adj >= model.rsquared_adj:
#                 print(f"Dropping '{excluded_feature}' due to high p-value = {max_p_value:.4f} (Adj R² improved: {model.rsquared_adj:.4f} → {new_model.rsquared_adj:.4f})")
#                 X = X_temp
#                 model = new_model
#                 continue
#             else:
#                 print(f"Retained '{excluded_feature}' despite high p-value = {max_p_value:.4f} (Adj R² drop)")
        
#         # Step 2: VIF elimination
#         vif = calculate_vif(X.drop(columns=["const"]))
#         high_vif = vif[vif["VIF"] > vif_threshold]

#         if not high_vif.empty:
#             worst_vif_feature = high_vif.sort_values("VIF", ascending=False).iloc[0]["feature"]
#             print(f"Dropping '{worst_vif_feature}' due to high VIF = {high_vif['VIF'].max():.2f}")
#             X = X.drop(columns=[worst_vif_feature])
#             model = sm.OLS(y, X).fit()
#             continue
        
#         # If no features were dropped in either step, we're done
#         break

#     return model


def OLS_backward_elimination(X, y, significance_level=0.1):
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
def fixed_effects_panel_backward_elimination(df, dep_var, features, entity, time, significance_level=0.25):
    df = df.set_index([entity, time])
    observation_size = len(df)  
    remaining_features = features.copy()
    prev_adj_r2 = -np.inf  
    while True:
        formula = f"{dep_var} ~ {' + '.join(remaining_features)} + EntityEffects + TimeEffects"

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
def random_effects_panel_backward_elimination(df, dep_var, features, entity, time, significance_level=0.25):
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


def plot_actual_vs_predicted(model, y, plot_name):
    try:
     fitted_values = model.fittedvalues
    except AttributeError:
       fitted_values = model.fitted_values

    plt.figure(figsize=(8, 6))
    plt.scatter(y, fitted_values, color='blue', label='Predicted vs Actual')
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Perfect Fit')  # خط مناسب
    plt.title(f'Actual vs Predicted Values for {plot_name}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.savefig(f"results/plot/{plot_name}_actual_vs_predicted.png", dpi=500)
    plt.show()
    
    
def hat_Value_plot(model,  plot_name):

    hat_values = model.get_influence().hat_matrix_diag
    plt.scatter(range(len(hat_values)), hat_values)
    plt.axhline(y=2 * np.mean(hat_values), color='r', linestyle='--')
    plt.title(f'Hat Value Plot for Logistic Regression for {plot_name}')
    plt.xlabel('Observations')
    plt.ylabel('Hat Values')
    plt.savefig(f"results/plot/{plot_name}_hat_value_plot.png", dpi=500)
    plt.show()








