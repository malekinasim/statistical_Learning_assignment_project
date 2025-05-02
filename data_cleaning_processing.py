
from utils.FileUtil import write_df_to_csv
from utils.data_loader import load_data
from utils.data_preprocessing import merge_data
from utils.data_analysis import math_grade_rate_plot, math_grade_rate_mean_by_municipality_plot,math_grade_rate_mean_by_year_plot, show_correlation_matrix, show_correlation_pair_plot
from utils.data_preprocessing import process_highly_correlated_features

file_paths = {
    'politic': 'data/Dalarna_MunicipalElectionResults.xlsx',
    'economic': 'data/Dalarna_AverageIncome.xlsx',
    'education': 'data/Dalarna_Population_HigherEducation.xlsx',
    'demographic': 'data/Dalarna_Population_HigherEducation.xlsx',
    'grade_results': 'data/Result_AK9_Dalarna_Kum.xlsx'
}


def features_groups_correlation_checking(df):
    income_vars = ["Math_grade_rate_result", "Average_income", "Income_inequality"]
    pop_vars = ["Math_grade_rate_result","Population_growth_rate", "Population"]
    edu_vars = ["Math_grade_rate_result", "education_share","Higher_education_percentage", "Lower_education_percentage", "Higher_to_lower_education_ratio"]

    show_correlation_matrix(df[income_vars], "results/plot/correlation_matrix_income.png")
    show_correlation_pair_plot(df[income_vars], "results/plot/correlation_plot_income.png")
    
    show_correlation_matrix(df[pop_vars], "results/plot/correlation_matrix_population.png")
    show_correlation_pair_plot(df[pop_vars], "results/plot/correlation_plot_population.png")
    
    show_correlation_matrix(df[edu_vars], "results/plot/correlation_matrix_education.png")
    show_correlation_pair_plot(df[edu_vars], "data/correlation_plot_education.png")

def load_clean_data():
    # 1. Load data
    politic_data, economic_data, education_data, demographic_data, grade_rate_result_data = load_data(file_paths)

    # 2. Merge data
    df = merge_data(politic_data, economic_data, education_data, demographic_data, grade_rate_result_data)
    write_df_to_csv(df, "results/data/", "merged_data.csv")

    # 3. Visualization
    math_grade_rate_plot(df, 'results/plot/math_grade_passing_rate_trend.png')
    math_grade_rate_mean_by_municipality_plot(df, "results/plot/average_math_grade_rate_by_city_result.png")
    math_grade_rate_mean_by_year_plot(df, "results/plot/average_math_grade_rate_by_year_result.png")

    # 4. Correlation analysis
    features_groups_correlation_checking(df)
    show_correlation_matrix(df[[col for col in df.columns if col not in ['Year', 'Municipality']]], "results/plot/all_features_correlation_matrix.png")
    show_correlation_pair_plot(df, "results/plot/all_features_correlation_plot.png")

    # 5. Drop highly correlated and high-VIF features
    exclude_features = ['Year', 'Municipality', 'Math_grade_rate_result']
    keep_features, dropped_features, result_df = process_highly_correlated_features(df, exclude_features, corr_threshold=0.6, vif_threshold=5.0)

    # 6. Save processed data
    print("Highly correlated features are:", dropped_features)
    print("Features after dropping:", keep_features)
    write_df_to_csv(result_df, "results/data/", "cleaned_data.csv")
   