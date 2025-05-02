import seaborn as sns
import matplotlib.pyplot as plt

    
def math_grade_rate_plot(df,output_path):
    graph = sns.FacetGrid(df, col="Municipality", hue="Municipality", col_wrap=2, height=4, aspect=1.5)
    graph.map(sns.lineplot, "Year", "Math_grade_rate_result", marker="o")
    plt.savefig(output_path)
    plt.show()
    
    
def math_grade_rate_mean_by_municipality_plot(df,output_path):
    df_grouped = df.groupby('Municipality')['Math_grade_rate_result'].mean().reset_index()
    sns.barplot(x='Municipality', y='Math_grade_rate_result', data=df_grouped)
    plt.title('Average Math Grade Passing Rate by Municipality')
    plt.xticks(rotation=90)
    plt.savefig(output_path)
    plt.show()

def math_grade_rate_mean_by_year_plot(df,output_path):
    df_grouped = df.groupby('Year', as_index=False)['Math_grade_rate_result'].mean()
    sns.lineplot(data=df_grouped, x='Year', y='Math_grade_rate_result', marker='o', color='steelblue')
    plt.title("average math grade passing rate over time (2015-2023)", fontsize=14)
    plt.xlabel("year")
    plt.ylabel("Math grade passing rate")
    plt.savefig(output_path)
    plt.show()
    
def show_correlation_matrix(df, output_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig(output_path)
    plt.show()

def show_correlation_pair_plot(df, output_path):
    sns.pairplot(df)
    plt.title('Correlation Pair Plot')
    plt.savefig(output_path)
    plt.show()
