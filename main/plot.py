import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# function to plot stress-strain data
def plot_stress_strain(data: pd.DataFrame, title: str):
    plt.plot(data['strain'], data['stress'])
    plt.title(title)
    plt.xlabel('Strain [-]')
    plt.ylabel('Engineering Stress [MPA]')
    plt.show()

# function to create  matrix for data frame
def create_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Változók páronkénti korrelációja')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()