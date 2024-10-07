import matplotlib.pyplot as plt
import seaborn as sns

def plot_risk_distribution(df):
    """Plot the distribution of credit risk categories"""
    sns.countplot(data=df, x='Credit_Score_Category')
    plt.title('Distribution of Credit Score Categories')
    plt.xlabel('Risk Category')
    plt.ylabel('Count')
    plt.show()

def plot_transaction_trends(df):
    """Plot trends in transaction amount over time"""
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df = df.sort_values('TransactionStartTime')
    plt.figure(figsize=(10, 6))
    plt.plot(df['TransactionStartTime'], df['Total_Transaction_Amount'], label='Total Transaction Amount')
    plt.title('Transaction Trends Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Transaction Amount')
    plt.legend()
    plt.show()
