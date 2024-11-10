**Time Series Chart**

# Re-import necessary libraries and data for a clean analysis
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Reload the Apple Financials data
xls = pd.ExcelFile('/mnt/data/Apple_Financials_Merged_Temitope Adebanjo.xlsx')
historical_performance_df = pd.read_excel(xls, '2. Historical Performance Analy', skiprows=2)

# Rename columns for clarity and clean up
historical_performance_df.columns = [
    "Index", "Year", "Revenue", "Net Income", "Revenue Growth", 
    "Net Income Growth", "Operating Income", "Operating Margin", "Net Profit Margin"
]
historical_performance_df = historical_performance_df.dropna().reset_index(drop=True)
historical_performance_df = historical_performance_df[["Year", "Revenue", "Net Income", "Operating Income"]]
historical_performance_df["Year"] = historical_performance_df["Year"].astype(int)

# Convert financial data columns to numeric types
historical_performance_df["Revenue"] = pd.to_numeric(historical_performance_df["Revenue"], errors='coerce')
historical_performance_df["Net Income"] = pd.to_numeric(historical_performance_df["Net Income"], errors='coerce')
historical_performance_df["Operating Income"] = pd.to_numeric(historical_performance_df["Operating Income"], errors='coerce')

# Time Series Chart Model
plt.figure(figsize=(10, 6))
plt.plot(historical_performance_df["Year"], historical_performance_df["Revenue"], label="Revenue")
plt.plot(historical_performance_df["Year"], historical_performance_df["Net Income"], label="Net Income")
plt.plot(historical_performance_df["Year"], historical_performance_df["Operating Income"], label="Operating Income")

plt.title("Time Series Chart of Apple's Financial Metrics")
plt.xlabel("Year")
plt.ylabel("Amount (in Million Dollars)")
plt.legend()
plt.grid(True)
plt.show()



******Clustering**
# Redefine the KMeans clustering instance
kmeans = KMeans(n_clusters=3, random_state=42)

# Apply KMeans clustering
clusters = kmeans.fit_predict(scaled_data)

# Add the clusters back to the DataFrame
historical_performance_df["Cluster"] = clusters

# Plotting the clustered data
plt.figure(figsize=(10, 6))
plt.scatter(historical_performance_df["Year"], historical_performance_df["Revenue"], c=historical_performance_df["Cluster"], cmap='viridis', s=100)
plt.title("Clustering of Apple's Financial Years Based on Revenue")
plt.xlabel("Year")
plt.ylabel("Revenue (in Million Dollars)")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.show()

![image](https://github.com/user-attachments/assets/e627a4c8-3215-4a8f-bf5d-418a7d127f90)


**
Forecast Model **
# Forecast Model: Predicting future Revenue using ARIMA model
from statsmodels.tsa.arima.model import ARIMA

# Setting up the ARIMA model for the Revenue data
# We'll use a simple ARIMA(1,1,1) model as an initial test model
revenue_series = historical_performance_df.set_index("Year")["Revenue"]
model = ARIMA(revenue_series, order=(1, 1, 1))
model_fit = model.fit()

# Forecast for the next 3 years
forecast = model_fit.forecast(steps=3)

# Plot the historical revenue along with the forecast
plt.figure(figsize=(10, 6))
plt.plot(revenue_series, label="Historical Revenue")
plt.plot(forecast.index, forecast, label="Forecasted Revenue", linestyle='--')
plt.title("Forecast Model: Projected Revenue for Future Years")
plt.xlabel("Year")
plt.ylabel("Revenue (in Million Dollars)")
plt.legend()
plt.grid(True)
plt.show()







**OUTLIERS**
# Outliers Model: Detecting unusual changes in Revenue growth rates
# Calculating the year-over-year growth rate for Revenue
historical_performance_df["Revenue Growth"] = historical_performance_df["Revenue"].pct_change() * 100

# Identifying outliers using Z-score method for Revenue Growth
revenue_growth_mean = historical_performance_df["Revenue Growth"].mean()
revenue_growth_std = historical_performance_df["Revenue Growth"].std()
z_scores = (historical_performance_df["Revenue Growth"] - revenue_growth_mean) / revenue_growth_std

# Defining outliers as those where the Z-score is above 2 or below -2
outliers = historical_performance_df[(z_scores > 2) | (z_scores < -2)]

# Plotting Revenue Growth with highlighted outliers
plt.figure(figsize=(10, 6))
plt.plot(historical_performance_df["Year"], historical_performance_df["Revenue Growth"], label="Revenue Growth")
plt.scatter(outliers["Year"], outliers["Revenue Growth"], color="red", label="Outliers", s=100)

plt.title("Outliers Model: Detection of Unusual Revenue Growth Rates")
plt.xlabel("Year")
plt.ylabel("Revenue Growth (%)")
plt.legend()
plt.grid(True)
plt.show()
![image](https://github.com/user-attachments/assets/21be7bc0-f7b4-48e0-b4ae-842dca937aac)



Each model is now complete, providing insights through:

Time Series Chart: Historical trends in key financial metrics.
Clustering Model: Segmenting years based on revenue and income.
Forecast Model: Projected revenue for the coming years.
Outliers Model: Identification of years with abnormal revenue growth.

