# Time Series Forecasting on Sales Dataset

## Problem Statement
You are tasked with analyzing the sales performance of a retail company over the past five years. The company operates both online and offline channels, and management is interested in understanding the sales trends, seasonality patterns, and the potential impact of external factors such as holidays, promotions, or economic indicators on their revenue.

Additionally, the company wants to forecast future sales to optimize inventory levels and staff allocation for the upcoming quarters.

## Process
- Clean Data
- Visualize Sales Trend over a long time period
- Check for Seasonality
- Analyze the effect of holidays on sales
- Split Data
- Create Models (Naive, Auto Regression, ARIMA)
- Model Visualizations/Forecasting
- Compare Models


## Results
- **Sales Trend** : The overall trend is sales is a slow increase. There are random spikes in sales, all varying in size, but the trend is an increase.
- **Seasonality** : There is no seasonality in the data at all. This shows that there are no patterns that occur that might impact sales.
- **Impact of holidays** : The holiday season shows to have a significant positive impact on sales. Sales made in the holiday season is consistently above the average amount of sales per month for the rest of the year.
- **Models** : The models have good forecasting of sales. The best performing model was the Naive Model, which has the lowest Mean Absolute Error.

# Conclusion
The sales in the comapany are not driven by any specific patterns of buying. The only time period in which sales are significantly increased are holiday seasons. Special holiday loyalty discounts can be offered to customers that buy outside of the holiday season to drive sales higher.