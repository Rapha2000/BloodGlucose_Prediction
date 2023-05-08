# %% [markdown]
# # Data Analysis

# %% [markdown]
# Here we will explore and analyze our data. 

# %% [markdown]
# ### Dataset

# %% [markdown]
# Here we focus only on patient 559's blood glucose level during the 2018 study period

# %% [markdown]
# ### Objectives

# %% [markdown]
# 1. Analyze blood glucose values and try to understand some mechanisms of blood glucose evolution
# 2. Find patterns in our data 

# %% [markdown]
# ### Loading and visualizing our data

# %%
# import of useful modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates as mpld
from datetime import date
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima

# %%
# pre-processing
train_df = pd.read_csv('data/csv/2018/train/559-ws-training-glucose_level.csv')
test_df = pd.read_csv('data/csv/2018/test/559-ws-testing-glucose_level.csv')

train_df.index = pd.to_datetime(train_df['ts'], format = "%d-%m-%Y %H:%M:%S")
test_df.index = pd.to_datetime(test_df['ts'], format = "%d-%m-%Y %H:%M:%S")

train_df = train_df.drop('ts', axis=1)
test_df = test_df.drop('ts', axis=1)

# %%
# data insights
print("Blood Glucose Level")
print(train_df.describe())

print("\nHead and tail of the serie")
print(train_df.head())
print(train_df.tail())

print("\nTemporal range of training & testing data: ") 
print("training: ", train_df.index.max() - train_df.index.min())
print("testing: ", test_df.index.max() - test_df.index.min())

# %%
# visualization
fig, ax = plt.subplots()
ax.plot(train_df.index, train_df['value'], label='Training set')
ax.plot(test_df.index, test_df['value'], label='Testing set')
ax.set_xticklabels(ax.get_xticklabels(), rotation='vertical')
plt.title("Blood Glucose Level (07-12-2021 : 17-01-2022)")
plt.xlabel("Time")
plt.ylabel("BG Level")
ax.legend()
plt.show()

# %% [markdown]
# We can see **great variability** in our data, with BG level reaching significant hyperglycemic values (>250 mg/dL) several times. Out of curiosity, on these 7 and a half weeks of training data, we can look at the percentage of time spent in each glycemic zone by the individual.

# %%
# extraction of blood glucose values >180 mg/dL
high_glyc = train_df.loc[train_df['value'] > 180]

# extraction of blood glucose values <80 mg/dL
low_glyc = train_df.loc[train_df['value'] < 80]

# percentages spent in hypoglycemia and hyperglycemia
print("Time spent with blood glucose above 180:", 100*len(high_glyc)/len(train_df['value']))
print("Time spent with blood glucose below 80:",100*len(low_glyc)/len(train_df['value']) )


# %% [markdown]
# Therefore, we notice that this individual is **40% of the time in hyperglycemia** and **7% of the time in hypoglycemia**, which means that he is in the **target zone only 53% of the time**.
# 
# This first indicator is a great way to get an idea of the glycemic balance of a diabetic. For this individual, it is therefore understandable that the usefulness of a decision support system could be greatly enhanced to try to maintain his or her blood sugar within the target zone. 

# %% [markdown]
# ### Seasonal Decompose of the time series

# %% [markdown]
# Seasonal decompose is a method used to decompose the components of a time series into the following:
# * Trend - increasing or decreasing value in the series.
# * Seasonality - repeating short-term cycle in the series.
# * Noise - random variation in the series.
# 
# The analysis of the components individually provide better insights for model selection.
# 

# %%
# additive decomposition
result_add = seasonal_decompose(train_df, period=288, model='additive') # 288 = 24*60/5 <--> 24h-period
result_add.plot()
plt.gcf().autofmt_xdate()
date_format = mpld.DateFormatter('%y-%m-%d')
plt.gca().xaxis.set_major_formatter(date_format)

# multiplicative decomposition
result_mul = seasonal_decompose(train_df, period=288, model='multiplicative')
result_mul.plot()
plt.gcf().autofmt_xdate()
date_format = mpld.DateFormatter('%y-%m-%d')
plt.gca().xaxis.set_major_formatter(date_format)
plt.show()

# %% [markdown]
# Although these graphs are not very visible, we can already see that the series seems stationary (no particular trend), but that it seems to have a **seasonality**.
# 
# Let's try to reduce the time window to better perceive this seasonality. 

# %%
# we first look at what happens on a weekly basis
week_1 = (train_df.index >= "2021-12-07") & (train_df.index <= "2021-12-14")
week_2 = (train_df.index >= "2021-12-14") & (train_df.index <= "2021-12-21")
week_3 = (train_df.index >= "2021-12-21") & (train_df.index <= "2021-12-28")
week_4 = (train_df.index >= "2021-12-28") & (train_df.index <= "2022-01-04")
week_5 = (train_df.index >= "2022-01-04") & (train_df.index <= "2022-01-11")
week_6 = (train_df.index >= "2022-01-11") 

week_1_df = train_df.loc[week_1]
week_2_df = train_df.loc[week_2]
week_3_df = train_df.loc[week_3]
week_4_df = train_df.loc[week_4]
week_5_df = train_df.loc[week_5]
week_6_df = train_df.loc[week_6]

# we then look at what happens on a daily basis
week_1_day_1_df = train_df.loc[train_df.index.date == date(2021,12,7)]
week_1_day_2_df = train_df.loc[train_df.index.date == date(2021,12,8)]
week_1_day_3_df = train_df.loc[train_df.index.date == date(2021,12,9)]
week_1_day_4_df = train_df.loc[train_df.index.date == date(2021,12,10)]
week_1_day_5_df = train_df.loc[train_df.index.date == date(2021,12,11)]
week_1_day_6_df = train_df.loc[train_df.index.date == date(2021,12,12)]
week_1_day_7_df = train_df.loc[train_df.index.date == date(2021,12,13)]

# %% [markdown]
# #### Week 1

# %%
result_add = seasonal_decompose(week_1_df, period=288, model='additive')
result_add.plot()
plt.gcf().autofmt_xdate()
date_format = mpld.DateFormatter('%y-%m-%d')
plt.gca().xaxis.set_major_formatter(date_format)

result_mul = seasonal_decompose(week_1_df, period=288, model='multiplicative')
result_mul.plot()
plt.gcf().autofmt_xdate()
date_format = mpld.DateFormatter('%y-%m-%d')
plt.gca().xaxis.set_major_formatter(date_format)
plt.show()

# %% [markdown]
# #### Week 2

# %%
result_add = seasonal_decompose(week_2_df, period=288, model='additive')
result_add.plot()
plt.gcf().autofmt_xdate()
date_format = mpld.DateFormatter('%y-%m-%d')
plt.gca().xaxis.set_major_formatter(date_format)

result_mul = seasonal_decompose(week_2_df, period=288, model='multiplicative')
result_mul.plot()
plt.gcf().autofmt_xdate()
date_format = mpld.DateFormatter('%y-%m-%d')
plt.gca().xaxis.set_major_formatter(date_format)
plt.show()

# %% [markdown]
# #### Week 3

# %%
result_add = seasonal_decompose(week_3_df, period=288, model='additive')
result_add.plot()
plt.gcf().autofmt_xdate()
date_format = mpld.DateFormatter('%y-%m-%d')
plt.gca().xaxis.set_major_formatter(date_format)

result_mul = seasonal_decompose(week_3_df, period=288, model='multiplicative')
result_mul.plot()
plt.gcf().autofmt_xdate()
date_format = mpld.DateFormatter('%y-%m-%d')
plt.gca().xaxis.set_major_formatter(date_format)
plt.show()

# %% [markdown]
# #### Week 4

# %%
result_add = seasonal_decompose(week_4_df, period=288, model='additive')
result_add.plot()
plt.gcf().autofmt_xdate()
date_format = mpld.DateFormatter('%y-%m-%d')
plt.gca().xaxis.set_major_formatter(date_format)

result_mul = seasonal_decompose(week_4_df, period=288, model='multiplicative')
result_mul.plot()
plt.gcf().autofmt_xdate()
date_format = mpld.DateFormatter('%y-%m-%d')
plt.gca().xaxis.set_major_formatter(date_format)
plt.show()

# %% [markdown]
# #### Week 5

# %%
result_add = seasonal_decompose(week_5_df, period=288, model='additive')
result_add.plot()
plt.gcf().autofmt_xdate()
date_format = mpld.DateFormatter('%y-%m-%d')
plt.gca().xaxis.set_major_formatter(date_format)

result_mul = seasonal_decompose(week_5_df, period=288, model='multiplicative')
result_mul.plot()
plt.gcf().autofmt_xdate()
date_format = mpld.DateFormatter('%y-%m-%d')
plt.gca().xaxis.set_major_formatter(date_format)
plt.show()

# %% [markdown]
# #### Week 6

# %%
result_add = seasonal_decompose(week_6_df, period=288, model='additive')
result_add.plot()
plt.gcf().autofmt_xdate()
date_format = mpld.DateFormatter('%y-%m-%d')
plt.gca().xaxis.set_major_formatter(date_format)

result_mul = seasonal_decompose(week_6_df, period=288, model='multiplicative')
result_mul.plot()
plt.gcf().autofmt_xdate()
date_format = mpld.DateFormatter('%y-%m-%d')
plt.gca().xaxis.set_major_formatter(date_format)
plt.show()

# %% [markdown]
# We can notice that there is a daily seasonality. Let's analyse it more precisely.

# %% [markdown]
# #### Days of the week 1

# %%
days_df = [week_1_day_1_df,week_1_day_2_df,week_1_day_3_df,week_1_day_4_df,week_1_day_5_df,week_1_day_6_df,
           week_1_day_7_df]

fig, axs = plt.subplots(nrows=7, ncols=2, figsize=(12, 24))

for i, day_df in enumerate(days_df):
        # i = days_df.index(day_df)
        result_add = seasonal_decompose(day_df, period=12, model='additive')
        result_mul = seasonal_decompose(day_df, period=12, model='multiplicative')
        
        axs[i, 0].plot(result_add.trend.index, result_add.trend.values, label='Trend')
        axs[i, 0].plot(result_add.seasonal.index, result_add.seasonal.values, label='Seasonality')
        axs[i, 0].plot(result_add.resid.index, result_add.resid.values, label='Residuals')
        axs[i, 0].set_title(f'Day {i+1} - Additive Decomposition')
        axs[i, 0].legend()
        axs[i, 0].xaxis.set_major_formatter(mpld.DateFormatter('%y-%m-%d'))
        plt.setp(axs[i, 0].get_xticklabels(), rotation=30, ha="right")
        
        axs[i, 1].plot(result_mul.trend.index, result_mul.trend.values, label='Trend')
        axs[i, 1].plot(result_mul.seasonal.index, result_mul.seasonal.values, label='Seasonality')
        axs[i, 1].plot(result_mul.resid.index, result_mul.resid.values, label='Residuals')
        axs[i, 1].set_title(f'Day {i+1} - Multiplicative Decomposition')
        axs[i, 1].legend()
        axs[i, 1].xaxis.set_major_formatter(mpld.DateFormatter('%y-%m-%d'))
        plt.setp(axs[i, 1].get_xticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.show()




# %% [markdown]
# No particular trend nor particular seasonality

# %% [markdown]
# ### Observations from Seasonal Decompose

# %% [markdown]
# 
# 1. The time serie seems to have no particular trend
# 2. A daily seasonality seems to be observed in our data 
# 

# %% [markdown]
# ### Model Selection (BROUILLON)

# %% [markdown]
# #### ARMA

# %% [markdown]
# From the above observations we can conclude that the ARMA (AutoRegressive Moving Average) model would be an appropriate choice as there is a constant seasonality component along with no particular trend.
# 
# The ARMA model uses the past values of the series to calculate a seasonal moving average that is used to predict future values. The seasonal moving average is obtained by averaging the values for the same season of the year over several previous periods. The size of the seasonal moving average window should be chosen according to the length of seasonality in the series.

# %%
result_stat = adfuller(train_df)
print('p-value:', result_stat[1])
print("Donc la série est stationnaire, on peut appliquer un modèle ARMA")

# %% [markdown]
# The parameters p, d, and q are the three key parameters of the ARMA (AutoRegressive Moving Average) method and must be chosen carefully to obtain good prediction performance.
# 
# * p corresponds to the order of the autoregression, i.e. the number of past terms of the series that are used to predict the current value.
# * q is the order of the moving average, i.e. the number of past terms of the prediction errors that are included in the current prediction.
# * d is the order of differentiation, i.e. the number of times the series must be differentiated to become stationary.
# 
# We will use the statsmodels package to determine the orders p,q and d. This library uses the **Akaike Information Criterion (AIC)** and **Bayesian Information Criterion (BIC)** to select the optimal ARIMA orders.

# %%
model = auto_arima(train_df, seasonal=True, m=24, trace=True)

# %%
model_arma = ARIMA(train_df, order=(2, 1, 2))
results_arma = model_arma.fit()
forecast_values_arma = results_arma.forecast(steps=test_df.shape[0])

# %%
plt.figure(figsize = (15,10))
plt.plot(train_df.index, train_df, marker="o", color="black", label='Actuals Train')
plt.plot(test_df.index, test_df, marker="o", color = "red", label = 'Actuals Test')
plt.plot(train_df.index, results_arma.fittedvalues, marker=".", color="blue", label='Predictions Train')
plt.plot(test_df.index, forecast_values_arma, marker=".", color = "green", label = 'Predictions Test')
plt.ylabel("Glycémie (mg/dL)")
plt.xlabel("Date")
plt.title("Evolution de la glycémie")
plt.legend(loc='best')
plt.show()

# %%
rmse_arma = np.sqrt(mean_squared_error(test_df, forecast_values_arma))
print("RMSE ARMA: ", rmse_arma)

# %% [markdown]
# #### Simple Exponential Smoothing (BROUILLON)

# %%
print("Dimensions du jeu d'entrainement = {}, soit = {} %".format(train_df.shape, 100*len(train_df)/(len(train_df)+len(test_df))))
print("Dimensions du jeu de test = {}, soit = {} %".format(test_df.shape, 100*len(test_df)/(len(train_df)+len(test_df))))

# %%
# On convertit notre jeu d'entraînement au bon format
train_df_ses = train_df.values.tolist()

# On définit la méthode avec laquelle on veut traiter nos données
ses_bg = SimpleExpSmoothing(train_df_ses, initialization_method = 'heuristic')

# On implémente un modèle où l'on applique notre méthode avec un paramètre alpha que l'on vient préciser
alpha = 0.01
model_ses = ses_bg.fit(smoothing_level = alpha, optimized = False)

# On prédit les valeurs sur notre jeu de test
forecast_values_ses = model_ses.forecast(test_df.shape[0])


# %%
plt.figure(figsize = (15,10))
plt.plot(train_df.index, train_df_ses, marker="o", color="black", label='Actuals Train')
plt.plot(test_df.index, test_df, marker="o", color = "red", label = 'Actuals Test')
plt.plot(train_df.index, model_ses.fittedvalues, marker=".", color="blue", label='Predictions Train')
plt.plot(test_df.index, forecast_values_ses, marker=".", color = "green", label = 'Predictions Test')
plt.ylabel("Glycémie (mg/dL)")
plt.xlabel("Date")
plt.title("Evolution de la glycémie")
plt.legend(loc='best')
plt.show()

# %%
rmse_ses = np.sqrt(mean_squared_error(test_df, forecast_values_ses))
print("RMSE Lissage Exponentiel Simple: ", rmse_ses)

# %% [markdown]
# On arrive à prédire la glycémie d'un patient à **68,39 mg/dL** près en moyenne, ce qui est une **mauvaise prédiction**.

# %% [markdown]
# #### Holt Winter's Seasonal Model (BROUILLON)

# %%
holt_bg = ExponentialSmoothing(train_df_ses, seasonal_periods = 288, seasonal="add", use_boxcox = False, initialization_method = "heuristic") 
model_holt = holt_bg.fit()
forecast_values_holt = model_holt.forecast(test_df.shape[0])

# %%
plt.figure(figsize = (15,10))
plt.plot(train_df.index, train_df_ses, marker="o", color="black", label='Actuals Train')
plt.plot(test_df.index, test_df, marker="o", color = "red", label = 'Actuals Test')
plt.plot(train_df.index, model_holt.fittedvalues, marker=".", color="blue", label='Predictions Train')
plt.plot(test_df.index, forecast_values_holt, marker=".", color = "green", label = 'Predictions Test')
plt.ylabel("Glycémie (mg/dL)")
plt.xlabel("Date")
plt.title("Evolution de la glycémie")
plt.legend(loc='best')
plt.show()

# %%
rmse_holt = np.sqrt(mean_squared_error(test_df, forecast_values_holt))
print("RMSE Holt Winter: ", rmse_holt)


