from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import sys
import statsmodels as sm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
# from sklearn.preprocessing import StandardScaler,MinMaxScaler


# import deep_learning_module
# import data_module


siteHeader = st.beta_container()


modelTraining = st.beta_container()


with siteHeader:
    st.title('TECHNOLOGY SALES FORECASTING')
    st.text('')

with st.beta_expander("DATASET"):
    st.header('SUPERSTORE DATASET')
    st.text('dataset is taken from tableau sample dataset, this data contains from year 2014 - 2017')
    sales_data = pd.read_csv('filtered_Superstore.csv')
    
    option = sales_data['Category'].unique()
    select_category = st.sidebar.selectbox('Select Category :',option,0)
    
    technology = sales_data.loc[sales_data['Category'] == select_category]
    st.dataframe(technology)
    
    technology = technology.sort_values('Order Date')
    technology.isnull().sum()
    technology = technology.groupby('Order Date')['Sales'].sum().reset_index()
    technology['Order Date'] = pd.to_datetime(technology['Order Date'])
    technology.set_index('Order Date', inplace=True)
    # st.dataframe(technology)
    y = technology['Sales'].resample('MS').mean()
    y.head()
    plt.figure(figsize=(20,10))
    plt.grid()
    plt.plot(y)
    plt.title(select_category)
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.show()
    st.write('Total Sales for '+select_category)
    st.pyplot(plt)
    decomposition = sm.tsa.seasonal_decompose(y, model='additive')
    fig = decomposition.plot()
    st.write('Decomposition Graph')
    st.pyplot(fig)

    def print_adf_result(adf_result):
        df_results = pd.Series(adf_result[0:4], index=['ADF Test Statistic','P-Value','# Lags Used','# Observations Used'])
    
        for key, value in adf_result[4].items():
            df_results['Critical Value (%s)'% key] = value
        print('Augmented Dickey-Fuller Test Results:')
        print(df_results)
    

        result = adfuller(y, maxlag=12)
        print_adf_result(result)


with st.beta_expander("SARIMA MODEL"):

    st.header('SARIMA MODEL')
    st.text('we are using sarima model to do the time-series forecasting')
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                # st.write('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue

    sarimax = SARIMAX(y, order=(0,1,1), seasonal_order=(0,1,1,12)).fit()
    st.write('Model Summary')
    sarimax.summary()

    st.pyplot(sarimax.plot_diagnostics(figsize=(20, 10)))
    
    residuals =pd.Series(sarimax.resid)
    def check_residuals(series):
        fig = plt.figure(figsize=(20, 10))    
        gs = fig.add_gridspec(2,2)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(series)
        ax1.set_title('residuals')
        
        ax2 = fig.add_subplot(gs[1,0])
        plot_acf(series, ax=ax2, title='ACF')
        
        ax3 = fig.add_subplot(gs[1,1])
        sns.kdeplot(series, ax=ax3)
        ax3.set_title('density')
        st.write('Residual , ACF & Density')
        st.pyplot(plt,True)
        
    check_residuals(residuals)


    #Compare forecast data and observed data with SARIMA
    pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
    pred_ci = pred.conf_int()
    ax = y['2014':].plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=0.2)
    ax.set_xlabel('Date')
    ax.set_ylabel(select_category)
    plt.legend()
    st.write('Comparison forcast data and observed data with SARIMA model')
    st.pyplot(plt,True)

    y_forecasted = pred.predicted_mean
    y_truth = y['2017-01-01':]
    mse = ((y_forecasted - y_truth) ** 2).mean()
    #print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
    st.write('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

    #Forecast sales for future years
    sarimax_forecast = sarimax.get_forecast(48)
    sarimax_forecast_conf_int = sarimax_forecast.conf_int()

    plt.plot(y, label='observed')
    plt.plot(sarimax_forecast.predicted_mean, label='forecast')


    plt.fill_between(sarimax_forecast_conf_int.index,
                    sarimax_forecast_conf_int.iloc[:, 0],
                    sarimax_forecast_conf_int.iloc[:, 1], color='k', alpha=.2)

    plt.legend()
    st.write('Forecast sales for future years')
    st.pyplot(plt,True)


