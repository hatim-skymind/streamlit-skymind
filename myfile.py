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
    st.title('Welcome to the Awesome project!')
    st.text('In this project I look into ... And I try ... I worked with the dataset from ...')

with st.beta_expander("DATASET"):
    st.header('SALES FORECASTING DATASET')
    st.text('I found this dataset at...  I decided to work with it because ...')
    sales_data = pd.read_csv('train.csv')
    # Category = sales_data['Category'].drop_duplicates()
    # select_category = st.sidebar.selectbox('Select Category :',Category)
    technology = sales_data.loc[sales_data['Category'] == 'Technology']
    st.dataframe(technology)
    cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name']
    technology.drop(cols, axis=1, inplace=True)
    technology = technology.sort_values('Order Date')
    technology.isnull().sum()
    technology = technology.groupby('Order Date')['Sales'].sum().reset_index()
    technology['Order Date'] = pd.to_datetime(technology['Order Date'])
    technology.set_index('Order Date', inplace=True)
    st.dataframe(technology)
    y = technology['Sales'].resample('MS').mean()
    y.head()
    plt.figure(figsize=(20,10))
    plt.grid()
    plt.plot(y)
    plt.title('Technology Sales')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.show()
    st.pyplot(plt)
    decomposition = sm.tsa.seasonal_decompose(y, model='additive')
    fig = decomposition.plot()
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
    st.text('Let\'s take a look into the features I generated.')
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    st.write('Examples of parameter combinations for Seasonal ARIMA...')
    st.write('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    st.write('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    st.write('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    st.write('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

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
    ax.set_ylabel('Technology Sales')
    plt.legend()
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
    st.pyplot(plt,True)


# with st.beta_expander("LSTM MODEL"):
#     num_epochs = 100
#     split_ratio = 0.70
#     batch_size = 2
#     window_size = 2
#     _step = 2

#     split_data = round(len(y)*split_ratio)
#     train_data = y[:split_data]
#     test_data = y[split_data:]
#     train_time = train_data.index
#     test_time = test_data.index
#     st.write("train_data_shape")
#     st.write(train_data.shape)
#     st.write("test_data_shape")
#     st.write(test_data.shape)

#     scaler = StandardScaler().fit(train_data.values.reshape(-1,1))
#     scaler_train_data = scaler.transform(train_data.values.reshape(-1,1))
#     scaler_test_data = scaler.transform(test_data.values.reshape(-1,1))
#     st.write(f"scaler_train_data shape : {scaler_train_data.shape}")
#     st.write(f"scaler_test_data shape : {scaler_test_data.shape}")

#     #Data sequencing
#     trainX ,trainY =  data_module.univariate_multi_step(scaler_train_data,window_size,n_step)
#     testX , testY = data_module.univariate_multi_step(scaler_test_data,window_size,n_step)

#     st.write(f"trainX shape:{trainX.shape} trainY shape:{trainY.shape}")
#     st.write(f"testX shape:{testX.shape} testX shape:{testY.shape}")

#     def key_assign(trainingX,testingX,trainingY,testingY):
#         """ 
#         Use to assign  the key to create the train_data_dict and test_data_dict   
#         Arguments:
#         trainingX -- feature for traning data 
#         testingX -- feature for testing data
#         trainingY -- label for traning data
#         testingY -- label for testing data   
#         Returns: 
#         train_data_dict -- dictionary of trainingX and trainingY
#         test_data_dict -- dictionary of testingX and testingY
#         """    
#         # Create a dictionary that can store the train set feature and label
#         train_data_dict = {"train_data_x_feature" : trainingX, "train_data_y_label" : trainingY}
        
#         # Create a dictionary that can store the test set feature and label
#         test_data_dict  = {"test_data_x_feature" : testingX , "test_data_y_label" : testingY }

#         return train_data_dict , test_data_dict

#     train_data_dictionary , test_data_dictionary = key_assign(trainingX = trainX,
#                                     testingX = testX,
#                                     trainingY = trainY,
#                                     testingY = testY)

#     def key_assign(trainingX,testingX,trainingY,testingY):
#         """ 
#         Use to assign  the key to create the train_data_dict and test_data_dict   
#         Arguments:
#         trainingX -- feature for traning data 
#         testingX -- feature for testing data
#         trainingY -- label for traning data
#         testingY -- label for testing data   
#         Returns: 
#         train_data_dict -- dictionary of trainingX and trainingY
#         test_data_dict -- dictionary of testingX and testingY
#         """    
#         # Create a dictionary that can store the train set feature and label
#         train_data_dict = {"train_data_x_feature" : trainingX, "train_data_y_label" : trainingY}
        
#         # Create a dictionary that can store the test set feature and label
#         test_data_dict  = {"test_data_x_feature" : testingX , "test_data_y_label" : testingY }

#         return train_data_dict , test_data_dict

#     train_data_dictionary , test_data_dictionary = key_assign(trainingX = trainX,
#                                     testingX = testX,
#                                     trainingY = trainY,
#                                     testingY = testY)

#     def sanity_check(data_1,data_2):
#         """ 
#         Print the shape of data_1 and data_2    
#         Arguments:
#         data_1 -- (dict) type of data
#         data_2 -- (dict) type of data 
#         """
#         for key_1 in data_1:
#             print(key_1 +" shape : " + str(data_1[key_1].shape))
#         for key_2 in data_2:
#             print(key_2 +" shape : " + str(data_2[key_2].shape))

#     sanity_check(train_data_dictionary,test_data_dictionary)





    
with modelTraining:
    st.header('Model training')
    st.text('In this section you can select the hyperparameters!')

