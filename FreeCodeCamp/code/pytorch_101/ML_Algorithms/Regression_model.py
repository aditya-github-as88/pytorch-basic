import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import math

class ModelRegression(object):
    def plot_Linear(self):
        #fetch the data
        path = "D:\GIT\Tech-Stack\Technology\Deep-Learning\pytorch-basic\Material\Links_notes\Algorithms\excel_docs\data_sets\home_prices.csv"
        df = pd.read_csv(path)
        print(df)
        
        print("\n\n")
        
        #plot the data
        plt.xlabel('area')
        plt.ylabel('home_prices $')
        plt.scatter(df.area,df.price, color = 'red' , marker = '+')
        # plt.show()
        
        reg = linear_model.LinearRegression()
        # reg.fit(df[['area']] , df.price)# traiing the LR model using data points
        print(f'\n area is {df["area"]} \n')
        print(f'\ntype of df area is {type(df[["area"]])}')
        reg.fit(df[["area"]] , df["price"])
        
        
        
        # prediction - general static
        x_pred = [[5000]]
        y_pred =  reg.predict(x_pred)
        print('\n predicted price for {0} is {1} \n'.format(x_pred,y_pred))
        print('\n reg coeff {0}'.format(reg.coef_))
        print('\n reg intercept {0}'.format(reg.intercept_))
        
        
        # prediction from range of area from csv
        path = "D:\GIT\Tech-Stack\Technology\Deep-Learning\pytorch-basic\Material\Links_notes\Algorithms\excel_docs\data_sets\Input-Area.csv"
        df1 = pd.read_csv(path)
        print(df1)
        # input_area = df1
        price_prediction =  reg.predict(df1)
        print(f'\n ___ prediction analysis_____\n')
        print(f'input area \n {df1}')
        print(f'price predicted \n {price_prediction}')
        
        # adding predicted prices to data frame
        df1['prices-predicted'] = price_prediction
        print(f'\n Data Frame for predicted prices \n {df1}\n')
        
        #ploltting the prediction
        plt.scatter(df1.area,df1["prices-predicted"], color = 'blue' , marker = '*')
        plt.show()
        
        
        # saving data frame to csv file
        output_path = "D:\GIT\Tech-Stack\Technology\Deep-Learning\pytorch-basic\Material\Links_notes\Algorithms\excel_docs\data_sets\Predictions\Prediction-Price.csv"
        df1.to_csv(output_path , index=False)
    
    def plot_multi_Variable(self):
        # price = m1 * area + m2 * bedrooms + m3 * age + b
        
         #fetch the data
        path = "D:\GIT\Tech-Stack\Technology\Deep-Learning\pytorch-basic\Material\Links_notes\Algorithms\excel_docs\data_sets\multivar_data.csv"
        df = pd.read_csv(path)
        
        print(f'\nMultivariable data set\n {df}\n')
        
        # handling null values via median
        # calculate median
        median_bedroom = math.floor(df.bedroom.median())
        print(f'\nmedian_bedroom\n {median_bedroom}\n')
        
        #filling NA with median
        df.bedroom = df.bedroom.fillna(median_bedroom)
        # print(f'\n new data is \n {df} \n')
        print(f'\n processed data after removing null values \n{df}\n')
        
        reg = linear_model.LinearRegression()
        reg.fit(df[['area','bedroom','age']] , df['price'])
        
        # finding the coeeficients
        print(f'\n  Coefficients for the model \n{reg.coef_}\n')
        
        #intercept
        print(f'\n  intercept for the model \n{reg.intercept_}\n')
        
        #prediction
        print(f'\n Prediction for the learned model is {reg.predict([[3000,3,40]])} \n')
        
        
    
def main():
    obj = ModelRegression()
    # obj.plot_Linear()
    obj.plot_multi_Variable()
    
if __name__ == '__main__':
    main()