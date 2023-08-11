import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

class ModelRegression(object):
    pass

    def plot_Linear(self):
        path = "D:\GIT\Tech-Stack\Technology\Deep-Learning\Material\Links_notes\Algorithms\excel_docs\data_sets\home_prices.csv"
        df = pd.read_csv(path)
        print(df)
        
        print("\n\n")
        
        plt.scatter(df.area,df.price)
        plt.show()
    
    
    
def main():
    obj = ModelRegression()
    obj.plot_Linear()
    
if __name__ == '__main__':
    main()