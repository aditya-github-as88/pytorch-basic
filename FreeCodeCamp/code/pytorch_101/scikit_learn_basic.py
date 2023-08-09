# scikit learn basics

import sklearn as skl
from sklearn import datasets
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

class scikit101(object):
    
    def __init__(self):
        pass
    
    def loadData(self):
        print(f'datasets {dir(datasets)}')
        iris = datasets.load_iris()
        print(f'\n iris.feature_names \n{iris.feature_names} ')
        print(f'\n iris.data \n{iris.data} ')
        print(f'\n iris.target \n{iris.target} ')
        print(f'\n iris.target_names \n{iris.target_names} ')
        print(f'\n iris.descr \n{iris.DESCR} ')
        
    def getOpenMLData(self):
        mice = fetch_openml(name = "miceprotein", version = 4)
        print(f"\n mice details {mice} \n")
        
    def getData(self):
        path = 'D:\GIT\Tech-Stack\Technology\Deep-Learning\pytorch-basic\FreeCodeCamp\Pytorch-Notes\Misc\data-sets\Seed_Data.csv'
        total_data = pd.read_csv(path)
        print(total_data.describe)
        X = total_data.iloc[:,0:7]# fetching 0 to 7 columns
        print(f'\n X.info {X.info()}\n')
        
        Y = total_data.iloc[:,7]
        print(f'\n Y.info {Y.info()}\n')
        print(f'\n Y.info {Y.describe()}\n')
        
        return (X,Y)
        
    def modelBuilding(self):
        X,Y  = self.getData()
        X_train , X_test , Y_train , Y_test = train_test_split(X,Y, test_size= 0.2 , random_state=13)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)
        clf = svm.SVC()
        print(f'\n {clf.fit(X_train,Y_train)} \n')
        
        #prediction
        
        pred_clf = clf.predict(X_test)
        print(f'\n {accuracy_score(Y_test , pred_clf)} \n')
        print(f'\n {classification_report(Y_test , pred_clf)} \n')
        
        
        
    
def main():
    obj = scikit101()
    # obj.loadData()
    # obj.getOpenMLData()
    # obj.getData()
    obj.modelBuilding()
    
if __name__ == '__main__':
    main()
    