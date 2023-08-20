from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt

"classification problem is a problem of predicting whether something is one thing or another and there can be multiple things as the options"

class Tensor102(object):
    def __init__(self,samples=1000,noise = 0.03 , random_seed = 42):
        self.samples = samples or 1000
        self.noise = noise or 0.4
        self.random_seed = random_seed or 42
        self.X = None
        self.y = None
        self.circles = None
        
    def createCircle(self):
        self.X,self.y = make_circles(self.samples , noise= self.noise , random_state= self.random_seed)
        
    def getXY(self, top = 10):    
        print(f"First {top} X features:\n{self.X[:top]}")
        print(f"First {top} y features:\n{self.y[:top]}")
        
    def getCircles(self , head =10):
        print(f"First {head} of circles are :\n{self.circles.head(head)}")
        
    def createDataFrame(self):
        self.circles = pd.DataFrame({"X1": self.X[:, 0],
                                    "X2": self.X[:, 1],
                                    "label": self.y
                                    })
        
        
    def plotCircle(self):
        plt.scatter(x=self.X[:, 0], 
            y=self.X[:, 1], 
            c=self.y, 
            cmap=plt.cm.RdYlBu)
        plt.show()
        
def main():
    obj = Tensor102(samples= 100 , noise=0.01 ,random_seed=42)
    obj.createCircle()
    obj.getXY()
    obj.createDataFrame()
    obj.getCircles()
    obj.plotCircle()
    
    
if __name__ == '__main__':
    main()