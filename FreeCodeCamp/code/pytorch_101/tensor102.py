import torch
import matplotlib.pyplot as plt
import torch.nn as nn

def plot_predictions(train_data=None, 
                     train_labels=None, 
                     test_data=None, 
                     test_labels=None, 
                     predictions=None):
    

        """
        Plots training data, test data and compares predictions.
        """
        plt.figure(figsize=(10, 7))

        # Plot training data in blue, c = color , s= size of array
        plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
        
        # Plot test data in green
        plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

        if predictions is not None:
            # Plot the predictions in red (predictions were made on the test data)
            plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

        # Show the legend
        plt.legend(prop={"size": 14})
        
        plt.show()

class Tensor102(object):
    def __init__(self,weight = 0 , bias = 0, start = 0 , stop = 0, step = 0):
        self.weight = weight or 0
        self.bias = bias or 0
        self.start = start or 0
        self.stop = stop or 0
        self.step = step or 0
        self.X = None
        self.Y = None
        
    def setXY(self):
        self.X = torch.arange(self.start, self.stop, self.step).unsqueeze(dim=1)
        self.Y = self.weight * self.X + self.bias
        
        # print(f'\n self.X is {self.X}')
        # print(f'\n self.Y is {self.Y}')
        
    def prepareData(self):
        # Create train/test split
        train_split = int(0.8 * len(self.X)) # 80% of data used for training set, 20% for testing 
        X_train, Y_train = self.X[:train_split], self.Y[:train_split]
        X_test, Y_test = self.X[train_split:], self.Y[train_split:]

        # len(X_train), len(y_train), len(X_test), len(y_test)
        
        # print(f'\n X_train is {X_train}')
        # print(f'\n Y_train is {Y_train}')
        # print(f'\n X_test is {X_test}')
        # print(f'\n Y_testY_test is {Y_test}')
        
        return (X_train, Y_train , X_test , Y_test)
    
    
def simplePlotPrediction():
    obj = Tensor102(weight = 0.7 , bias = 0.3 , start = 0 , stop=1 , step=0.02)	
    obj.setXY()
    a,b,c,d = obj.prepareData()
    plot_predictions(train_data=a, 
                     train_labels=b, 
                     test_data=c, 
                     test_labels=d)

# Create a Linear Regression model class
class LinearRegressionModel(nn.Module): # <- almost everything in PyTorch is a nn.Module (think of this as neural network building blocks)
    def __init__(self):
        super().__init__() 
        self.weights = nn.Parameter(torch.randn(1, # <- start with random weights (this will get adjusted as the model learns)
                                                dtype=torch.float), # <- PyTorch float32 by default
                                   requires_grad=True) # <- can we update this value with gradient descent?)

        self.bias = nn.Parameter(torch.randn(1, # <- start with random bias (this will get adjusted as the model learns)
                                            dtype=torch.float), # <- PyTorch float32 by default
                                requires_grad=True) # <- can we update this value with gradient descent?))
        
        #---------------************ideal weights**********----------------
        # self.weights = nn.Parameter(torch.tensor(0.7))
        # self.bias = nn.Parameter(torch.tensor(0.3))
        #---------------************ideal weights**********----------------
        self.loss_fn = None
        self.optimizer = None
        
    # Forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data (e.g. training/testing features)
        return self.weights * x + self.bias # <- this is the linear regression formula (Y = (weight)x + Bias)
    
def trainingLoop():
     # Set manual seed since nn.Parameter are randomly initialzied
    torch.manual_seed(42)

    # Create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))
    model_0 = LinearRegressionModel()

    # List named parameters 
    model_dict = model_0.state_dict()
    print(f'\n\n model_dict is {model_dict}')
    
    # Make predictions with model
    obj = Tensor102(weight = 0.7 , bias = 0.3 , start = 0 , stop=1 , step=0.02)	
    obj.setXY()
    X_train, Y_train , X_test , Y_test = obj.prepareData()
    
    #set the loss function and optimzer function
    model_0.loss_fn = nn.L1Loss()
    model_0.optimizer = torch.optim.SGD(params=model_0.parameters(), # parameters of target model to optimize
                                    lr=0.0001) # learning rate (how much the optimizer should change parameters at each step, higher=more (less stable), lower=less (might take a long time))
    
    epochs = 10
    y_pred = None
    # set the epochs and pass the data through the model
    for ep in range(epochs):
        
        #put the model in training mode( # default mode, updates params settings)
        model_0.train()
        
        #1 . Forward pass on train data using forward () method -> y = mx + c
        y_pred = model_0(X_train)
        
        # 2 . calculate the loss from predictions above
        loss = model_0.loss_fn(y_pred,Y_train)
        
        #3. Zero the gradients
        model_0.optimizer.zero_grad()
        
        #4. perform back propogation on the loss
        loss.backward()
        
        #5. step the optimizer( gradient descent)
        model_0.optimizer.step()
        
        # 6. Testing 
        model_0.eval()
        
    plot_predictions(train_data=X_train, 
                     train_labels=Y_train, 
                     test_data=X_test, 
                     test_labels=Y_test,
                     predictions=y_pred)
    
def randomModelPrediction():
    # Set manual seed since nn.Parameter are randomly initialzied
    torch.manual_seed(42)

    # Create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))
    model_0 = LinearRegressionModel()

    # Check the nn.Parameter(s) within the nn.Module subclass we created
    model_params = list(model_0.parameters())
    print(f'\n\n model_params is {model_params}')
    
    # List named parameters 
    model_dict = model_0.state_dict()
    print(f'\n\n model_dict is {model_dict}')
    
    # Make predictions with model
    obj = Tensor102(weight = 0.7 , bias = 0.3 , start = 0 , stop=1 , step=0.02)	
    obj.setXY()
    X_train, Y_train , X_test , Y_test = obj.prepareData()
    
    with torch.inference_mode(): 
        y_preds = model_0(X_test) # this actually calls the foreard method
    
    # print(f'\n\nX_test is {X_test}')    
    # print(f'\n\ny_preds is {y_preds}')
    
    plot_predictions(train_data=X_train, 
                     train_labels=Y_train, 
                     test_data=X_test, 
                     test_labels=Y_test,
                     predictions=y_preds)

    # Note: in older PyTorch code you might also see torch.no_grad()
    # with torch.no_grad():
    #   y_preds = model_0(X_test)



def main():
    # simplePlotPrediction()
    # randomModelPrediction()
    trainingLoop()
    pass
    
if __name__ == '__main__':
    main()

    
    
