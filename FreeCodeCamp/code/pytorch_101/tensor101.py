import torch
class Tensor101(object):
    def __init__(self):
        self.test_tensor = None
    
    
        
    def getTensorTypes(self):
        # function to get all types of Tensors ( scaler , vector , Matrix , Tensor(Multi-dim))
        # random Tensors
        
        # scaler
        # scaler=torch.tensor(5)
        # print('----Scaler Tensor----')
        # print(f'Tensor is {scaler}')
        # print(f'Tensor dimension is {scaler.dim()}')
        # print(f'Tensor ndim is {scaler.ndim}')
        # print(f'Value in the tensor is {scaler.item()}')
        

        # # #Vector
        # vector=torch.tensor([5,2])
        # print('----Vector Tensor----')
        # print(f'Tensor is {vector}')
        # print(f'Tensor dimension is {vector.dim()}')
        # print(f'Tensor ndim is {vector.ndim}')
        # print(f'Value in the tensor is {vector[0]}')
        # print(f'Value in the tensor is {vector.__getitem__(0)}')
        
        #  #Matrix
        # Matrix=torch.tensor([[5,2],[2,3],[8,9]])
        # print('----Matrix Tensor----')
        # print(f'Tensor is {Matrix}')
        # print(f'Tensor dimension is {Matrix.dim()}')
        # print(f'Tensor ndim is {Matrix.ndim}')
        # print(f'Value in the tensor is {Matrix[0]}')
        # print(f'Value in the tensor is {Matrix.__getitem__(0)}')
        # print(f"Size of the tensor is {Matrix.size()}")
        # print(f"Shape of the tensor is {Matrix.shape}")
        
        # #Tensor
        # Tensor=torch.tensor([[[5,2],[2,3],[8,9]],
        #                     [[1,2],[2,3],[3,4]],
        #                     [[11,13],[6,7],[8,9]]])
        # print('----Tensor Tensor----')
        # print(f'Tensor is {Tensor}')
        # print(f'Tensor dimension is {Tensor.dim()}')
        # print(f'Tensor ndim is {Tensor.ndim}')
        # print(f'Value in the tensor is {Tensor[0]}')
        # print(f'Value in the tensor is {Tensor.__getitem__(0)}')
        # print(f"Size of the tensor is {Tensor.size()}")
        # print(f"Shape of the tensor is {Tensor.shape}")
        
        #Random Tensor
        tens_random = torch.rand(size = (2,3))
        print('----Random Tensor----')
        print(f'Tensor is {tens_random}')
        print(f'Tensor dimension is {tens_random.dim()}')
        print(f'Tensor ndim is {tens_random.ndim}')
        print(f'Value in the tensor is {tens_random[0]}')
        print(f'Value in the tensor is {tens_random.__getitem__(0)}')
        print(f"Size of the tensor is {tens_random.size()}")
        print(f"Shape of the tensor is {tens_random.shape}")
        
        
    
    def getZerosAndOnes(self):
        #masking tensors with zeros and ones
        #tensor range
        #zeros like
        
        #1 . masking with zeros
        zero_tensor = torch.zeros(size = (2,3))
        print(f'Zero Tensor is {zero_tensor}')
        print(f'Data type of zero Tensor is {zero_tensor.dtype}')        
        
        #2 . masking with ones
        ones_tensor = torch.ones(size = (2,3))
        print(f'Ones Tensor is {ones_tensor}')
        print(f'Data type of ones Tensor is {ones_tensor.dtype}')
        
        #.3 Tensor range and arange
        tt= torch.range(1,10) ## --> torch.range is deprecated
        print(f'Tensor made from range (1,10) is --> {tt}')
        
        tt= torch.arange(1,10) # step as 1 by default
        print(f'Tensor made from arange (1,10) is --> {tt}')
        
        tt= torch.arange( start =1,end =  10,step= 3) # step as 3
        print(f'Tensor made from arange (1,10,3) is --> {tt}')
        
        #4. zeros like
        tt= torch.arange(1,10)
        print(f'Zeros like tensor is {torch.zeros_like(tt)}')
    
    def getTensorDataTypes(self):
        #float , int , 16,32,64
        #multiplying different data types and overflowing
        pass
    
    def getTensorManipulation(self):
        #add , sub,mul,divide
        #Matrix Mul
        #matrix transpose and mull
        pass
    
    def getTensorAggregation(self):
        #min , max  ,means , sum
        #positions min / max
        #change tensor data type
        pass
    
    def getTensoroperations(self):
        # re shaping
        #stacking
        #squeezing and unsqueezin
        pass
        

def main():
    obj = Tensor101()
    # obj.getTensorTypes()
    obj.getZerosAndOnes()
        
if __name__ == '__main__':
    main()