import torch,logging

class Tensor101(object):
    def __init__(self):
        self.test_tensor = None
        self.log = logging.getLogger(__name__)
    
    
        
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
        
        print('\n\n')
        

        # # #Vector
        # vector=torch.tensor([5,2])
        # print('----Vector Tensor----')
        # print(f'Tensor is {vector}')
        # print(f'Tensor dimension is {vector.dim()}')
        # print(f'Tensor ndim is {vector.ndim}')
        # print(f'Value in the tensor is {vector[0]}')
        # print(f'Value in the tensor is {vector.__getitem__(0)}')
        
        print('\n\n')
        
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
        
        print('\n\n')
        
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
        
        print('\n\n')
        
        #Random Tensor
        # tens_random = torch.rand(size = (2,3))
        # print('----Random Tensor----')
        # print(f'Tensor is {tens_random}')
        # print(f'Tensor dimension is {tens_random.dim()}')
        # print(f'Tensor ndim is {tens_random.ndim}')
        # print(f'Value in the tensor is {tens_random[0]}')
        # print(f'Value in the tensor is {tens_random.__getitem__(0)}')
        # print(f"Size of the tensor is {tens_random.size()}")
        # print(f"Shape of the tensor is {tens_random.shape}")
        
        print('\n\n')
        pass
    
    def getZerosAndOnes(self):
        #masking tensors with zeros and ones
        #tensor range
        #zeros like
        
        # #1 . masking with zeros
        # zero_tensor = torch.zeros(size = (2,3))
        # print(f'Zero Tensor is {zero_tensor}')
        # print(f'Data type of zero Tensor is {zero_tensor.dtype}')        
        
        print('\n')
        
        # #2 . masking with ones
        # ones_tensor = torch.ones(size = (2,3))
        # print(f'Ones Tensor is {ones_tensor}')
        # print(f'Data type of ones Tensor is {ones_tensor.dtype}')
        
        print('\n')
        
        # #.3 Tensor range and arange
        # tt= torch.range(1,10) ## --> torch.range is deprecated
        # print(f'Tensor made from range (1,10) is --> {tt}')
        
        print('\n')
        
        # tt= torch.arange(1,10) # step as 1 by default
        # print(f'Tensor made from arange (1,10) is --> {tt}')
        
        print('\n')
        
        # tt= torch.arange( start =1,end =  10,step= 3) # step as 3
        # print(f'Tensor made from arange (1,10,3) is --> {tt}')
        
        print('\n')
        
        # #4. zeros like
        # tt= torch.arange(1,10)
        # print(f'Zeros like tensor is {torch.zeros_like(tt)}')
        # print(f'Ones like tensor is {torch.ones_like(tt)}')
        
        pass
    
    def getTensorDataTypes(self):
        #float , int , 16,32,64
        #multiplying different data types and overflowing
        
        #int tensor
        tt = [1,2,3,4,5]
        
        int_32_tensor = torch.tensor(data = tt)
        print(f"int_32_tensor is -->  {int_32_tensor} and its default data type --> {int_32_tensor.dtype}")#--> by default will be int64
        
        int_32_tensor = torch.tensor(data = tt , dtype=torch.int32)
        print(f"int_32_tensor is -->  {int_32_tensor} and its data type --> {int_32_tensor.dtype}")#--> here it will be int32
        
        int_32_tensor = torch.tensor(data = tt , dtype=torch.int8)
        print(f"int_32_tensor is -->  {int_32_tensor} and its data type --> {int_32_tensor.dtype}")#--> here it will be int8
        
        int_32_tensor = torch.tensor(data = tt , dtype=torch.int16)
        print(f"int_32_tensor is -->  {int_32_tensor} and its data type --> {int_32_tensor.dtype}")#--> here it will be int16
        
        # float tensor
        print('\n\n')
        
        tt= [1.0,2.0,3.0,4.0,5.0]
        
        float_32_tensor = torch.tensor(data = tt)
        print(f"float_32_tensor is -->  {float_32_tensor} and its default data type --> {float_32_tensor.dtype}")#--> by default will be float32
        
        float_32_tensor = torch.tensor(data = tt , dtype= torch.float16)
        print(f"float_32_tensor is -->  {float_32_tensor} and its  data type --> {float_32_tensor.dtype}")#--> here it will be 16
        
        float_32_tensor = torch.tensor(data = tt , dtype= torch.float64)
        print(f"float_32_tensor is -->  {float_32_tensor} and its  data type --> {float_32_tensor.dtype}")#--> here it will be 64
        
        pass
    
    def getTensorManipulation(self):
        #add , sub,mul,divide
        #Matrix Mul
        #matrix transpose and mull
        
        print('\n\n')
        
        #1. Addition in tensors
        tt=torch.arange(1,5)
        print(f"Adding 1 to tensor {tt} is {tt+1}")
        
        print('\n\n')
        
        #2. Subtraction in tensors
        tt=torch.arange(1,5)
        print(f"Subtracting 1 from tensor {tt} is {tt-1}")
        
        print('\n\n')
        #3. Multiplication in tensors
        tt=torch.arange(1,5)
        print(f"Multiplying tensor {tt} * 2 is {tt*2}")
        
        print('\n\n')
        
        #4. Division in tensors
        tt=torch.arange(0,10,2)
        print(f"Dividing tensor {tt} by 2 is {tt/2}")
        
        print('\n\n')
        print("#######--------------<>----Matrix multiplcation-----<>-------------###########")
        
        #1. Element wise multiplication( one dim)
        mat1 = torch.tensor([1,2,3])
        mat2 = torch.tensor([1,2,3])
        
        result = mat1 * mat2
        result = torch.mul(mat1,mat2)
        result = torch.multiply(mat1,mat2)
        print(f"Multiplying Matrix 1 {mat1} with Matrix 2 {mat2} results in --> {result}") ## element wise multiplication( for 1 dimension matrix or vectors)
        
        #2. Two dimension matrix multiplication
        mat1 = torch.tensor([[1,2,3],
                             [4,5,6],
                             [7,8,9]] , dtype=torch.float32)#3*3 matrix
        
        mat2 = torch.tensor([[7, 10 , 9],
                         [8, 11 , 12]], dtype=torch.float32)#2*3 matrix
        
        #wrong dimension
        try:
            print(mat1)
            print('\n')
            print(mat2)
            result = torch.matmul(mat1,mat2)
        except Exception as e:
            # self.log.exception('Exception stack trace-----------------\n')
            print(f' --- Exception has occured --> {e}')
            
        #Transpose
        result = torch.mm(mat1,mat2.T())
        print(f'After Transposing , result is \n--> {result}')
    
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
    # obj.getZerosAndOnes()
    # obj.getTensorDataTypes()
    obj.getTensorManipulation()
        
if __name__ == '__main__':
    main()