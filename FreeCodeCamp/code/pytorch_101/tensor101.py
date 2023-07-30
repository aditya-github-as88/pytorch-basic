import torch,logging
import numpy as np
class Tensor101(object):
    def __init__(self):
        self.test_tensor = None
        self.log = logging.getLogger(__name__)
    
    
        
    def getTensorTypes(self):
        # function to get all types of Tensors ( scaler , vector , Matrix , Tensor(Multi-dim))
        # random Tensors
        
        def getScaler():
            # scaler
            scaler=torch.tensor(5)
            print('\n\n')
            print('----Scaler Tensor----')
            print(f'\nTensor is {scaler}')
            print(f'\nTensor dimension is {scaler.dim()}')
            print(f'\nTensor ndim is {scaler.ndim}')
            print(f'\nValue in the tensor is {scaler.item()}')
            print('\n\n')
        
        def getVector():
            # # #Vector
            vector=torch.tensor([5,2])
            print('\n\n')
            print('----Vector Tensor----')
            print(f'\nTensor is {vector}')
            print(f'\nTensor dimension is {vector.dim()}')
            print(f'\nTensor ndim is {vector.ndim}')
            print(f'\nValue in the tensor is {vector[0]}')
            print(f'\nValue in the tensor is {vector.__getitem__(0)}')
            print('\n\n')

        def getMatrix():
            #  #Matrix
            Matrix=torch.tensor([[5,2],[2,3],[8,9]])
            print('\n\n')
            print('----Matrix Tensor----')
            print(f'\nTensor is {Matrix}')
            print(f'\nTensor dimension is {Matrix.dim()}')
            print(f'\nTensor ndim is {Matrix.ndim}')
            print(f'\nValue in the tensor is {Matrix[0]}')
            print(f'\nValue in the tensor is {Matrix.__getitem__(0)}')
            print(f"\nSize of the tensor is {Matrix.size()}")
            print(f"\nShape of the tensor is {Matrix.shape}")
            print('\n\n')
        
        def getTensor():
            # #Tensor
            Tensor=torch.tensor([[[5,2],[2,3],[8,9]],
                                [[1,2],[2,3],[3,4]],
                                [[11,13],[6,7],[8,9]]])
            print('\n\n')
            print('----Tensor Tensor----')
            print(f'\nTensor is {Tensor}')
            print(f'\nTensor dimension is {Tensor.dim()}')
            print(f'\nTensor ndim is {Tensor.ndim}')
            print(f'\nValue in the tensor is {Tensor[0]}')
            print(f'\nValue in the tensor is {Tensor.__getitem__(0)}')
            print(f"\nSize of the tensor is {Tensor.size()}")
            print(f"\nShape of the tensor is {Tensor.shape}")
            print('\n\n')
        
        def getRandomTensor():
            
            #Random Tensor
            tens_random = torch.rand(size = (2,3))
            print('\n\n')
            print('----Random Tensor----')
            print(f'\nTensor is {tens_random}')
            print(f'\nTensor dimension is {tens_random.dim()}')
            print(f'\nTensor ndim is {tens_random.ndim}')
            print(f'\nValue in the tensor is {tens_random[0]}')
            print(f'\nValue in the tensor is {tens_random.__getitem__(0)}')
            print(f"\nSize of the tensor is {tens_random.size()}")
            print(f"\nShape of the tensor is {tens_random.shape}")
            print('\n\n')
        
        getScaler()
        getVector()
        getMatrix()
        getTensor()
        getRandomTensor()
    
    def getZerosAndOnes(self):
        #masking tensors with zeros and ones
        #tensor range
        #zeros like
        
        def getZeroMasking():
            #1 . masking with zeros
            zero_tensor = torch.zeros(size = (2,3))
            print('\n\n')
            print(f'\nZero Tensor is {zero_tensor}')
            print(f'\nData type of zero Tensor is {zero_tensor.dtype}')        
            print('\n\n')
        
        def getOnesMasking():
            #2 . masking with ones
            ones_tensor = torch.ones(size = (2,3))
            print('\n\n')
            print(f'\nOnes Tensor is {ones_tensor}')
            print(f'\nData type of ones Tensor is {ones_tensor.dtype}')
            print('\n\n')
        
        def useRangeArange():
            # # #.3 Tensor range and arange
            print('\n\n')
            tt= torch.range(1,10) ## --> torch.range is deprecated
            print(f'\nTensor made from range (1,10) is --> {tt}')
            
            print('\n\n')
            tt= torch.arange(1,10) # step as 1 by default
            print(f'\nTensor made from arange (1,10) is --> {tt}')
            print('\n\n')
            
            tt= torch.arange( start =1,end =  10,step= 3) # step as 3
            print(f'\nTensor made from arange (1,10,3) is --> {tt}')
            print('\n\n')
        
        def getZerosOnesLike():
            #4. zeros like
            tt= torch.arange(1,10)
            print('\n\n')
            print(f'\nZeros like tensor is {torch.zeros_like(tt)}')
            print(f'\nOnes like tensor is {torch.ones_like(tt)}')
            print('\n\n')
    
        getZeroMasking()    
        getOnesMasking()
        useRangeArange()
        getZerosOnesLike()
    
    def getTensorDataTypes(self):
        #float , int , 16,32,64
        #multiplying different data types and overflowing
        
        
        def getIntTensor():
            tt = [1,2,3,4,5]
            print('\n\n')
            int_32_tensor = torch.tensor(data = tt)
            print(f"\nint_32_tensor is -->  {int_32_tensor} and its default data type --> {int_32_tensor.dtype}")#--> by default will be int64
            
            int_32_tensor = torch.tensor(data = tt , dtype=torch.int32)
            print(f"\nint_32_tensor is -->  {int_32_tensor} and its data type --> {int_32_tensor.dtype}")#--> here it will be int32
            
            int_32_tensor = torch.tensor(data = tt , dtype=torch.int8)
            print(f"\nint_32_tensor is -->  {int_32_tensor} and its data type --> {int_32_tensor.dtype}")#--> here it will be int8
            
            int_32_tensor = torch.tensor(data = tt , dtype=torch.int16)
            print(f"\nint_32_tensor is -->  {int_32_tensor} and its data type --> {int_32_tensor.dtype}")#--> here it will be int16
            
            print('\n\n')
        
        # float tensor
        
        def getFloatTensor():
            print('\n\n')
            tt= [1.0,2.0,3.0,4.0,5.0]
            
            float_32_tensor = torch.tensor(data = tt)
            print(f"\nfloat_32_tensor is -->  {float_32_tensor} and its default data type --> {float_32_tensor.dtype}")#--> by default will be float32
            
            float_32_tensor = torch.tensor(data = tt , dtype= torch.float16)
            print(f"\nfloat_32_tensor is -->  {float_32_tensor} and its  data type --> {float_32_tensor.dtype}")#--> here it will be 16
            
            float_32_tensor = torch.tensor(data = tt , dtype= torch.float64)
            print(f"\nfloat_32_tensor is -->  {float_32_tensor} and its  data type --> {float_32_tensor.dtype}")#--> here it will be 64
            print('\n\n')
        
        getIntTensor()
        getFloatTensor()
    
    def getTensorManipulation(self):
        #add , sub,mul,divide
        #Matrix Mul
        #matrix transpose and mull
        
        
        def getAddSubMulDiv():
            #1. Addition in tensors
            print('\n\n')
            tt=torch.arange(1,5)
            print(f"\nAdding 1 to tensor {tt} is {tt+1}")
            
            print('\n\n')
            
            #2. Subtraction in tensors
            tt=torch.arange(1,5)
            print(f"\nSubtracting 1 from tensor {tt} is {tt-1}")
            
            print('\n\n')
            #3. Multiplication in tensors
            tt=torch.arange(1,5)
            print(f"\nMultiplying tensor {tt} * 2 is {tt*2}")
            
            print('\n\n')
            
            #4. Division in tensors
            tt=torch.arange(0,10,2)
            print(f"\nDividing tensor {tt} by 2 is {tt/2}")
            print('\n\n')
        
        def getElementWiseMultiplication():
            #1. Element wise multiplication( one dim)
            print("#######--------------<>----Matrix multiplcation-----<>-------------###########")
            print('\n\n')
            mat1 = torch.tensor([1,2,3])
            mat2 = torch.tensor([1,2,3])
            
            result = mat1 * mat2
            result = torch.mul(mat1,mat2)
            result = torch.multiply(mat1,mat2)
            print(f"\nMultiplying Matrix 1 {mat1} with Matrix 2 {mat2} results in --> {result}") ## element wise multiplication( for 1 dimension matrix or vectors)
            print('\n\n')
        
        def getTwoDimMultiplication():
            #2. Two dimension matrix multiplication
            mat1 = torch.tensor([[1,2,3],
                                [4,5,6],
                                [7,8,9]] , dtype=torch.float32)#3*3 matrix
            
            mat2 = torch.tensor([[7, 10 , 9],
                            [8, 11 , 12]], dtype=torch.float32)#2*3 matrix
            
            #wrong dimension
            print('\n\n')
            print(f' --- Exception will happen when multiplying below 2 matrices')
            try:
                print(mat1)
                print('\n')
                print(mat2)
                result = torch.matmul(mat1,mat2)
            except Exception as e:
                # self.log.exception('Exception stack trace-----------------\n')
                print(f' --- Exception has occured --> {e}')
                
            #Transpose
            print('\n\n')
            print(f'\nTransposing matrix 2 \n {mat2.t()} \n')
            print(f'\nTransposing matrix 2 \n {mat2.T} \n')
            result = torch.mm(mat1,mat2.t())
            print(f'\nAfter Transposing , result is \n {result} \n')
            print('\n\n')
            
        getAddSubMulDiv()
        getElementWiseMultiplication()
        getTwoDimMultiplication()
        
        
    def getTensorAggregation(self):
        """
        min , max  ,means , sum
        positions min / max
        change tensor data type
        """        
        def getSimpleAggregation():
            print('\n\n')
            print("#######--------------<>----Mean, Max and other aggr methods on Tensors-----<>-------------###########")
            #1 Mean, Max and other aggr methods on Tensors
            x = torch.arange(0, 100, 10)
            print(f"\nMinimum: {x.min()}")
            print(f"\nMaximum: {x.max()}")
            # print(f"Mean: {x.mean()}") # this will error
            print(f"\nMean: {x.type(torch.float32).mean()}") # won't work without float datatype
            print(f"\nSum: {x.sum()}")
        
        def getTorchMinMax():
            #2 . using torch.min/max methods for aggregation
            print('\n\n')
            print("#######--------------<>----Inbuild Torch aggregation methods-----<>-------------###########")
            x = torch.arange(0, 100, 10)
            print(f'\n torch.max(x) is {torch.max(x)}')
            print(f'\n torch.min(x) is {torch.min(x)}')
            print(f'\n torch.sum(x) is {torch.sum(x)}')
            print(f'\n torch.mean(x) is {torch.mean(x,dtype=torch.float32)}')
            print(f'\n torch.mean(x.type(torch.float32)) is {torch.mean(x.type(torch.float32)), }')
        
        def getPositionalMinMax():
            #3 . Positional Min / Max 
            """
            Sometimes we need to find positions of the min/max elementy in tensor
            """
            print('\n\n')
            print("#######--------------<>----Positional Min / Max-----<>-------------###########")
            
            # Create a tensor
            tensor = torch.arange(10, 100, 10)
            print(f"\n Tensor: {tensor}")
            
            # Returns index of max and min values
            print(f"\nIndex where max value occurs: {tensor.argmax()}")
            print(f"\nIndex where min value occurs: {tensor.argmin()}")

        def changeTensorDataType():
            #4. Change Tensor data type
            """float 32 --> float 16
            torch.Tensor.type(dtype=None)
            """
            print('\n\n')
            print("#######--------------<>----Changing Tensor data type-----<>-------------###########")
            # Create a tensor and check its datatype
            tensor_float32 = torch.arange(10., 100., 10.)
            print(f'\n tensor.dtype is {tensor_float32.dtype}')
            print(f'\n32 bit Tensor is {tensor_float32}')
            

            # Create a float16 tensor
            tensor_float16 = tensor_float32.type(torch.float16)
            print(f'\n tensor.dtype is {tensor_float16.dtype}')
            print(f'\n16 bit Tensor is {tensor_float16}')
        
        getSimpleAggregation()
        getTorchMinMax()
        getPositionalMinMax()
        changeTensorDataType()
    
    
    def getTensorOperations(self):
        """Reshaping
           Stacking
           Squeezing
           UnSqueezing
        """ 
        #1.Reshaping Tensors       
        def Reshaping():
            #1 . Changing the shape/dimension
            print('\n\n')
            print("#######--------------<>----Changing Shape and dimension of Tensor-----<>-------------###########")
            x = torch.arange(1,9)
            print(f'\nTensor x is {x}')
            print(f'\nTensor x shape is {x.shape}')
            print(f'\nTensor x size is {x.size()}')
            print(f'\nTensor x dimension is {x.dim()}')
            
            #Adding a new dimension
            print('\n\n')
            print("#######--------------<>----Adding a new dimension-----<>-------------###########")
            x_1 = x.reshape(1,8)
            print(f'\nTensor x_1 is {x_1}')
            print(f'\nTensor x_1 shape is {x_1.shape}')
            print(f'\nTensor x_1 size is {x_1.size()}')
            print(f'\nTensor x_1 dimension is {x_1.dim()}')
            
            #Adding a another dimension
            print('\n\n')
            print("#######--------------<>----Adding one more dimension[ERROR]-----<>-------------###########")
            try:
                x_2 = x.reshape(2,8)
                print(f'\nTensor x_2 is {x_2}')
                print(f'\nTensor x_2 shape is {x_2.shape}')
                print(f'\nTensor x_2 size is {x_2.size()}')
                print(f'\nTensor x_2 dimension is {x_2.dim()}')
            except Exception as e:
                self.log.error('\n\n x_2 = x.reshape(2,8) \n Sorry this resulted in an Exception\n\n')
            
            #Reshaping to different dimension(2,4)
            print('\n\n')
            print("#######--------------<>----Reshaping to 2,4-----<>-------------###########")
            x_3 = x.reshape(2,4)
            print(f'\nTensor x_3 is {x_3}')
            print(f'\nTensor x_3 shape is {x_3.shape}')
            print(f'\nTensor x_3 size is {x_3.size()}')
            print(f'\nTensor x_3 dimension is {x_3.dim()}')
            
            #Reshaping to different dimension(2,2,2)
            print('\n\n')
            print("#######--------------<>----Reshaping to 2,2,2-----<>-------------###########")
            x_4 = x.reshape(2,2,2)
            print(f'\nTensor x_4 is {x_4}')
            print(f'\nTensor x_4 shape is {x_4.shape}')
            print(f'\nTensor x_4 size is {x_4.size()}')
            print(f'\nTensor x_4 dimension is {x_4.dim()}')
            
            #Reshaping to different dimension(1,2,2,2)
            print('\n\n')
            print("#######--------------<>----Reshaping to 1,2,2,2-----<>-------------###########")
            x_5 = x.reshape(1,2,2,2)
            print(f'\nTensor x_5 is {x_5}')
            print(f'\nTensor x_5 shape is {x_5.shape}')
            print(f'\nTensor x_5 size is {x_5.size()}')
            print(f'\nTensor x_5 dimension is {x_5.dim()}')
        
        #2. Changing the view of Tensors
        def ViewChange():
            x = torch.arange(1,9)
            x_1 = x.view(8)
            x_2 = torch.view_copy(x,x.dtype)
            
            print('\n\n')
            print("#######--------------<>----x_1 = x.view(8)-----<>-------------###########")
            print(f'\n x is {x}')
            print(f'\n x_1 is {x_1}')
            print(f'\n Shape of x is {x.shape}')
            print(f'\n Shape of x_1 is {x_1.shape}')
            print('\n\n')
            
            print("#######--------------<>----Changing the value in x_1 view affects x as well-----<>-------------###########")
            
            x_1[0] = 5
            print(f'\n x now after change is {x}')
            print(f'\n x_1 changing the view is {x_1}')
            
            print('\n\n')
            print("#######--------------<>----Limiting size to 5 only-----<>-------------###########")
            x_1 = x_1[:5]
            print(f'\n Shape of x is after limiting to 5 is {x.shape}')
            print(f'\n Shape of x_1 is after limiting to 5 is {x_1.shape}')
            print('\n\n')
            
            print("#######--------------<>----Change in x_2 affects x??----<>-------------###########")
            
            print(f'\n  x is {x}')
            print(f'\n  x_2 is {x_2}')
            print(f'\n  Shape of x is {x.shape}')
            print(f'\n  Shape of x_2 is {x_2.shape}')
            
            print('\n\n')
            print("#######--------------<>----Change value in x2----<>-------------###########")
            x_2[0] = 999
            
            print(f'\n x is {x}')
            print(f'\n x_2 (after change) is {x_2}')
            print(f'\n Shape of x is {x.shape}')
            print(f'\n Shape of x_2 (after change) is {x_2.shape}')
        
        
        #3.Stacking Tensors
        def Stacking():
            """Stacking Tensors V stack , H Stack
            """            
            x = torch.arange(1,8)
            x_stacked = torch.stack([x,x])
            
            print('\n\n')
            print("#######--------------<>----Stacking Tensor-----<>-------------###########")
            print(f'\n Tensor x is {x}')
            print(f'\n Tensor x_stacked is {x_stacked}')
            
            print('\n\n')
            print("#######--------------<>----Stacking Tensor with 1 dimensions added-----<>-------------###########")
            print(f'\n Original Tensor x is {x}')
            x_stacked_1 = torch.stack([x,x],dim = 1)
            print(f'\n Stacked Tensor x_stacked_1 is {x_stacked_1}')
            
            print('\n\n')
            print("#######--------------<>----Stacking Tensor with 2 dimensions added-----<>-------------###########")
            print(f'\n Original Tensor x is {x}')
            x_stacked_2 = torch.stack([x,x,x],dim = 1)
            print(f'\n Stacked Tensor x_stacked_2 is {x_stacked_2}')
            
            print('\n\n')
            print("#######--------------<>----V Stack Tensor-----<>-------------###########")
            x_vstack = torch.vstack([x,x])
            print(f'\n Original Tensor x is {x}')
            print(f'\n Tensor V stacked is {x_vstack}')
            
            print('\n\n')
            print("#######--------------<>----H Stack Tensor-----<>-------------###########")
            x_vstack = torch.hstack([x,x])
            print(f'\n Original Tensor x is {x}')
            print(f'\n Tensor H stacked is {x_vstack}')
        
        #4. Squeezing
        def Squeezing():
            """Removes all single dimension from a tensor
            """            
            #1 . 
            print('\n\n')
            print("#######--------------<>----Tensor Squeezing-----<>-------------###########")
            x = torch.arange(1,9)
            print(f'\nTensor x is {x}')
            print(f'\nTensor x shape is {x.shape}')
            print(f'\nTensor x size is {x.size()}')
            print(f'\nTensor x dimension is {x.dim()}')
            
            #Adding a new dimension
            print('\n\n')
            print("#######--------------<>----Adding a new dimension-----<>-------------###########")
            x_1 = x.reshape(1,8)
            print(f'\nTensor x_1 is {x_1}')
            print(f'\nTensor x_1 shape is {x_1.shape}')
            print(f'\nTensor x_1 size is {x_1.size()}')
            print(f'\nTensor x_1 dimension is {x_1.dim()}')
            print(f'\nTensor x_1 **squeezed** {x_1.squeeze()}')
            
            print(f'\n\n\nTensor x **unsqueezed** with 0 dimension {x.unsqueeze(dim=0)}')
            print(f'\n\n\nTensor x **unsqueezed** with 1 dimension {x.unsqueeze(dim=1)}')
            print(f'\n\n\nTensor x **unsqueezed** with -1 dimension {x.unsqueeze(dim=-1)}')
            print(f'\n\n\nTensor x **unsqueezed** with -2 dimension {x.unsqueeze(dim=-2)}')
            
            try:
                print(f'\n\n\nTensor x **unsqueezed** ie. added a new dimension {x.unsqueeze(dim=2)}')
            except Exception as e:
                print(f'_____Sorry this resulted in an exception {e}')
        
        #5 . Permute    
        def Permute():
            """Rearrange tensor dimension (only change the view)
            """            
            #1 . 
            print('\n\n')
            print("#######--------------<>----Tensor Permute( 2 dimension)-----<>-------------###########")
            x = torch.rand(2,3)
            print(f'\nTensor x is {x}')
            print(f'\nTensor x shape is {x.shape}')
            print(f'\nTensor x size is {x.size()}')
            print(f'\nTensor x dimension is {x.dim()}')
            
            x_new_permute =torch.permute(x,(1,0))
            print(f'\n x_new_permute is {x_new_permute}')
            print(f'\nTensor x_new_permute shape is {x_new_permute.shape}')
            print(f'\nTensor x_new_permute size is {x_new_permute.size()}')
            print(f'\n x_new_permute is {x_new_permute.dim()}')
            
            #changing the value in permute will affect in original tensor
            x_new_permute[0][0] = 999
            print(f'\n x_new_permute is {x_new_permute}')
            print(f'\n x is {x}')
            
            
            print('\n\n')
            print("#######--------------<>----Tensor Permute( 3 dimension)-----<>-------------###########")
            
            x = torch.rand(2,3,2)
            print(f'\nTensor x is {x}')
            print(f'\nTensor x shape is {x.shape}')
            print(f'\nTensor x size is {x.size()}')
            print(f'\nTensor x dimension is {x.dim()}')
            
            x_new_permute =torch.permute(x,(1,0,2))
            print(f'\n x_new_permute is {x_new_permute}')
            print(f'\nTensor x_new_permute shape is {x_new_permute.shape}')
            print(f'\nTensor x_new_permute size is {x_new_permute.size()}')
            print(f'\n x_new_permute is {x_new_permute.dim()}')
            
        
        def Indexing():
            x = torch.arange(1, 10).reshape(1, 3, 3)
            
            print('\n\n')
            print("#######--------------<>----Tensor Indexing-----<>-------------###########")
            
            print(f'\nTensor x is {x}')
            print(f'\nTensor x shape is {x.shape}')
            print(f'\nTensor x size is {x.size()}')
            print(f'\nTensor x dimension is {x.dim()}')
            
            # Let's index bracket by bracket
            print('\n\n')
            print("#######--------------<>----Index by bracket -- 1-----<>-------------###########")
            print(f"\nFirst square bracket:\n{x[0]}") 
            print(f"\nSecond square bracket: {x[0][0]}") 
            print(f"\nThird square bracket: {x[0][0][0]}")
            
            # Let's index bracket by bracket
            print('\n\n')
            print("#######--------------<>----Index by bracket -- 2-----<>-------------###########")
            print(f"\nFirst square bracket:\n{x[0]}") 
            print(f"\nSecond square bracket: {x[0,0]}") 
            print(f"\nThird square bracket: {x[0,0,0]}")
            
            # Let's index bracket by bracket
            
            print('\n\n')
            print("#######--------------<>----Index by bracket using (;)-----<>-------------###########")
            print(f"\nFirst square bracket:\n{x[:]}") 
            print(f"\nSecond square bracket: {x[:,0]}") 
            print(f"\nThird square bracket: {x[:,:,1]}")
            print(f"\nThird square bracket: {x[:,1,0]}")
            print(f"\nThird square bracket: {x[0,1,:]}")
            
        
        Reshaping()
        ViewChange()
        Stacking()
        Squeezing()
        Permute()
        Indexing()
        
    def pytorchfromNumpy(self):
        """
        The two main methods you'll want to use for NumPy to PyTorch (and back again) are:

        torch.from_numpy(ndarray) - NumPy array -> PyTorch tensor.
        torch.Tensor.numpy() - PyTorch tensor -> NumPy array.
        """ 
        def numpyToPytorch():
            print('\n\n')
            print("#######--------------<>----Numpy To Pytorch-----<>-------------###########")
            array = np.arange(1.0, 8.0)
            tensor = torch.from_numpy(array)
            print(f'\n Numpy Array is {array}')
            print(f'\n Pytorch Tensor is {tensor}')
            
            print('\n\n')
            print("#######--------------<>----Change numpy value-----<>-------------###########")
            array = array + 1
            print(f'\n Now Numpy Array is {array}')
            print(f'\n Now Pytorch Tensor is {tensor}')
            
            
        
        def pytorchToNumpy():
            pass
        
        numpyToPytorch()
        pytorchToNumpy()
               
def main():
    obj = Tensor101()
    # obj.getTensorTypes()
    # obj.getZerosAndOnes()
    # obj.getTensorDataTypes()
    # obj.getTensorManipulation()
    # obj.getTensorAggregation()
    # obj.getTensorOperations()
    obj.pytorchfromNumpy()
    
        
if __name__ == '__main__':
    main()