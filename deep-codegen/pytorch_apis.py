import torch as th
import gp_apis
import time

class addVec_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, dim_0, device0):
        res = gp_apis.gp_addVec(input1, input2, dim_0, device0)
        ctx.save_for_backward(input1, input2)
        return res

    @staticmethod
    def backward(ctx, dZ):
        grad_input1 = dZ
        grad_input2 = dZ
        return grad_input1, grad_input2, None, None


def addVec(input1, input2, dim_0, device0):
    return addVec_impl.apply(input1, input2, dim_0, device0)

class MatrixMuliplication_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, dim_0, dim_1, device0):
        res = gp_apis.gp_MatrixMuliplication(input1, input2, dim_0, dim_1, device0)
        # ctx.backward_cache = input1, input2
        ctx.backward_cache = input1, input2, dim_0, dim_1, device0
        return res

    @staticmethod
    def backward(ctx, dZ):
        X, W, dim_0, dim_1, device0 = ctx.backward_cache
        # dX = th.mm(dZ, W.t())
        # dW = th.mm(X.t(), dZ)
        # print('MatrixMuliplication backward')
        # print("dZ shape: ", dZ.shape)
        # print("X shape: ", X.shape)
        # print("W shape: ", W.shape)
        # print("W.T", W.T.contiguous().shape)
        dX = gp_apis.gp_MatrixMuliplication(dZ, W.T.contiguous(), dZ.shape[0], W.shape[0], device0)
        # print("dX shape: ", dX.shape)
        dW = gp_apis.gp_MatrixMuliplication(X.T.contiguous(), dZ, X.shape[1], dZ.shape[1], device0)
        # print("dW shape: ", dW.shape)
        # print(dX.shape)
        return dX, dW, None, None, None

def MatrixMuliplication(input1, input2, dim_0, dim_1, device0):
    return MatrixMuliplication_impl.apply(input1, input2, dim_0, dim_1, device0)

