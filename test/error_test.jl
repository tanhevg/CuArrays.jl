using CuArrays
using CuArrays.CUDNN
using CuArrays.CUDNN: cudnnTensorDescriptor_t, cudnnStatus_t, libcudnn, cudnnDataType_t, CUDNN_STATUS_SUCCESS, CUDNNError, cudnnDataType, @check, cudnnGetErrorString

d = Ref{cudnnTensorDescriptor_t}()
x = cu(rand(3,4,5))
sz = Cint.(size(x)) |> reverse |> collect
st = Cint.(strides(x)) |> reverse |> collect

d[] = 0
@check ccall((:cudnnCreateTensorDescriptor,libcudnn),
             cudnnStatus_t,
             (Ptr{cudnnTensorDescriptor_t},),
             d)

@check ccall((:cudnnSetTensorNdDescriptor,libcudnn),
           cudnnStatus_t,
           (cudnnTensorDescriptor_t,cudnnDataType_t,Cint,Ptr{Cint},Ptr{Cint}),
           d[],cudnnDataType(eltype(x)),length(sz),sz,
           st)
           # rand(100, 200, 300))
