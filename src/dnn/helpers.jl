# For low level cudnn functions that require a pointer to a number
cptr(x,a::CuArray{Float64})=Float64[x]
cptr(x,a::CuArray{Float32})=Float32[x]
cptr(x,a::CuArray{Float16})=Float32[x]

# Conversion between Julia and CUDNN datatypes
cudnnDataType(::Type{Float16})=CUDNN_DATA_HALF
cudnnDataType(::Type{Float32})=CUDNN_DATA_FLOAT
cudnnDataType(::Type{Float64})=CUDNN_DATA_DOUBLE
juliaDataType(a)=(a==CUDNN_DATA_HALF ? Float16 :
                  a==CUDNN_DATA_FLOAT ? Float32 :
                  a==CUDNN_DATA_DOUBLE ? Float64 : error())

tuple_strides(A::Tuple) = _strides((1,), A)
_strides(out::Tuple{Int}, A::Tuple{}) = ()
_strides(out::NTuple{N,Int}, A::NTuple{N}) where {N} = out
function _strides(out::NTuple{M,Int}, A::Tuple) where M
    Base.@_inline_meta
    _strides((out..., out[M]*A[M]), A)
end

# Descriptors

mutable struct TensorDesc; ptr; end
free(td::TensorDesc) = cudnnDestroyTensorDescriptor(td.ptr)
Base.unsafe_convert(::Type{cudnnTensorDescriptor_t}, td::TensorDesc) = td.ptr
Base.unsafe_convert(::Type{Ptr{Nothing}}, td::TensorDesc) = convert(Ptr{Nothing}, td.ptr)

function TensorDesc(T::Type, size::NTuple{N,Integer}, strides::NTuple{N,Integer} = tuple_strides(size)) where N
    sz = Cint.(size) |> reverse |> collect
    st = Cint.(strides) |> reverse |> collect
    d = Ref{cudnnTensorDescriptor_t}()
    cudnnCreateTensorDescriptor(d)
    cudnnSetTensorNdDescriptor(d[], cudnnDataType(T), length(sz), sz, st)
    this = TensorDesc(d[])
    finalizer(free, this)
    return this
end

TensorDesc(a::CuArray) = TensorDesc(eltype(a), size(a), strides(a))

mutable struct FilterDesc
  ptr
end
free(fd::FilterDesc)=cudnnDestroyFilterDescriptor(fd.ptr)
Base.unsafe_convert(::Type{cudnnFilterDescriptor_t}, fd::FilterDesc)=fd.ptr
Base.unsafe_convert(::Type{Ptr{Nothing}}, fd::FilterDesc)=fd.ptr

function createFilterDesc()
  d = Ref{cudnnFilterDescriptor_t}()
  cudnnCreateFilterDescriptor(d)
  return d[]
end

function FilterDesc(T::Type, size::Tuple; format = CUDNN_TENSOR_NCHW)
    # The only difference of a FilterDescriptor is no strides.
    sz = Cint.(size) |> reverse |> collect
    d = createFilterDesc()
    version() >= v"5" ?
        cudnnSetFilterNdDescriptor(d, cudnnDataType(T), format, length(sz), sz) :
    version() >= v"4" ?
        cudnnSetFilterNdDescriptor_v4(d, cudnnDataType(T), format, length(sz), sz) :
        cudnnSetFilterNdDescriptor(d, cudnnDataType(T), length(sz), sz)
    this = FilterDesc(d)
    finalizer(free, this)
    return this
end

FilterDesc(a::CuArray; format = CUDNN_TENSOR_NCHW) = FilterDesc(eltype(a), size(a), format = format)

function Base.size(f::FilterDesc)
  typ = Ref{Cuint}()
  format = Ref{Cuint}()
  ndims = Ref{Cint}()
  dims = Vector{Cint}(undef, 8)
  cudnnGetFilterNdDescriptor(f, 8, typ, format, ndims, dims)
  @assert ndims[] ≤ 8
  return (dims[1:ndims[]]...,) |> reverse
end

mutable struct ConvDesc; ptr; end
free(cd::ConvDesc) = cudnnDestroyConvolutionDescriptor(cd.ptr)
Base.unsafe_convert(::Type{cudnnConvolutionDescriptor_t}, cd::ConvDesc)=cd.ptr

function cdsize(w, nd)
    isa(w, Integer) && return Cint[fill(w,nd)...]
    length(w) == nd && return Cint[reverse(w)...]
    length(w) == 2*nd && return Cint[reverse(w[nd+1:end])...]
    throw(DimensionMismatch())
end

pdsize(w, nd)=Cint[reverse(psize(w,nd))...]
function psize(w, nd)
    isa(w, Integer) && return Cint[fill(w,nd)...]
    length(w) == nd && return w
    length(w) == 2*nd && return w[1:nd]
    throw(DimensionMismatch())
end

function ConvDesc(T, N, padding, stride, dilation, mode)
    cd = Ref{cudnnConvolutionDescriptor_t}()
    cudnnCreateConvolutionDescriptor(cd)
    version() >= v"4" ? cudnnSetConvolutionNdDescriptor(cd[],N,cdsize(padding,N),cdsize(stride,N),cdsize(dilation,N),mode,cudnnDataType(T)) :
    version() >= v"3" ? cudnnSetConvolutionNdDescriptor_v3(cd[],N,cdsize(padding,N),cdsize(stride,N),cdsize(dilation,N),mode,cudnnDataType(T)) :
    cudnnSetConvolutionNdDescriptor(cd[],N,cdsize(padding,N),cdsize(stride,N),cdsize(dilation,N),mode)
    this = ConvDesc(cd[])
    finalizer(free, this)
    return this
end

function ConvDesc(T, cdims::DenseConvDims)
    pd = NNlib.padding(cdims)
    if !all(pd[1:2:end] .== pd[2:2:end])
        @warn("CuDNN does not support asymmetric padding; defaulting to symmetric choice")
    end
    return ConvDesc(T, NNlib.spatial_dims(cdims), pd[1:2:end], NNlib.stride(cdims),
                       NNlib.dilation(cdims), NNlib.flipkernel(cdims))
end

mutable struct PoolDesc; ptr; end
free(pd::PoolDesc)=cudnnDestroyPoolingDescriptor(pd.ptr)
Base.unsafe_convert(::Type{cudnnPoolingDescriptor_t}, pd::PoolDesc)=pd.ptr

function PoolDesc(nd, window, padding, stride, mode, maxpoolingNanOpt=CUDNN_NOT_PROPAGATE_NAN)
    pd = Ref{cudnnPoolingDescriptor_t}()
    cudnnCreatePoolingDescriptor(pd)
    cudnnSetPoolingNdDescriptor(pd[],mode,maxpoolingNanOpt,nd,pdsize(window,nd),pdsize(padding,nd),pdsize(stride,nd))
    this = PoolDesc(pd[])
    finalizer(free, this)
    return this
end

function PoolDesc(pdims::PoolDims, mode, maxpoolingNanOpt=CUDNN_NOT_PROPAGATE_NAN)
    pd = NNlib.padding(pdims)
    if !all(pd[1:2:end] .== pd[2:2:end])
        @warn("CuDNN does not support asymmetric padding; defaulting to symmetric choice")
    end
    return PoolDesc(NNlib.spatial_dims(pdims), NNlib.kernel_size(pdims), pd[1:2:end],
                    NNlib.stride(pdims), mode, maxpoolingNanOpt)
end

mutable struct ActivationDesc; ptr; end
free(ad::ActivationDesc)=cudnnDestroyActivationDescriptor(ad.ptr)
Base.unsafe_convert(::Type{cudnnActivationDescriptor_t}, ad::ActivationDesc)=ad.ptr

function ActivationDesc(mode, coeff, reluNanOpt=CUDNN_NOT_PROPAGATE_NAN)
    ad = Ref{cudnnActivationDescriptor_t}()
    cudnnCreateActivationDescriptor(ad)
    cudnnSetActivationDescriptor(ad[],mode,reluNanOpt,coeff)
    this = ActivationDesc(ad[])
    finalizer(free, this)
    return this
end
