# Deprecated functionality

import Base: @deprecate_binding

if isdefined(CuArrays, CUBLAS)
    @deprecate_binding BLAS CUBLAS
end

if isdefined(CuArrays, CUFFT)
    @deprecate_binding FFT CUFFT
end
