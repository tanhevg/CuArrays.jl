export CUDNNError

struct CUDNNError <: Exception
    code::cudnnStatus_t
    msg::AbstractString
end
Base.show(io::IO, err::CUDNNError) = print(io, "CUDNNError(code $(err.code), $(err.msg))")

function CUDNNError(status::cudnnStatus_t)
    msg = unsafe_string(cudnnGetErrorString(status))
    return CUDNNError(status, msg)
end

# macro check(dnn_func)
#     quote
#         local err::cudnnStatus_t
#         err = $(esc(dnn_func))
#         if err != CUDNN_STATUS_SUCCESS
#             throw(CUDNNError(err))
#         end
#         err
#     end
# end

struct CUDNNError1 <: Exception
    code::cudnnStatus_t
    msg::AbstractString
    func_name::Symbol
    args::Tuple
end

import Base.show
function show(io::IO, err::CUDNNError1)
    println(io, "CUDNNError(code $(err.code), $(err.msg))")
    printstyled(IOContext(io, :color=>true), "$(err.func_name)(", color=:green)
    for (i, arg) in enumerate(err.args)
        printstyled(IOContext(io, :color=>true), "\n\targ$i: ", color=:green)
        argstr = sprint(show, arg; context=:limit=>true);
        argstrs = split(argstr, '\n'; keepempty=false)
        print(io, join(argstrs, "\n\t\t"))
    end
    printstyled(IOContext(io, :color=>true), "\n)", color=:green)
end

function CUDNNError1(status::cudnnStatus_t, func_name, func_args...)
    msg = unsafe_string(cudnnGetErrorString(status))
    return CUDNNError1(status, msg, func_name, func_args)
end

macro check(func_call)
    if !(func_call isa Expr && func_call.head == :call && func_call.args[1] == :ccall)
        throw("Expecting ccall")
    end
    ccall_func_name = func_call.args[2]
    ccall_func_ret_type = func_call.args[3]
    ccall_func_arg_types = func_call.args[4]
    ccall_func_args_exprs = func_call.args[5:end]

    quote
        err = ccall($(esc(ccall_func_name)), 
            $(esc(ccall_func_ret_type)), 
            $(esc(ccall_func_arg_types)), 
            $(esc.(ccall_func_args_exprs)...))
        if err != CUDNN_STATUS_SUCCESS
            throw(CUDNNError1(err, 
                $(esc(ccall_func_name))[1],
                $(esc.(ccall_func_args_exprs)...)))
        end
        err
    end
end