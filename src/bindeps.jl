# discovering binary CUDA dependencies

using Pkg, Pkg.Artifacts
using Libdl

const libcublas = Ref{String}("cublas")
const libcusparse = Ref{String}("cusparse")
const libcusolver = Ref{String}("cusolver")
const libcufft = Ref{String}("cufft")
const libcurand = Ref{String}("curand")
const libcudnn = Ref{String}("cudnn")
const libcutensor = Ref{String}("cutensor")


## discovery

# NOTE: we don't use autogenerated JLLs, because we have multiple artifacts and need to
#       decide at run time (i.e. not via package dependencies) which one to use.
const cuda_artifacts = Dict(
    v"10.2" => ()->artifact"CUDA10.2",
    v"10.1" => ()->artifact"CUDA10.1",
    v"10.0" => ()->artifact"CUDA10.0",
    v"9.2"  => ()->artifact"CUDA9.2",
    v"9.0"  => ()->artifact"CUDA9.0",
)

# utilities to look up stuff in the artifact (at known locations, so not using CUDAapi)
function get_library(artifact, name)
    filename = if Sys.iswindows()
        "$name.dll"
    elseif Sys.isapple()
        "lib$name.dylib"
    else
        "lib$name.so"
    end
    joinpath(artifact, Sys.iswindows() ? "bin" : "lib", filename)
end

# try use CUDA from an artifact
function use_artifact_cuda()
    # select compatible artifacts
    if haskey(ENV, "JULIA_CUDA_VERSION")
        wanted_version = VersionNumber(ENV["JULIA_CUDA_VERSION"])
        filter!(((version,artifact),) -> version == wanted_version, cuda_artifacts)
    else
        driver_version = CUDAdrv.release()
        filter!(((version,artifact),) -> version <= driver_version, cuda_artifacts)
    end

    # download and install
    artifact = nothing
    release = nothing
    for version in sort(collect(keys(cuda_artifacts)); rev=true)
        try
            artifact = cuda_artifacts[version]()
            release = version
            break
        catch
        end
    end
    artifact == nothing && error("Could not find a compatible artifact.")

    # discover libraries
    for name in  ("cublas", "cusparse", "cusolver", "cufft", "curand")
        handle = getfield(CuArrays, Symbol("lib$name"))

        # on Windows, the library name is version dependent
        if Sys.iswindows()
            name *= release >= v"10.1" ? "64_$(release.major)" : "64_$(release.major)$(release.minor)"
        end

        handle[] = get_library(artifact, name)
        Libdl.dlopen(handle[])
    end

    @debug "Using CUDA $(release) from an artifact at $(artifact)"
    return release, [artifact]
end

# try to use CUDA from a local installation
function use_local_cuda()
    dirs = find_toolkit()

    tool = find_cuda_binary("nvdisasm")
    tool == nothing && error("Your CUDA installation does not provide the nvdisasm binary")
    version = parse_toolkit_version(tool)

    # discover libraries
    for name in  ("cublas", "cusparse", "cusolver", "cufft", "curand")
        handle = getfield(CuArrays, Symbol("lib$name"))

        path = find_cuda_library(name, dirs, [version])
        if path !== nothing
            handle[] = path
        end
    end

    @debug "Using local CUDA $(version) at $(join(dirs, ", "))"
    return version, dirs
end

const cudnn_artifacts = Dict(
    v"10.2" => ()->artifact"CUDNN+CUDA10.2",
    v"10.1" => ()->artifact"CUDNN+CUDA10.1",
    v"10.0" => ()->artifact"CUDNN+CUDA10.0",
    v"9.2"  => ()->artifact"CUDNN+CUDA9.2",
    v"9.0"  => ()->artifact"CUDNN+CUDA9.0",
)

function use_artifact_cudnn(cuda_release)
    artifact = try
        cudnn_artifacts[cuda_release]()
    catch ex
        @debug "Could not use CUDNN from artifacts" exception=(ex, catch_backtrace())
        return
    end

    libcudnn[] = get_library(artifact, Sys.iswindows() ? "cudnn64_7" : "cudnn")
    Libdl.dlopen(libcudnn[])
    @debug "Using CUDNN from an artifact at $(artifact)"
end

function use_local_cudnn(cuda_dirs)
    path = find_cuda_library("cudnn", cuda_dirs, [v"7"])
    if path !== nothing
        libcudnn[] = path
        @debug "Using local CUDNN at $(path)"
    end
end

const cutensor_artifacts = Dict(
    v"10.2" => ()->artifact"CUTENSOR+CUDA10.2",
    v"10.1" => ()->artifact"CUTENSOR+CUDA10.1",
)

function use_artifact_cutensor(cuda_release)
    artifact = try
        cutensor_artifacts[cuda_release]()
    catch ex
        @debug "Could not use CUTENSOR from artifacts" exception=(ex, catch_backtrace())
        return
    end

    libcutensor[] = get_library(artifact, "cutensor")
    Libdl.dlopen(libcutensor[])
    @debug "Using CUTENSOR from an artifact at $(artifact)"
end

function use_local_cutensor(cuda_dirs)
    path = find_cuda_library("cutensor", cuda_dirs, [v"1"])
    if path !== nothing
        libcutensor[] = path
        @debug "Using local CUTENSOR at $(path)"
    end
end

function __init_bindeps__(; silent=false, verbose=false)
    cuda = try
        parse(Bool, get(ENV, "JULIA_CUDA_USE_BINARYBUILDER", "true")) ||
            error("Use of CUDA artifacts not allowed by user")
        cuda_release, artifact = use_artifact_cuda()
        use_artifact_cudnn(cuda_release)
        use_artifact_cutensor(cuda_release)
        cuda_release
    catch ex
        @debug "Could not use CUDA from artifacts" exception=(ex, catch_backtrace())
        cuda_version, cuda_dirs = use_local_cuda()
        use_local_cudnn(cuda_dirs)
        use_local_cutensor(cuda_dirs)
        VersionNumber(cuda_version.major, cuda_version.minor)
    end

    # library dependencies
    CUBLAS.version()
    CUSPARSE.version()
    CUSOLVER.version()
    CUFFT.version()
    CURAND.version()
    # CUDNN and CUTENSOR are optional

    # library compatibility
    if has_cutensor()
        cutensor = CUTENSOR.version()
        if cutensor < v"1"
            silent || @warn("CuArrays.jl only supports CUTENSOR 1.0 or higher")
        end

        cutensor_cuda = CUTENSOR.cuda_version()
        if cutensor_cuda.major != cuda.major || cutensor_cuda.minor != cuda.minor
            silent || @warn("You are using CUTENSOR $cutensor for CUDA $cutensor_cuda with CUDA toolkit $cuda; these might be incompatible.")
        end
    end
    if has_cudnn()
        cudnn = CUDNN.version()
        if cudnn < v"7.6"
            silent || @warn("CuArrays.jl only supports CUDNN v7.6 or higher")
        end

        cudnn_cuda = CUDNN.cuda_version()
        if cudnn_cuda.major != cuda.major || cudnn_cuda.minor != cuda.minor
            silent || @warn("You are using CUDNN $cudnn for CUDA $cudnn_cuda with CUDA toolkit $cuda; these might be incompatible.")
        end
    end
end
