export AdaptiveComputation,
    NoAttention,
    no_attention,
    UniformAttention,
    uniform_attention

@with_kw mutable struct AdaptiveComputation <: AuxillaryState
    acceptance::Float64 = 0.
    arrousal::Float64 = 0
    importance::Vector{Float64}
    sensitivities::Vector{Float64}
end

struct NoAttention <: AttentionModule end
const no_attention = NoAttention()

function write_delta_pi!(::NoAttention, ::Int64, ::Any)
    return nothing
end

function adaptive_compute!(::InferenceChain, ::NoAttention)
    return nothing
end

function AdaptiveComputation(::NoAttention)
    AdaptiveComputation(;importance = Float64[],
                        sensitivities = Float64[])
end


struct UniformAttention <: AttentionModule end
const uniform_attention = UniformAttention()

function AdaptiveComputation(::UniformAttention)
    AdaptiveComputation(;importance = Float64[],
                        sensitivities = Float64[])
end

function write_delta_pi!(::UniformAttention, ::Int64, ::Float64)
    return nothing
end

function adaptive_compute!(c::InferenceChain, ::UniformAttention)
    @unpack proc, query, state = c
    kern = tr::Gen.Trace -> perception_mcmc_kernel(tr, 3, 5)
    pf_move_reweight!(state, kern)
    return nothing
end

include("factorized.jl")
