export AdaptiveComputation,
    NoAttention,
    no_attention

@with_kw mutable struct AdaptiveComputation <: AuxillaryState
    acceptance::Float64 = 0.
    arrousal::Float64 = 0
    importance::Vector{Float64}
    sensitivities::Vector{Float64}
end

struct NoAttention <: AttentionModule end
const no_attention = NoAttention()

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

function adaptive_compute!(c::InferenceChain, ::UniformAttention)
    @unpack proc, query, state = c
    (t, _, _) = query.args
    for i = 1:proc.particles
        state.traces[i] = regenerate_random_latent(state.traces[i],
                                                   t)
    end

    return nothing
end
