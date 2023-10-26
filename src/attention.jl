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
    kern = tr::Gen.Trace -> move_reweight(tr, select_random_agent(tr))
    pf_move_reweight!(state, kern)
    return nothing
end

function select_random_agent(tr::Gen.Trace)
    t, _... = get_args(tr)
    states = get_retval(tr)
    state = last(states)
    nagents = length(state.gstate.scene.dynamic) - 1 # not counting player
    keys = state.gstate.scene.dynamic.keys
    selected = categorical(Fill(1.0 / nagents, nagents))
    agent_idx = keys[selected + 1]
    Gen.select(:kernel => t => :dynamics => :agents => agent_idx => :action)
end
