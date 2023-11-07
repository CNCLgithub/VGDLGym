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
    choices = get_choices(tr)
    sub_addr = :kernel => t => :dynamics => :agents
    #NOTE: retval may return newly spawned agents
    nagents = length(tr[sub_addr]) - 1 # not counting player
    selected = categorical(Fill(1.0 / nagents, nagents))
    ai = selected + 1 # from 1 -> N (because of Gen.Map)
    # display(choicemap(get_submap(get_choices(tr), :kernel => t => :dynamics => :agents)))
    addr = :kernel => t => :dynamics => :agents =>
        ai => :action
    Gen.select(addr)
end
