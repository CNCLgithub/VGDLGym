# Define `GenCompose.InfereceChain` for seq processing

export AdaptivePF,
    IncPF,
    IncPerceptionModule

using Gen_Compose: PFChain

@with_kw struct AdaptivePF <: Gen_Compose.AbstractParticleFilter
    particles::Int = 1
    ess::Real = particles * 0.5
    attention::AttentionModule
end

function Gen_Compose.PFChain(q::Q,
                             p::P,
                             i::Int = 1) where
    {Q<:IncrementalQuery,
     P<:AdaptivePF}

    state = Gen_Compose.initialize_procedure(p, q)
    aux = AdaptiveComputation(p.attention)
    return PFChain{Q, P}(q, p, state, aux, i, typemax(i))
end

function Gen_Compose.step!(chain::PFChain{<:IncrementalQuery, <:AdaptivePF})
    @unpack query, proc, state, step = chain
    @unpack args, argdiffs, constraints = query
    # Resample before moving on...
    Gen.maybe_resample!(state, ess_threshold=proc.ess)
    # update the state of the particles
    Gen.particle_filter_step!(state, args, argdiffs,
                              constraints)
    adaptive_compute!(chain, proc.attention)
    return nothing
end

function Gen_Compose.initialize_procedure(proc::T,
                                          query::IncrementalQuery
                                          ) where {
                                              T<:Gen_Compose.AbstractParticleFilter
                                          }
    @unpack model, args, constraints = query
    state = Gen.initialize_particle_filter(model,
                                           args,
                                           constraints,
                                           proc.particles)
    return state
end

const IncPF = PFChain{<:IncrementalQuery, AdaptivePF}

mutable struct IncPerceptionModule{T} <: PerceptionModule{T}
    chain::IncPF
end

function IncPerceptionModule(model::Gen.GenerativeFunction,
                             wm::T, init_state::WorldState{T},
                             proc_args::Tuple,
                             ) where {T<:WorldModel}
    args = (0, init_state, wm) # t = 0
    # argdiffs: only `t` changes
    argdiffs = (Gen.UnknownChange(), Gen.NoChange(), Gen.NoChange())
    query = IncrementalQuery(model, Gen.choicemap(),
                             args, argdiffs, 1)
    proc = AdaptivePF(proc_args...)
    chain = PFChain(query, proc)
    IncPerceptionModule{T}(chain)
end


function perceive!(pm::IncPerceptionModule, cm::Gen.ChoiceMap,
                   time::Int)
    # update chain with new constraints
    chain = pm.chain
    new_args = (time,)
    chain.query = increment(chain.query, cm, new_args)

    # run inference procedure
    step!(chain)
    chain.step += 1

    # update reference in perception module
    pm.chain = chain
    return nothing
end

function transfer(::MAPTransfer, perception::IncPerceptionModule)
    @unpack chain  = perception
    @unpack state = chain
    map_idx = argmax(state.log_weights)
    map_trace = state.traces[map_idx]
end

function viz_world_state(pm::IncPerceptionModule)
    _, _, wm = pm.chain.query.args
    gr = graphics(wm)
    traces = Gen.sample_unweighted_traces(pm.chain.state,
                                          length(pm.chain.state.traces))
    f = tr::Gen.Trace -> render(gr, last(get_retval(tr)).gstate)
    img = mean(map(f, traces))
    viz_obs(img)
    return nothing
end
