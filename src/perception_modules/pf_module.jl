# Define `GenCompose.InfereceChain` for seq processing

export AdaptivePF,
    IncPF,
    IncPerceptionModule,
    render_world_state

using Gen_Compose: PFChain

@with_kw struct AdaptivePF <: Gen_Compose.AbstractParticleFilter
    particles::Int = 1
    ess::Real = particles * 0.5
    # attention::AttentionModule
end

function Gen_Compose.PFChain(q::Q,
                             p::P,
                             i::Int = 1) where
    {Q<:IncrementalQuery,
     P<:AdaptivePF}

    state = Gen_Compose.initialize_procedure(p, q)
    # aux = AdaptiveComputation(p.attention)
    aux = EmptyAuxState()
    return PFChain{Q, P}(q, p, state, aux, i, typemax(i))
end

function Gen_Compose.step!(chain::PFChain{<:IncrementalQuery, <:AdaptivePF},
                           attention::AttentionModule)
    @unpack query, proc, state, step = chain
    @unpack args, argdiffs, constraints = query
    # Resample before moving on...
    Gen.maybe_resample!(state, ess_threshold=proc.ess)
    # update the state of the particles
    Gen.particle_filter_step!(state, args, argdiffs,
                              constraints)
    adaptive_compute!(chain, attention)
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
                             proc_args::NamedTuple,
                             ) where {T<:WorldModel}
    args = (0, init_state, wm) # t = 0
    # argdiffs: only `t` changes
    argdiffs = (Gen.UnknownChange(), Gen.NoChange(), Gen.NoChange())
    query = IncrementalQuery(model, Gen.choicemap(),
                             args, argdiffs, 1)
    proc = AdaptivePF(;proc_args...)
    chain = PFChain(query, proc)
    IncPerceptionModule{T}(chain)
end


function perceive!(pm::IncPerceptionModule{W},
                   am::AttentionModule,
                   wm::W,
                   st::GameState,
                   action::Int64) where {W<:WorldModel}

    gr = graphics(wm)
    obs = render(gr, st)
    ai = agent_idx(wm)
    cm = Gen.choicemap(
        (:kernel => st.time => :observe, obs),
        # add action index as constraint
        (:kernel => st.time => :agent => ai, action)
    )

    # update chain with new constraints
    chain = pm.chain
    new_args = (st.time,)
    chain.query = increment(chain.query, cm, new_args)

    # run inference procedure
    step!(chain, am)
    chain.step += 1

    # update reference in perception module
    pm.chain = chain
    return obs
end

function transfer(::MAPMemory, perception::IncPerceptionModule)
    @unpack chain  = perception
    @unpack state = chain
    map_idx = argmax(state.log_weights)
    map_trace = state.traces[map_idx]
end

function render_world_state(pm::IncPerceptionModule)
    _, _, wm = pm.chain.query.args
    gr = graphics(wm)
    traces = Gen.sample_unweighted_traces(pm.chain.state,
                                          length(pm.chain.state.traces))
    f = tr::Gen.Trace -> render(gr, last(get_retval(tr)).gstate)
    mean(map(f, traces))
end

function viz_perception_module(pm::IncPerceptionModule)
    img = render_world_state(pm)
    viz_obs(img)
    return nothing
end
