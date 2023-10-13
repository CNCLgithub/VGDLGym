# Define `GenCompose.InfereceChain` for seq processing


using Gen_Compose: PFChain

@with_kw struct AdaptivePF <: Gen_Compose.AbstractParticleFilter
    particles::Int = 1
    ess::Real = particles * 0.5
    attention::AbstractAttentionModel
end

function Gen_Compose.PFChain{Q, P}(q::Q,
                                   p::P,
                                   n::Int,
                                   i::Int = 1) where
    {Q<:IncrementalQuery,
     P<:AdaptivePF}

    state = Gen_Compose.initialize_procedure(p, q)
    aux = AdaptiveComputation(p.attention)
    return PFChain{Q, P}(q, p, state, aux, i, n)
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

function Gen_Compose.initialize_procedure(proc::AbstractParticleFilter,
                                          query::IncrementalQuery)
    args = initial_args(query)
    constraints = initial_constraints(query)
    state = Gen.initialize_particle_filter(query.forward_function,
                                           args,
                                           constraints,
                                           proc.particles)
    return state
end

const IncPF = PFChain{<:IncrementalQuery, AdaptivePF}
mutable struct IncPerceptionModule{T} <: PerceptionModule{T}
    chain::IncPF
end


function perceive!(pm::IncPerceptionModule, cm::Gen.ChoiceMap)
    # update chain with new constraints
    chain = pm.chain
    new_args = (chain.step + 1,)
    chain.query = increment(chain.query, cm, new_args)

    # run inference procedure
    step!(chain)

    # update reference in perception module
    pm.chain = chain
    return nothing
end
