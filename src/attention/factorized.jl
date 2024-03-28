export FactorizedAttention

mutable struct FactorizedAttention <: AttentionModule
    delta_pi::Matrix{Float64}
    delta_s::Matrix{Float64}
end

function FactorizedAttention(wm::VGDLWorldModel,
                             ws::VGDLWorldState)
    st = game_state(ws)
    dpi = fill(-Inf, st.scene.bounds)
    dps = fill(-Inf, st.scene.bounds)
    FactorizedAttention(dpi, dps)
end


function clear_delta_pi!(att::FactorizedAttention)
    fill!(att.delta_pi, -Inf)
    return nothing
end


function clear_delta_s!(att::FactorizedAttention)
    fill!(att.delta_s, -Inf)
    return nothing
end

function write_delta_pi!(att::FactorizedAttention,
                         idx::Int64, val::Float64)
    idx == 0 && return nothing
    att.delta_pi[idx] = loghalf + logsumexp(att.delta_pi[idx], val)
    return nothing
end

function write_delta_s!(att::FactorizedAttention,
                         idx::Int64, val::Float64)
    idx == 0 && return nothing
    att.delta_s[idx] = loghalf + logsumexp(att.delta_s[idx], val)
    return nothing
end

function adaptive_compute!(c::InferenceChain, att::FactorizedAttention)
    @unpack proc, query, state = c
    @unpack delta_pi, delta_s = att
    # TODO: add to parameter spec
    tau = 1.0
    arousal = 10

    task_relevance = delta_pi #.+ delta_s

    # update arousal
    # logsumsens = logsumexp(task_relevance)
    # amp = m * (logsumsens + x0)
    # arousal = floor(Int64, clamp(amp, 0., max_arrousal))

    # update importance
    importance = vec(softmax(task_relevance, tau))
    clear_delta_pi!(att)
    clear_delta_s!(att)

    # REVIEW: nested tuple feels gross
    # maybe re-implement with `struct`
    kern_args = (att, agent_kernel, (importance,))

    # will loop through particles and update delta_S
    pf_move_reweight!(state, apply_computation!, kern_args,
                      arousal)
    return nothing
end


function apply_computation!(tr::Gen.Trace, att::AttentionModule,
                            kern, # TODO: type?
                            kern_args::Tuple)

    tr, idx, rel_weight = kern(tr, kern_args...)
    write_delta_s!(att, idx, rel_weight)
    (tr, rel_weight)
end
