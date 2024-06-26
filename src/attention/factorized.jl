export FactorizedAttention,
    viz_attention

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
    att.delta_pi[idx] += loghalf
    att.delta_pi[idx] = logsumexp(att.delta_pi[idx], val)
    return nothing
end

function write_delta_s!(att::FactorizedAttention,
                         idx::Int64, val::Float64)
    idx == 0 && return nothing
    att.delta_s[idx] += loghalf
    att.delta_s[idx] = logsumexp(att.delta_s[idx], val)
    return nothing
end

function adaptive_compute!(c::InferenceChain, att::FactorizedAttention)
    @unpack proc, query, state = c
    @unpack delta_pi, delta_s = att
    # TODO: add to parameter spec
    tau = 0.001
    arousal = 20

    # task_relevance = logsumexp.(delta_pi, delta_s)
    task_relevance = delta_pi .+ delta_s
    # task_relevance = delta_s

    # update arousal
    # logsumsens = logsumexp(task_relevance)
    # amp = m * (logsumsens + x0)
    # arousal = floor(Int64, clamp(amp, 0., max_arrousal))

    # update importance
    importance = vec(softmax(task_relevance, tau))
    # clear_delta_pi!(att)
    # clear_delta_s!(att)

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
    # write_delta_s!(att, idx, rel_weight)
    (tr, rel_weight)
end

function viz_attention(att::FactorizedAttention,
                       path::String="")
    @unpack delta_pi, delta_s = att
    ws = softmax(delta_pi .+ delta_s, 5.0)
    nx, ny = size(delta_s)
    cis = CartesianIndices((nx, ny))
    img = zeros((3, nx, ny))
    for i = cis
        x,y = i.I # REVIEW: assumes `i` is `CartesianIndex{2}`
        w = ws[i]
        img[1, x, y] = min(w * 5 , 1.0)
    end

    if path != ""
        save(path, repeat(render_obs(img), inner = (10,10)))
    end
    viz_obs(img)
    return nothing
end
