using LinearAlgebra
using Colors
using ImageCore
using ImageInTerminal

function action_to_idx(agent::VGDL.Agent, action::Type{<:VGDL.Rule})
    findfirst(x -> x == action, actionspace(agent))
end

function VGDL.action_step(gs::GameState, actions::Dict{Int64, Int64})
    queues = OrderedDict{Int64, PriorityQueue}()
    scene = gs.scene
    # action phase
    for i = scene.dynamic.keys
        el = scene.dynamic[i]
        rule = if haskey(actions, i)
            actionspace(el)[actions[i]]
        else
            evolve(el, gs)
        end
        q = PriorityQueue{Rule, Int64}()
        sync!(q, promise(rule)(i, 0))
        queues[i] = q
    end
    return queues
end

viz_obs(obs) = display(colorview(RGB, obs))

# function softmax(x::Array{Float64}; t::Float64 = 1.0)
#     x = x .- maximum(x)
#     exs = exp.(x ./ t)
#     sxs = sum(exs)
#     n = length(x)
#     isnan(sxs) || iszero(sxs) ? fill(1.0/n, n) : exs ./ sxs
# end

function softmax(x::Array{Float64}; t::Float64 = 1.0)
    out = similar(x)
    softmax!(out, x; t = t)
    return out
end

function softmax!(out::Array{Float64}, x::Array{Float64}; t::Float64 = 1.0)
    nx = length(x)
    maxx = maximum(x)
    sxs = 0.0

    if maxx == -Inf
        out .= 1.0 / nx
        return nothing
    end

    @inbounds for i = 1:nx
        out[i] = @fastmath exp((x[i] - maxx) / t)
        sxs += out[i]
    end
    rmul!(out, 1.0 / sxs)
    return nothing
end
