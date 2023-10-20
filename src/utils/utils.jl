using Colors
using ImageCore
using ImageInTerminal

function action_to_idx(agent::VGDL.Agent, action::VGDL.Rule)
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
