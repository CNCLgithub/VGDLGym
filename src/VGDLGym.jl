module VGDLGym

using VGDL
using Gen
using Gen_Compose

abstract type ObservationModel end

"""
    render(::ObservationModel, ::GameState)

Produces a `Gen.ChoiceMap` that can be used for inverse-graphics.
"""
function render end

struct SimpleGraphics <: ObservationModel
    color_map::Dict #TODO
end

function render(om::SimpleGraphics, st::GameState)
    #TODO
    # something like: https://github.com/YoyoZhang24/VGDL/blob/main/src/scene.jl#L69-L90
end

struct Chervation
    cm[:kernel => state.time => :observe] = render(om, gs)
    # add action as constraint
    cm[:kernel => state.time => :agent => agent.agent_idx] =
        action_to_idx(agent, action) # TODO

    perceive!(agent, cm)
    next_action = plan!(agent)
end


function perceive!(agent::GenAgent,
                   cm::Gen.ChoiceMap)
    perceive!(agent.perception, cm)
end
