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

struct ChoiceMapObservation <: VGDL.Observation
    cm::Gen.ChoiceMap
end

function VGDL.observe(agent::GenAgent, i::Int64, gs::GameState, kdtree::KDTree)
    om = observation_model(agent)
    obs = render(om, gs)
    cm = choicemap()
    cm[:kernel => gs.time => :observe] = render(om, gs)
    ChoiceMapObservation(cm)
end


abstract type WorldModel end
abstract type WorldState{T<:WorldModel} end

include("world_models/world_models.jl")

abstract type PerceptionModule{T<:WorldModel} end
abstract type PlanningModule{T<:WorldModel} end


# Define agents that infer the world
struct GenAgent{W<:WorldModel} <: VGDL.Agent
    perception::PerceptionModule{W}
    planning::PlanningModule{W}
    world::WorldState{W}
end


"""
    observation_model(a::GenAgent)

The observation model associated with the agent
"""
function observation_model end

struct

function VGDL.plan(agent::GenAgent, i::Int64, obs::ChoiceMapObservation)

end




end
