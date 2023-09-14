module VGDLGym

using VGDL
using Gen
using Gen_Compose

# Define agents that infer the world
abstract type GenAgent <: VGDL.Agent end

abstract type ObservationModel end

"""
    render(::ObservationModel, ::GameState)

Produces a `Gen.ChoiceMap` that can be used for inverse-graphics.
"""
function render end

include("observation_models.jl")

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

function VGDL.plan(agent::GenAgent, i::Int64, obs::ChoiceMapObservation)

end




end
