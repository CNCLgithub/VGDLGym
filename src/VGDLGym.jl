module VGDLGym

using VGDL
using Gen
using Gen_Compose

#################################################################################
# GenAgent
#################################################################################

abstract type WorldModel end

"The graphics module for a world model"
function graphics end

abstract type WorldState{T<:WorldModel} end

include("world_models/world_models.jl")

abstract type GraphicsModule end
abstract type PerceptionModule{T<:WorldModel} end

abstract type TransferModule end
abstract type PlanningModule{T<:WorldModel} end


# Define agents that infer the world
mutable struct GenAgent{W<:WorldModel} <: VGDL.Agent
    world_model::W
    perception::PerceptionModule{W}
    planning::PlanningModule{W}
end

function GenAgent(wm::W,
                  Q::Type{PerceptionModule},
                  P::Type{PlanningModule})
    where {W<:WorldModel}

    perception = Q{W}(wm)
    planning = P{W}(wm)
    GenAgent{W}(wm, perception, planning)
end

"""
    observation_model(a::GenAgent)

The observation model associated with the agent
"""
function observation_model(a::GenAgent)
    graphics(a.world_model)
end


"""
    render(::ObservationModel, ::GameState)

Renders a scene.
"""
function render end

function process!(agent::GenAgent, st::GameState, action::VGDL.Action)
    om = observation_model(agent)
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

function transfer(agent::GenAgent)
    @unpack perception, planning = agent
    transfer(planning, perception)
end

function plan!(agent::GenAgent)
    plan!(agent.planning, agent.wm, transfer(agent))
end

#################################################################################
# Gym
#################################################################################

abstract type Gym end

mutable struct SoloGym <: Gym
    g::Game
    imap::InteractionMap
    tvec::Vector{TerminationRule}

    state::GameState

    agent::GenAgent
end


function SoloGym(g::Game, init_state::GameState, agent::GenAgent)
    imap = compile_interaction_set(g)
    tset = termination_set(g)
    SoloGym(g, imap, tset, init_state, agent)
end

function run_gym!(gym::SoloGym)

    # initialize world model via perception


    while !isfinished(gym.state, gym.tset)
        # plan next action
        action = plan!(gym.agent)

        # receive agent's action and generate actions for NPCs
        queues =
            action_step(gym.g, gym.state, [agent => action])
        # resolve interactions in game state
        next_state  = update_step(gym.state, gym.imap, queues)

        # update agent's percepts
        # also pass the planned action
        perceive!(gym.agent, next_state, action)

        # update reference to new game state
        gym.state = next_state
    end
    return nothing
end

