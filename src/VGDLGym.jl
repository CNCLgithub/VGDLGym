module VGDLGym

using VGDL
using Gen
using Gen_Compose
using GenParticleFilters
using Parameters
using FillArrays: Fill
using DataStructures: OrderedDict, PriorityQueue
using Accessors
using StaticArrays
using Graphs

#################################################################################
# Exports
#################################################################################
export WorldModel,
    WorldState,
    AttentionModule,
    PerceptionModule,
    TransferModule,
    PlanningModule,
    GenAgent,
    Gym,
    run_gym!,
    SoloGym

#################################################################################
# GenAgent
#################################################################################

abstract type WorldModel end

"The graphics module for a world model"
function graphics end

abstract type WorldState{T<:WorldModel} end

include("utils/utils.jl")
# TODO: re-organize
include("goals.jl")
include("world_models/world_models.jl")

abstract type AttentionModule end
include("attention.jl")

abstract type PerceptionModule{T<:WorldModel} end
abstract type TransferModule end
abstract type PlanningModule{T<:WorldModel} end

include("transfer.jl")
include("perception_modules/perception_modules.jl")
include("planning_modules/planning_modules.jl")

# Define agents that infer the world
mutable struct GenAgent{W<:WorldModel} <: VGDL.Agent
    world_model::W
    tm::TransferModule
    perception::PerceptionModule{W}
    planning::PlanningModule{W}
end


function perceive!(agent::GenAgent, st::GameState, action::Int)
    gr = graphics(agent.world_model)
    obs = render(gr, st)
    cm = Gen.choicemap()
    cm[:kernel => st.time => :observe] = obs
    println("Time $(st.time)")
    viz_obs(obs)
    # add action index as constraint
    agent_idx = agent.world_model.agent_idx # TODO: getter
    cm[:kernel => st.time => :agent => agent_idx] =
        action

    perceive!(agent.perception, cm, st.time)
    viz_world_state(agent.perception)
    return nothing
end

function plan!(agent::GenAgent)
    @unpack world_model, tm, perception, planning = agent
    transfer(tm, world_model, perception, planning)
end

#################################################################################
# Gym
#################################################################################

abstract type Gym end

mutable struct SoloGym <: Gym
    imap::InteractionMap
    tvec::Vector{TerminationRule}

    state::GameState

    agent::GenAgent
    agent_idx::Int
end


function run_gym!(gym::SoloGym)

    agent_idx = gym.agent_idx
    action = 0 # not used in first loop
    while !isfinished(gym.state, gym.tvec)

        # plan next action
        action = plan!(gym.agent)

        # receive agent's action and generate actions for NPCs
        queues =
            action_step(gym.state, Dict(agent_idx => action))
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

end
