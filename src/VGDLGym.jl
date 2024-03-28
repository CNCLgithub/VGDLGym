module VGDLGym

using VGDL
using NearestNeighbors
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
include("attention/attention.jl")

abstract type PerceptionModule{T<:WorldModel} end
abstract type PlanningModule{T<:WorldModel} end
abstract type MemoryModule{T<:WorldModel} end

mutable struct GenAgent{W<:WorldModel,
                        V<:PerceptionModule{W},
                        P<:PlanningModule{W},
                        M<:MemoryModule{W},
                        A<:AttentionModule
                        }<: VGDL.Agent
    world_model::W
    perception::V
    planning::P
    memory::M
    attention::A
end

world_model(a::GenAgent) = a.world_model
agent_idx(a::GenAgent) = agent_idx(world_model(a))


include("memory.jl")
include("perception_modules/perception_modules.jl")
include("planning_modules/planning_modules.jl")

# Define agents that infer the world
# mutable struct GenAgent{W<:WorldModel} <: VGDL.Agent
#     world_model::W
#     tm::TransferModule
#     perception::PerceptionModule{W}
#     planning::PlanningModule{W}
# end


# TODO: dispatch
function perceive!(agent::GenAgent, st::GameState, action::Int)
    @unpack perception, attention, world_model = agent
    obs = perceive!(perception, attention, world_model, st, action)
    return obs
end

function plan!(agent::GenAgent)
    @unpack (world_model, memory, perception, planning,
             attention) = agent
    plan_in = transfer(memory, perception)
    action = plan!(planning, attention, world_model, plan_in)
    return action
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
        obs = perceive!(gym.agent, next_state, action)

        println("Time $(next_state.time)")
        # println("Player position: $(next_state.scene.dynamic[1].position)")
        println("\tWorld State")
        viz_obs(obs)
        println("\tAgent State")
        println("\t\tPercept")
        viz_perception_module(gym.agent.perception)
        println("\t\tHorizon")
        viz_planning_module(gym.agent.planning)

        # update reference to new game state
        gym.state = next_state
    end
    return nothing
end

end
