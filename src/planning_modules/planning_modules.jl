export RandomPlanner,
    GreedyPlanner

struct RandomPlanner{W<:WorldModel} <: PlanningModule{W} end

function plan!(planner::RandomPlanner{<:W}, wm::W, ws::WorldState{<:W}
               ) where {W <: VGDLWorldModel}
    # select random action
    agent = ws.gstate.scene.dynamic[wm.agent_idx]
    aspace = actionspace(agent)
    n = length(aspace)
    aidx = categorical(Fill(1.0 / n, n))
end

struct GreedyPlanner{W<:WorldModel} <: PlanningModule{W} end

function plan!(planner::GreedyPlanner{<:W}, wm::W, ws::WorldState{<:W}
               ) where {W <: VGDLWorldModel}
    scene = ws.gstate.scene
    agent = scene.dynamic[wm.agent_idx]
    action = plan(agent, DirectObs(ws.gstate), greedy_policy)
    action_to_idx(agent, action)
end
