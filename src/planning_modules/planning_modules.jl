
struct MAPTransfer <:TransferModule end
const map_transfer = MAPTransfer()

function transfer(::MAPTransfer, perception::IncPerceptionModule{<:W})
    where {W <: WorldModel}

    @unpack chain  = perception
    @unpack state = chain

    map_idx = argmax(state.logscores)
    Gen.get_retval(state.traces[map_idx])# ::W
end

mutable struct RandomPlanner{W} <: PlanningModule{W<:WorldModel}
    tm::TransferModule
end


function plan!(planner::RandomPlanner{<:W}, wm::W, ws::WorldState{<:W})
    where {W <: VGDLWorldModel}

    # select random action
    aspace = actionspace(ws.gstate.agents[wm.agent_idx])
    n = length(aspace)
    aidx = categorical(Fill(1.0 / n, n))
    action = aspace[aidx]
    return action(wm.agent_idx)
end
