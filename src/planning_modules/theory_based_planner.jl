export TheoryBasedPlanner

mutable struct TheoryBasedPlanner{W<:WorldModel} <: PlanningModule{W}
    goals::Vector{TerminationRule}
    subgoals
    gradients
    horizon::AStarHorizon
end


# get_all(T)
# eval(get_all(T)) = all(map(eval, [get(x_1), get(x_2), ..., get(x_n)]))

function plan!(planner::TheoryBasedPlanner{<:W}, wm::W, ws::WorldState{<:W}
               ) where {W <: VGDLWorldModel}

    # re-evaluate subgoals?
    new_sg = deconstruct(planner.goals, ws)
    sg_diff = setdiff(planner.subgoals, new_sg)
    if !isempty(sg_diff)
        replan!(planner, wm, ws, new_sg)
    else
        # reweight horizon
        reweight!(planner.horizon, planner.goals)
    end

    action = next_action(planner.horizon)


    scene = ws.gstate.scene
    agent = scene.dynamic[wm.agent_idx]
    action_to_idx(wm.agent, action)
end


mutable struct AStarHorizon{U,T}
    open_set::Vector{U}
    closed_set::Vector{U}
    g_score::Vector{T}
    came_from::Vector{U}
    distmx::Matrix{T}
    subgoals
    gradients
    heuristic
    selected_goal
end

function replan!(planner::TheoryBasedPlanner{<:W}, wm::W, ws::WorldState{<:W},
                 subgoals) where {W <: VGDLWorldModel}

    # 1. convert world state to graph
    gr = to_graph(ws)

    # 2. setup a-star to select subgoal
    gradients = map(get_gradient, subgoals)
    horizon = AStarHorizon(gr, subgoals, gradients)

    # 3. expand horizon with max steps, looking for subgoals
    completed = false
    n = 0
    while !completed || n < budget
        selected = explore_horizon!(horizon)
        completed = selected != 0
        n += 1
    end

end


@gen (static) function astar_production(n::AStarProdNode)
    # goal: select best next action or terminate
    #
    # assuming not terminates yet, pick next action
    aws = action_weights(n)
    aidx = @trace(categorical(aws), :action)
    # deterministic forward step
    next_state = update_step(n, aidx)

    # determine if new state satisfies any goals
    # or if the planning budget is exhausted
    satisfied = map(sg -> satisfies(sg, next_state), n.sub_goals)
    w = production_weight(n, satisfied)
    s = @trace(bernoulli(w), :produce)
    children::Vector{QTProdNode} =
        s ? AStarProdNode(n, next_state) : AStarProdNode[]
    result = Production(n, children)
    return result
end

function production_weight(n::AStarProdNode, satisfied::Vector{Bool})
    (n.step == n.max_steps || any(satisfied)) |> Float64
end

function VGDL.update_step(n::AStarProdNode, action_idx::Int)
    queues = action_step(n.world_state,
                         Dict(n.agent_idx => action_idx))
    update_step(n.world_state, n.imap, queues)
end

function action_weights(n::AStarProdNode)
    # assuming sub-goals are not satisfied
    gradients = map(gr -> gr(n.world_state))
    normed_weights = n.gradients / sum(n.gradients)
end

@gen static function astar_aggregation(n::AStarProdNode,
                                       children::Vector{AStarAggNode})
    agg::AStarAggNode = AStarAggNode(n, children)
    return agg
end

function AStarAggNode(n::AStarProdNode, c::Vector{AStarAggNode})

    if isempty(c)
        # at the end of the frontier
        if n.satisfied === 0
            # exhausted resources

    else
    end
end

const quad_tree_prior = Recurse(astar_production,
                                astar_aggregation,
                                1, # max children
                                QTProdNode,# U (production to children)
                                QTProdNode,# V (production to aggregation)
                                QTAggNode) # W (aggregation to parents)
