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

    next_states = map(a -> n.evolve(n.state, a), 1:n.nactions)
    hs = map(n.heuristic, next_states) # in logspace [-Inf, 0]
    aws = softmax(hs)
    action = @trace(categorical(hs), :action)
    # deterministic forward step
    next_state = next_states[action]

    # determine if new state satisfies any goals
    # or if the planning budget is exhausted
    # if `next_state` fails, the trace will terminate
    # and go back to the planner to regenerate a new branch
    w = production_weight(n, hs)
    s = @trace(bernoulli(w), :produce) # REVIEW: make deterministic?
    children::Vector{AStarProdNode} =
        s ? [AStarProdNode(n, next_state)] : AStarProdNode[]
    step_reward = hs[action]
    result = Production(step_reward, children)
    return result
end

@gen static function astar_aggregation(r::Float64,
                                       children::Vector{Float64})
    total_reward::Float64 = r + sum(children)
    return total_reward
end

function production_weight(n::AStarProdNode, heuristics::Vector{Float64})
    (n.step < n.max_steps || !(any(iszero, satisfied))) |> Float64
end

function VGDL.update_step(n::AStarProdNode, action::Int)
    prev = n.state
    t, init_state, wm = get_args(prev)
    args = (t + 1, init_state, wm)
    argdiffs = (UnknownChange(), NoChange(), NoChange())
    cm = Gen.choicemap(
        (:kernel => (t+1) => :agent => wm.agent_idx, action)
    )
    next, _... = Gen.update(prev, args, argdiffs, cm)
    next
end

function AStarNode(prev::AStarProdNode, next_state::Gen.Trace)
    setproperties(prev; state = next_state, step = prev.step + 1)
end

struct AStarNode
    "(state) -> value"
    heuristic::Function
    nactions::Int64
    state::Gen.Trace
    "Function (state, action) -> new state"
    evolve::Function
    step::Int64
    maxsteps::Int64
end


struct AStarAggNode
    actions::PersistentList{Int64}
    actions::PersistentList{Int64}
end

function AStarAggNode(n::AStarProdNode, c::Vector{AStarAggNode})

    # at the end of the frontier
    # either because exhausted resources
    # or because a (+) subgoal is reached
    seq = isempty(c) ? PerstitentList{Int64}() : first(c)

    # select subgoal
    sg_idx = argmax(n.gradients)

    # add action
    cons(n.action, seq)
end

const quad_tree_prior = Recurse(astar_production,
                                astar_aggregation,
                                1, # max children
                                QTProdNode,# U (production to children)
                                QTProdNode,# V (production to aggregation)
                                QTAggNode) # W (aggregation to parents)
