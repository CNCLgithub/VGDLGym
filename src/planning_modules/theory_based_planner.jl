export TheoryBasedPlanner


function to_graph(ws::VGDLWorldState)
    state = ws.gstate
    dy, _ = state.scene.bounds
    nv = length(state.scene.static)
    adj_matrix = fill(false, (nv, nv))
    for i = 1:(nv -1), j = (i+1):nv
        d = abs(j - i)
        !(d == 1 || d == dy) && continue
        adj_matrix[i, j] = adj_matrix[j, i ] = true
    end
    SimpleGraph(adj_matrix)
end

function deconstruct(goals::Vector{Goal}, ws::VGDLWorldState, wm::VGDLWorldModel)
    state = ws.gstate
    g = to_graph(ws)
    agent = state.scene.dynamic[wm.agent_idx]
    dy, dx = state.scene.bounds
    lpos = (agent.pos[2] - 1) * dy + agent.pos[2]
    gds = gdistances(g, lpos)
    fdist = (el) -> begin
        y,x = el.pos
        idx = (x - 1) * dy + y
        -log(gds[idx] + 1)
    end

    sub_goals = vcat(map(g -> decompose(g, state), goals))
    gradients =

end








mutable struct TheoryBasedPlanner{W<:WorldModel} <: PlanningModule{W}
    goals::Vector{TerminationRule}
    subgoals
    gradients
    horizon::AStarHorizon
end


# get_all(T)
# eval(get_all(T)) = all(map(eval, [get(x_1), get(x_2), ..., get(x_n)]))

function plan!(planner::TheoryBasedPlanner{<:W}, wm::W, ws::WorldState{<:W}
               ) where {W <: WorldModel}

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

mutable struct TBPState{W<:WorldModel}
    horizon::Gen.Trace
    goals::Vector{<:Goal{W}}
    sub_goals::Vector{<:SubGoal{W}}

end

#

function replan!(planner::TheoryBasedPlanner{<:W}, wm::W, ws::WorldState{<:W},
                 subgoals) where {W <: VGDLWorldModel}

    # 1. convert world state to graph
    gr = to_graph(ws)

    # 2. setup a-star to select subgoal
    gradients = map(gradient, subgoals)
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


@gen (static) function astar_production(n::AStarNode)
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
    children::Vector{AStarNode} =
        s ? [AStarNode(n, next_state)] : AStarNode[]
    step_reward = hs[action]
    result = Production(step_reward, children)
    return result
end

@gen static function astar_aggregation(r::Float64,
                                       children::Vector{Float64})
    total_reward::Float64 = r + sum(children)
    return total_reward
end

function production_weight(n::AStarNode, heuristics::Vector{Float64})
    (n.step < n.max_steps || !(any(iszero, satisfied))) |> Float64
end

function VGDL.update_step(n::AStarNode, action::Int)
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

function AStarNode(prev::AStarNode, next_state::Gen.Trace)
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

# struct AStarAggNode
#     actions::PersistentList{Int64}
#     actions::PersistentList{Int64}
# end

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

const astar_gm = Recurse(astar_production,
                         astar_aggregation,
                         1, # max children
                         AStarNode,# U (production to children)
                         Float64,# V (production to aggregation)
                         Float64) # W (aggregation to parents)
