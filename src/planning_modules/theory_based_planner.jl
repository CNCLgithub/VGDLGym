export replan!

function consolidate(sgs::Vector{Goal})
    tr::Gen.Trace -> begin
        _, _, wm = get_args(tr)
        st = game_state(last(get_retval(tr)))
        info = Info(st, wm.agent_idx)
        sum(map(g -> gradient(g, info), sgs))
    end
end

# function replan!(planner::TheoryBasedPlanner{<:W}, wm::W, ws::WorldState{<:W},
function replan!(wm::W, ws::WorldState{<:W},
                 subgoals::Vector{Goal},
                 ) where {W <: VGDLWorldModel}
    heuristic = consolidate(subgoals)
    args = (0, ws, wm)
    tr, _ = Gen.generate(vgdl_wm, args)
    start_node = AStarNode(heuristic,
                           wm.nactions,
                           tr,
                           evolve,
                           0,
                           10,
                           )

    ptr, _ = Gen.generate(astar_plan, (start_node,))
    ptr
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


function AStarNode(maxsteps::Int64,
                   subgoals::Vector{Goal},
                   nactions::Int64,
                   forward::Function,
                   world_state::Gen.Trace)

    AStarNode(consolidate(subgoals),
              nactions,
              world_state,
              forward,
              0,
              maxsteps)
end


@gen (static) function astar_production(n::AStarNode)
    # goal: select best next action or terminate
    #
    next_states = explore(n)
    # heuristic refs could fail after action
    # example: butterfly is caught
    # question: should heuristic return 0?
    # question: should subgoals be revised?
    hfunc = n.heuristic
    hs = map(hfunc, next_states) # in logspace [-Inf, 0]
    aws = softmax(hs)
    action = @trace(categorical(aws), :action)
    # deterministic forward step
    next_state = next_states[action]
    step_heuristic = hs[action]

    # determine if new state satisfies any goals
    # or if the planning budget is exhausted
    # if `next_state` fails, the trace will terminate
    # and go back to the planner to regenerate a new branch
    w = production_weight(n, step_heuristic)
    s = @trace(bernoulli(w), :produce) # REVIEW: make deterministic?
    children::Vector{AStarNode} =
        s ? [AStarNode(n, next_state)] : AStarNode[]
    result = Production(step_heuristic, children)
    return result
end

@gen static function astar_aggregation(r::Float64,
                                       children::Vector{Float64})
    total_reward::Float64 = r + get(children, 1, 0.0)
    return total_reward
end

function production_weight(n::AStarNode, heuristic::Float64)
    (n.step < n.maxsteps && heuristic < 0) |> Float64
end

function evolve(n::AStarNode, action::Int)
    prev = n.state
    t, init_state, wm = get_args(prev)
    args = (t + 1, init_state, wm)
    argdiffs = (UnknownChange(), NoChange(), NoChange())
    cm = Gen.choicemap(
        (:kernel => (t+1) => :agent => wm.agent_idx, action)
    )
    next, ls, _... = Gen.update(prev, args, argdiffs, cm)
    next
end

function explore(n::AStarNode)
    efunc = n.evolve
    steps = 1:(n.nactions)
    map(a -> efunc(n, a), steps)
end

function AStarNode(prev::AStarNode, next_state::Gen.Trace)
    # @show (prev.heuristic(next_state))
    setproperties(prev; state = next_state, step = prev.step + 1)
end

# struct AStarAggNode
#     actions::PersistentList{Int64}
#     actions::PersistentList{Int64}
# end


const astar_recurse = Recurse(astar_production,
                              astar_aggregation,
                              1, # max children
                              AStarNode,# U (production to children)
                              Float64,# V (production to aggregation)
                              Float64) # W (aggregation to parents)

@gen function astar_plan(start_node::AStarNode)
    plan_cost = @trace(astar_recurse(start_node, 1), :recurse)
    return plan_cost
end


# export TheoryBasedPlanner


# mutable struct TheoryBasedPlanner{W<:WorldModel} <: PlanningModule{W}
#     goals::Vector{TerminationRule}
#     horizon::Gen.Trace
# end


# # get_all(T)
# # eval(get_all(T)) = all(map(eval, [get(x_1), get(x_2), ..., get(x_n)]))

# function plan!(planner::TheoryBasedPlanner{<:W}, wm::W, ws::WorldState{W},
#                ) where {W <: WorldModel}

#     ws = get_world_state(wm, ps)
#     # re-evaluate subgoals?
#     new_sg = deconstruct(planner.goals, ws)
#     sg_diff = setdiff(planner.subgoals, new_sg)
#     if !isempty(sg_diff)
#         replan!(planner, wm, ws, ps, new_sg)
#     else
#         # reweight horizon
#         reweight!(planner.horizon, planner.goals)
#     end

#     action = next_action(planner.horizon)
#     agent = scene.dynamic[wm.agent_idx]
#     action_to_idx(wm.agent, action)
# end
