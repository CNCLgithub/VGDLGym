#################################################################################
# Exports
#################################################################################
export replan!,
    extend_plan,
    integrate_update

#################################################################################
# TheoryBasedPlanner
#################################################################################

mutable struct TheoryBasedPlanner{W<:WorldModel} <: PlanningModule{W}
    world_model::W
    goals::Vector{Goal}
    horizon::Gen.Trace # from AStarRecurse
    agraph::SimpleGraph
end


function plan!(planner::TheoryBasedPlanner{<:W}, wm::W, ws::WorldState{W},
               ) where {W <: WorldModel}

    gs = gamestate(ws)
    info = Info(gs, )
    # re-evaluate subgoals?
    new_sg = decompose(planner.goals, ws)
    sg_diff = setdiff(planner.subgoals, new_sg)
    if !isempty(sg_diff)
        replan!(planner, wm, ws, ps, new_sg)
    else
        # reweight horizon
        reweight!(planner.horizon, planner.goals)
    end

    action = next_action(planner.horizon)
    agent = scene.dynamic[wm.agent_idx]
    action_to_idx(wm.agent, action)
end

function consolidate(sgs::Vector{<:Goal}, agraph)
    n = length(sgs)

    tr::Gen.Trace -> begin
        _, _, wm = get_args(tr)
        st = game_state(last(get_retval(tr)))

        agent = st.scene.dynamic[wm.agent_idx]
        dy, dx = st.scene.bounds
        lpos = (agent.position[2] - 1) * dy + agent.position[2]
        d = gdistances(agraph, lpos)
        info = Info(st, d)
        results = Vector{Float64}(undef, n)
        @inbounds for i = 1:n
            results[i] = gradient(sgs[i], info)
        end
        results
    end
end

# function replan!(planner::TheoryBasedPlanner{<:W}, wm::W, ws::WorldState{<:W},
function replan!(wm::W, ws::WorldState{<:W},
                 subgoals::Vector{<:Goal},
                 max_steps::Int64
                 ) where {W <: VGDLWorldModel}
    d = affordances(game_state(ws))
    heuristic = consolidate(subgoals, d)
    args = (0, ws, wm)
    tr, _ = Gen.generate(vgdl_wm, args)
    start_node = AStarNode(heuristic,
                           wm.nactions,
                           tr,
                           0,
                           max_steps,
                           )

    ptr, _ = Gen.generate(astar_recurse, (start_node, 1))
    ptr
end

function extend_plan(plan_tr::Gen.Trace, steps::Int64)
    n = length(plan_tr.production_traces)
    start_node = get_retval(plan_tr).node
    start_node = setproperties(start_node;
                               maxsteps = n + steps)
    extended, _... = Gen.generate(astar_recurse, (start_node, n + 1))
    extended
end

function integrate_update(plan_tr::Gen.Trace,
                          state::Gen.Trace,
                          current::Int64,
                          maxsteps::Int64)
    # retreive node that will be revisited
    choices = get_choices(plan_tr)
    n = length(plan_tr.production_traces)
    parent_addr = (current - 1, Val(:production))
    # parent_subtrace = get_submap(choices, parent_addr).trace
    parent_subtrace = plan_tr.production_traces[current - 1]
    node = get_retval(parent_subtrace).value.node
    # update node with new state and adjust horizon size
    k = min(n, current + maxsteps - 1)
    node = setproperties(node;
                         state = state,
                         maxsteps = k)
    # extract choices from current plan
    cm = choicemap()
    prev_w::Float64 = 0.0
    for t = current:k
        addr = (t, Val(:production)) => :action
        cm[addr] = plan_tr[addr]
        prev_w += Gen.project(plan_tr.production_traces[t],
                               select(:action))
    end
    # TODO: project is not implemented for `Recurse`
    # prev_w = Gen.project(plan_tr, sl)
    new_tr, nw = Gen.generate(astar_recurse,
                              (node, current),
                              cm)
    nw - prev_w
end


#################################################################################
# AStarRecurse
#################################################################################

struct AStarNode
    "(state) -> [value]"
    heuristic::Function
    "Number of actions the agent can take"
    nactions::Int64
    "Current predicted world state"
    state::Gen.Trace
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

struct AStarStep
    node::AStarNode
    action::Int64
    heuristics::Vector{Vector{Float64}}
end

@gen (static) function astar_production(n::AStarNode)
    # new states from each possible action
    next_states = explore(n)
    hfunc = n.heuristic
    hs = map(hfunc, next_states)
    # aggregate heuristic score per action
    agg_hs = map(logsumexp, hs)
    aws = softmax(agg_hs)
    action = @trace(categorical(aws), :action)
    # predicted state associated with action
    next_state = next_states[action]
    step_heuristics = hs[action]
    node = AStarNode(n, next_state) # passed to children
    step = AStarStep(node, action, hs) # passed to aggregrate
    # determine if new state satisfies any goals
    # or if the planning budget is exhausted
    # if `next_state` fails, the trace will terminate
    # and go back to the planner to regenerate a new branch
    w = production_weight(n, step_heuristics)
    s = @trace(bernoulli(w), :produce) # REVIEW: make deterministic?
    children::Vector{AStarNode} = s ? [node] : AStarNode[]
    result::Production{AStarStep, AStarNode} =
        Production(step, children)
    return result
end

@gen static function astar_aggregation(s::AStarStep,
                                       children::Vector{AStarStep})

    leaf::AStarStep = isempty(children) ? s : first(children)
    return leaf
end

function production_weight(n::AStarNode, heuristic::Vector{Float64})
    (n.step+1 < n.maxsteps && !any(iszero, heuristic)) |> Float64
end

function evolve(n::AStarNode, action::Int)
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

function explore(n::AStarNode)
    map(a -> evolve(n, a), 1:(n.nactions))
end

function AStarNode(prev::AStarNode, next_state::Gen.Trace)
    # @show (prev.heuristic(next_state))
    setproperties(prev; state = next_state, step = prev.step + 1)
end

const astar_recurse = Recurse(astar_production,
                              astar_aggregation,
                              1, # max children
                              AStarNode,# U (production to children)
                              AStarStep,# V (production to aggregation)
                              AStarStep) # W (aggregation to parents)
