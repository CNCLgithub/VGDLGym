#################################################################################
# Exports
#################################################################################
export consolidate,
    select_subgoal,
    replan,
    shift_plan,
    integrate_update,
    TheoryBasedPlanner

#################################################################################
# TheoryBasedPlanner
#################################################################################

mutable struct TheoryBasedPlanner{W<:WorldModel} <: PlanningModule{W}
    world_model::W
    goals::Vector{Goal}
    search_steps::Int64
    replan_steps::Int64
    integration_steps::Int64
    shift_steps::Int64
    horizon::Union{Gen.RecurseTrace, Nothing}
    subgoals::Vector{Goal}
end

function plan!(pl::TheoryBasedPlanner{<:W},
               wm::W,
               percept::Gen.Trace,
               ) where {W <: WorldModel}

    ws::WorldState{W} = extract_ws(wm, percept)
    info = Info(wm, ws)

    # re-evaluate subgoals
    subgoals = pl.subgoals
    new_subgoals::Vector{Goal} =
        reduce(vcat, map(g -> decompose(g, info), pl.goals))
    sg_diff = setdiff(subgoals, new_subgoals)

    local horizon::Gen.RecurseTrace
    replanned::Bool = false
    if length(sg_diff) !== length(subgoals) || isnothing(pl.horizon)
        # full replan
        gs = game_state(ws)
        d = affordances(gs)
        subgoals = new_subgoals
        heuristic = consolidate(subgoals, d)
        horizon = replan(wm, ws, heuristic,
                         pl.search_steps)
        # REVIEW: select more than 1?
        subgoals = [subgoals[select_subgoal(horizon)]]
        heuristic = consolidate(subgoals, d)
        horizon = replan(wm, ws, heuristic,
                         pl.replan_steps)
        replanned = true
    else
        horizon = pl.horizon
    end


    (current_t, _, _) = get_args(percept)
    if !isnothing(pl.horizon)
        # divergent world state
        (_, alpha) =
            integrate_update(horizon,
                            percept,
                            current_t,
                            pl.integration_steps)

        if rand() > -alpha
            # replan
            # REVIEW: replan + new subgoal selection?
            # REVIEW: could extend plan from `integrate_update`
            horizon = replan(wm, ws, heuristic,
                            pl.replan_steps)
            replanned = true
        end
    end

    if !replanned
        horizon = shift_plan(horizon, current_t,
                             pl.shift_steps)
    end

    pl.horizon = horizon
    pl.subgoals = subgoals

    planned_action(pl, horizon, current_t+1)
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

function planned_action(pl::TheoryBasedPlanner{<:W}, tr::Gen.RecurseTrace,
                        t::Int64) where {W<:VGDLWorldModel}
    step = get_step(tr, t)
    step.action
end

function replan(wm::W, ws::WorldState{<:W},
                heuristic,
                max_steps::Int64
                ) where {W <: VGDLWorldModel}
    # initialize world state at current time
    gs = game_state(ws)
    args = (0, ws, wm)
    tr, _ = Gen.generate(vgdl_wm, args)
    # planning node points to current world
    # state and heuristic
    start_node = AStarNode(heuristic,
                           wm.nactions,
                           tr,
                           0,
                           max_steps,
                           )

    args = (start_node, gs.time + 1)
    ptr, _ = Gen.generate(astar_recurse, args)
    ptr
end

function select_subgoal(horizon::Gen.RecurseTrace)
    final_step = get_retval(horizon)
    @unpack heuristics = final_step
    n = length(heuristics[1]) # subgoals
    v = fill(-Inf, n)
    @inbounds for a = 1:length(heuristics)
        v = max.(v, heuristics[a])
    end
    sgi = argmax(v)
end

function depth(tr::Gen.RecurseTrace)
    ks = keys(tr.production_traces)
    maximum(ks)
end

function get_step(tr::Gen.RecurseTrace, i::Int64)
    parent_subtrace = tr.production_traces[i]
    get_retval(parent_subtrace).value
end
function get_node(tr::Gen.RecurseTrace, i::Int64)
    get_step(tr, i).node
end

function splice_plan(source::Gen.Trace,
                     origin::Int64, # where to cut the new trace
                     steps::Int64,
                     node_data::NamedTuple = NamedTuple())

    n = length(source.production_traces)
    origin = min(origin, n)
    maxsteps = origin + steps

    # extract the new start node
    node = get_node(source, origin)
    node = setproperties(node;
                         node_data...,
                         maxsteps = maxsteps)

    # copy over previously planned choices
    cm = choicemap()
    for i = origin:(min(n, maxsteps))
        addr = (i, Val(:production)) => :action
        cm[addr] = source[addr]
    end

    args = (node, origin)
    new_tr, w = Gen.generate(astar_recurse, args, cm)
end

function shift_plan(plan_tr::Gen.Trace,
                    current::Int64,
                    steps::Int64)
    n = depth(plan_tr)
    current = min(current, n)
    splice_plan(plan_tr, current, steps + 1)
end

function integrate_update(plan_tr::Gen.Trace,
                          state::Gen.Trace,
                          current::Int64,
                          maxsteps::Int64)

    # update with new state
    node_data = (; state = state)
    # adjust horizon size
    n = depth(plan_tr)
    k = min(n, current + maxsteps - 1)
    @show current
    @show k
    display(get_choices(plan_tr))
    # generate from current -> k,
    # using new local start node
    # and copying relevant action decisions from `plan_tr`
    new_tr, nw = splice_plan(plan_tr, current, k, node_data)

    # compute probability of original subplan
    # TODO: project is not implemented for `Recurse`
    # prev_w = Gen.project(plan_tr, selection)
    prev_w::Float64 = 0.0
    for t = current:depth(new_tr)
        prev_w += Gen.project(plan_tr.production_traces[t],
                              select(:action))
    end
    (new_tr, nw - prev_w)
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
