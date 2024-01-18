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


    (current_t, _, _) = get_args(percept)
    # @show current_t

    # re-evaluate subgoals
    subgoals = pl.subgoals
    new_subgoals::Vector{Goal} =
        reduce(vcat, map(g -> decompose(g, info), pl.goals))
    sg_diff = intersect(subgoals, new_subgoals)

    # remaining length of horizon
    hlen = isnothing(pl.horizon) ?
        0 : horizon_length(pl.horizon, current_t)
    @show hlen

    local horizon::Gen.RecurseTrace, heuristic::Function
    replanned::Bool = false
    if length(sg_diff) !== length(subgoals) || hlen < 1
        # full replan
        println("Full replan")
        gs = game_state(ws)
        d = affordances(gs)
        subgoals = new_subgoals
        heuristic = consolidate(subgoals, d)
        horizon = replan(wm, ws, heuristic,
                         current_t,
                         pl.search_steps)
        # REVIEW: select more than 1?
        subgoals = [subgoals[select_subgoal(horizon)]]
        heuristic = consolidate(subgoals, d)
        horizon = replan(wm, ws, heuristic, current_t,
                         pl.replan_steps)
        replanned = true
    else
        horizon = pl.horizon
        heuristic = get_heursitc(pl.horizon)
        hlen = horizon_length(horizon, current_t)
    end

    # @show replanned

    if !replanned && hlen > 1
        # divergent world state
        (_, alpha) =
            integrate_update(horizon,
                            percept,
                            current_t,
                            pl.integration_steps)

        if rand() > -alpha
            println("replanning from state divergence")
            # replan
            # REVIEW: replan + new subgoal selection?
            # REVIEW: could extend plan from `integrate_update`
            gs = game_state(ws)
            d = affordances(gs)
            heuristic = consolidate(subgoals, d)
            horizon = replan(wm, ws, heuristic, current_t,
                             pl.replan_steps)
            # REVIEW: select more than 1?
            subgoals = [subgoals[select_subgoal(horizon)]]
            heuristic = consolidate(subgoals, d)
            horizon = replan(wm, ws, heuristic, current_t,
                             pl.replan_steps)
            replanned = true
        end
    end

    if !replanned && hlen < 3
        # did not replan for the following reasons:
        # 1) the horizon is too short and needs to be extended
        # 2) predictions have not diverged enough
        # 3) no change in subgoals
        horizon, _ = shift_plan(horizon, current_t,
                                pl.shift_steps)
    end

    pl.horizon = horizon
    pl.subgoals = subgoals

    ai = planned_action(pl, horizon, current_t)
    # @show ai
    ai
end


function replan(wm::W, ws::WorldState{<:W},
                heuristic,
                t::Int64,
                max_steps::Int64
                ) where {W <: VGDLWorldModel}
    # initialize world state at current time
    gs = game_state(ws)
    player = gs.scene.dynamic[1]
    args = (0, ws, wm)
    tr, _ = Gen.generate(vgdl_wm, args)
    # planning node points to current world
    # state and heuristic
    start_node = AStarNode(heuristic,
                           length(actionspace(player)),
                           tr,
                           t,
                           t + max_steps,
                           )

    args = (start_node, t)
    ptr, _ = Gen.generate(astar_recurse, args)
    ptr
end


function splice_plan(source::Gen.Trace,
                     origin::Int64, # where to cut the new trace
                     steps::Int64,
                     node_data::NamedTuple = NamedTuple())

    n = depth(source)
    maxsteps = origin + steps

    # @show origin
    # @show maxsteps
    # @show keys(source.production_traces)
    # extract the new start node
    node = get_node(source, origin) # REVIEW
    node = setproperties(node;
                         node_data...,
                         maxsteps = maxsteps)
    # @show node.step
    # @show first(get_args(node.state))

    # copy over previously planned choices
    cm = choicemap()
    for i = origin:(min(n, maxsteps))
        addr = (i, Val(:production)) => :action
        cm[addr] = source[addr]
    end

    args = (node, origin)
    new_tr, w = Gen.generate(astar_recurse, args, cm)
    new_node = get_node(new_tr, origin)
    # @show new_node.step
    # @show first(get_args(new_node.state))
    new_tr, w
end

function shift_plan(plan_tr::Gen.Trace,
                    current::Int64,
                    steps::Int64)
    # println("shift plan")
    n = depth(plan_tr)
    origin = min(current, n)
    # node_data = (; step = origin)
    splice_plan(plan_tr, origin, steps) #, node_data)
end

function integrate_update(plan_tr::Gen.Trace,
                          state::Gen.Trace,
                          current::Int64,
                          maxsteps::Int64)

    # println("integrate update")
    # update with new state
    node_data = (; state = state)
    # adjust horizon size
    n = depth(plan_tr)
    k = min(n - current, maxsteps)
    # @show current
    # @show k
    # generate from current -> k,
    # using new local start node
    # and copying relevant action decisions from `plan_tr`
    new_tr, nw = splice_plan(plan_tr, current, k, node_data)

    # compute probability of original subplan
    # TODO: project is not implemented for `Recurse`
    # prev_w = Gen.project(plan_tr, selection)
    prev_w::Float64 = 0.0
    k = depth(new_tr)
    # @show k
    # @show(keys(new_tr.production_traces))
    for t = current:k
        # @show t
        prev_w += Gen.project(plan_tr.production_traces[t],
                              select(:action))
    end
    (new_tr, nw - prev_w)
end

#################################################################################
# Horizon interface
#################################################################################

function select_subgoal(horizon::Gen.RecurseTrace)
    final_step = get_retval(horizon)
    @unpack heuristics = final_step
    display(heuristics)
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
    subtrace = tr.production_traces[i]
    get_retval(subtrace).value
end
function get_node(tr::Gen.RecurseTrace, i::Int64)
    subtrace = tr.production_traces[i]
    last(get_args(subtrace))
end

function consolidate(sgs::Vector{<:Goal}, agraph)
    n = length(sgs)
    # REVIEW: convert to struct instead of anon func?
    tr::Gen.Trace -> begin
        _, _, wm = get_args(tr)
        st = game_state(last(get_retval(tr)))
        agent = st.scene.dynamic[wm.agent_idx]
        dy, dx = st.scene.bounds
        lpos = get_vertex(st.scene.bounds, agent.position)
        d = gdistances(agraph, lpos)
        info = Info(st, d)
        results = Vector{Float64}(undef, n)
        @inbounds for i = 1:n
            results[i] = gradient(sgs[i], info)
        end
        results
    end
end

function horizon_length(tr::Gen.RecurseTrace, t::Int64)
    depth(tr) - t
end

function get_heursitc(tr::Gen.RecurseTrace)
    snode, _ = get_args(tr)
    snode.heuristic
end

function planned_action(pl::TheoryBasedPlanner{<:W}, tr::Gen.RecurseTrace,
                        t::Int64) where {W<:VGDLWorldModel}
    step = get_step(tr, t)
    @show step.heuristics
    step.action
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
    agg_hs = map(maximum, hs)
    aws = softmax(agg_hs, 0.01)
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
    (n.step < n.maxsteps && !any(iszero, heuristic)) ? 1.0 : 0.0
end

function evolve(n::AStarNode, action::Int)
    prev = n.state
    t, ws, wm = get_args(prev)
    args = (t + 1, ws, wm)
    argdiffs = (UnknownChange(), NoChange(), NoChange())
    gs = t == 0 ? ws.gstate : last(get_retval(prev)).gstate
    cm = choicemap()
    cm[:kernel => (t + 1) => :agents =>
        wm.agent_idx => :action] = action
    n = length(gs.scene.dynamic)
    for ai = 2:n
        cm[:kernel => (t + 1) => :agents =>
            ai => :action] = 5 # no action
    end

    next, _... = Gen.update(prev, args, argdiffs, cm)
    next
end

function explore(n::AStarNode)
    map(a -> evolve(n, a), 1:(n.nactions))
end

function AStarNode(prev::AStarNode, next_state::Gen.Trace)
    setproperties(prev;
                  state = next_state,
                  step = prev.step + 1)
end

const astar_recurse = Recurse(astar_production,
                              astar_aggregation,
                              1, # max children
                              AStarNode,# U (production to children)
                              AStarStep,# V (production to aggregation)
                              AStarStep) # W (aggregation to parents)
