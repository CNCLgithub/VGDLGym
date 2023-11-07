using Gen
using VGDL
using VGDLGym
using VGDLGym: reference, retrieve
using Statistics
# using Profile
# using StatProfilerHTML

function viz_trace(gr, tr::Gen.Trace)
    (t, ws, wm) = get_args(tr)
    state = t == 0 ? ws.gstate : last(get_retval(tr)).gstate
    img = render(gr, state)
    VGDLGym.viz_obs(img)
end
function viz_batch(gr, trs::Vector{<:Gen.Trace})
    for i = 1:length(trs)
        st = last(get_retval(trs[i]))
        # display(choicemap(get_submap(get_choices(trs[i]),
        #                      :kernel => 1 )))
        # @show st.gstate.scene.dynamic[1].position
    end
    f = tr::Gen.Trace -> render(gr, last(get_retval(tr)).gstate)
    img = mean(map(f, trs))
    VGDLGym.viz_obs(img)
end

function step_frontier(wm, ws,
                heuristic,
                t::Int64,
                max_steps::Int64
                )
    gs = VGDLGym.game_state(ws)
    args = (0, ws, wm)
    tr, _ = Gen.generate(vgdl_wm, args)
    # planning node points to current world
    # state and heuristic
    start_node = VGDLGym.AStarNode(heuristic,
                           wm.nactions,
                           tr,
                           t,
                           t + max_steps,
                           )
    (tr, VGDLGym.explore(start_node))
end

function test()
    G = ButterflyGame
    imap = compile_interaction_set(G)
    tset = termination_set(G)
    init_state = load_level(G, 1)
    # limit time
    init_state.max_time = 500

    @assert typeof(init_state.scene.dynamic[1]) <: Player
    @show init_state.scene.dynamic[1].position
    noise = 0.2 # noise for graphics

    wm, ws = init_world_model(VGDLWorldModel,
                              G,
                              PixelGraphics,
                              init_state,
                              noise)

    goals = [
        Goal(AllRef{Butterfly}(), Get()),
        # Goal(AllRef{Pinecone}(), Count())
    ]

    info = Info(wm, ws)
    sgs = reduce(vcat, map(g -> decompose(g, info), goals))

    agraph = affordances(init_state)
    heuristic = consolidate(sgs, agraph)

    init_trace, traces = step_frontier(wm, ws, heuristic, 0, 1)
    viz_trace(wm.graphics, init_trace)
    viz_batch(wm.graphics, traces)

    aws = map(maximum, map(heuristic, traces))
    display(aws)
    display(VGDLGym.softmax(aws, 0.01))

    # println("replan")
    @time tr = replan(wm, ws, heuristic, 0, 5)
    # choices = get_choices(tr)

    selected_subgoal = select_subgoal(tr)
    @show selected_subgoal
    subgoal = sgs[selected_subgoal]
    @show (retrieve(reference(subgoal), info)).position
    small_heuristic = consolidate([subgoal], agraph)
    steps = 10
    tr = replan(wm, ws, small_heuristic, 0, steps)

    for i = 1:6
        @show i
        step = VGDLGym.get_step(tr, i)
        @show step.action
        @show map(maximum, step.heuristics)
        viz_trace(wm.graphics, step.node.state)
    end
    # small_heuristic = consolidate([sgs[selected_subgoal]], agraph)
    # @time small_tr = replan(wm, ws, small_heuristic, 5)
    # # display(choices)
    # # display(get_choices(small_tr))

    # # @show typeof(get_retval(tr.production_traces[1]))
    # # @show typeof(get_submap(choices, (1, Val(:production))))
    # # @show choices[(1, Val(:production)) => :action]

    # println("plan shift")
    # @time (shifted, sw) = shift_plan(tr, 5, 5)
    # # display(choices)
    # # display(get_choices(shifted))

    # node = get_retval(get_submap(choices, (1, Val(:production))).trace).value.node
    # action = choices[(1, Val(:production)) => :action]
    # alternate_state = VGDLGym.evolve(node, action)

    # println("integrate update")
    # @time (new_tr, w) = integrate_update(tr, alternate_state, 2, 2)
    # @show w


    
    # @show tr[(1, Val(:production))]
    # @show length(sgs)
    # @show get_retval(tr)
    # Profile.init(n = 10^7, delay = 0.0001)
    # Profile.clear()
    # @profilehtml replan!(wm, ws, sgs);

    # display(get_choices(tr))
end

test();
