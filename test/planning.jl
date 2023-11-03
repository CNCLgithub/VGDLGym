using Gen
using VGDL
using VGDLGym
# using Profile
# using StatProfilerHTML


function test()
    G = ButterflyGame
    imap = compile_interaction_set(G)
    tset = termination_set(G)
    init_state = load_level(G, 1)
    # limit time
    init_state.max_time = 500

    @assert typeof(init_state.scene.dynamic[1]) <: Player

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

    println("replan")
    @time tr = replan(wm, ws, heuristic, 5)
    choices = get_choices(tr)

    selected_subgoal = select_subgoal(tr)
    @show selected_subgoal
    small_heuristic = consolidate([sgs[selected_subgoal]], agraph)
    @time small_tr = replan(wm, ws, small_heuristic, 5)
    # display(choices)
    # display(get_choices(small_tr))

    # @show typeof(get_retval(tr.production_traces[1]))
    # @show typeof(get_submap(choices, (1, Val(:production))))
    # @show choices[(1, Val(:production)) => :action]

    println("plan shift")
    @time (shifted, sw) = shift_plan(tr, 5, 5)
    # display(choices)
    # display(get_choices(shifted))

    node = get_retval(get_submap(choices, (1, Val(:production))).trace).value.node
    action = choices[(1, Val(:production)) => :action]
    alternate_state = VGDLGym.evolve(node, action)

    println("integrate update")
    @time (new_tr, w) = integrate_update(tr, alternate_state, 2, 2)
    @show w


    
    # @show tr[(1, Val(:production))]
    # @show length(sgs)
    # @show get_retval(tr)
    # Profile.init(n = 10^7, delay = 0.0001)
    # Profile.clear()
    # @profilehtml replan!(wm, ws, sgs);

    # display(get_choices(tr))
end

test();
