using VGDL
using VGDLGym


function test()
    G = ButterflyGame
    imap = compile_interaction_set(G)
    tset = termination_set(G)
    init_state = load_level(G, 1)
    # limit time
    init_state.max_time = 100

    @assert typeof(init_state.scene.dynamic[1]) <: Player

    noise = 0.1 # noise for graphics

    wm, ws = init_world_model(VGDLWorldModel,
                              G,
                              PixelGraphics,
                              init_state,
                              noise)

    tm = map_transfer

    proc_args = (100, 50, uniform_attention)
    q = IncPerceptionModule(vgdl_wm_perceive,
                            wm,
                            ws,
                            proc_args)

    goals = [
        Goal(AllRef{Butterfly}(), Get()),
        # Goal(AllRef{Pinecone}(), Count())
    ]
    p = TheoryBasedPlanner{VGDLWorldModel}(
        wm,
        goals,
        4,
        4,
        3,
        5,
        nothing,
        Goal[]
    )

    agent = GenAgent(wm, tm, q, p)
    agent_idx = 1 # by convention
    gym = SoloGym(wm.imap, wm.tvec, init_state, agent,
                  agent_idx)

    run_gym!(gym)


end

test();
