using VGDL
using VGDLGym


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

    tm = map_transfer

    proc_args = (100, 50, uniform_attention)
    q = IncPerceptionModule(vgdl_wm_perceive,
                            wm,
                            ws,
                            proc_args)

    p = GreedyPlanner{VGDLWorldModel}()

    agent = GenAgent(wm, tm, q, p)
    agent_idx = 1 # by convention
    gym = SoloGym(wm.imap, wm.tvec, init_state, agent,
                  agent_idx)

    run_gym!(gym)


end

test();
