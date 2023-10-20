using VGDL
using VGDLGym


function test()
    G = ButterflyGame
    imap = compile_interaction_set(G)
    tset = termination_set(G)
    init_state = load_level(G, 1)

    graphics = PixelGraphics(G)
    noise = 0.1
    @assert typeof(init_state.scene.dynamic[1]) <: Player
    wm = VGDLWorldModel(1, # Player is the first agent
                        imap,
                        tset,
                        graphics,
                        noise)
    ws = VGDLWorldState(init_state)

    tm = map_transfer

    proc_args = (20, 10, no_attention)
    q = IncPerceptionModule(vgdl_wm_perceive,
                            wm,
                            VGDLWorldState(init_state),
                            proc_args)

    p = RandomPlanner{VGDLWorldModel}()

    agent = GenAgent(wm, tm, q, p)
    agent_idx = 1 # by convention
    gym = SoloGym(imap, tset, init_state, agent,
                  agent_idx)

    run_gym!(gym)


end

test();
