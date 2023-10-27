using VGDL
using VGDLGym


function test()
    G = ButterflyGame
    imap = compile_interaction_set(G)
    tset = termination_set(G)
    init_state = load_level(G, 1)
    # limit time
    init_state.max_time = 500

    graphics = PixelGraphics(G)
    noise = 0.2
    @assert typeof(init_state.scene.dynamic[1]) <: Player
    wm = VGDLWorldModel(1, # Player is the first agent
                        imap,
                        tset,
                        graphics,
                        noise)
    ws = VGDLWorldState(init_state)

    p = TheoryBasedPlanner(wm, ws)


    agent = GenAgent(wm, tm, q, p)
    agent_idx = 1 # by convention
    gym = SoloGym(imap, tset, init_state, agent,
                  agent_idx)

    run_gym!(gym)


end

test();
