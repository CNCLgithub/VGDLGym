using Gen
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

    goals = [
        Goal(AllRef{Butterfly}(), Get()),
        Goal(AllRef{Pinecone}(), Count())
    ]

    info = Info(init_state, 1)
    sgs = reduce(vcat, map(g -> decompose(g, info), goals))


    @time tr = replan!(wm, ws, sgs)

    # display(get_choices(tr))


end

test();
