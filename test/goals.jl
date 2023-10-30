using VGDL
using VGDLGym


function test()
    G = ButterflyGame
    imap = compile_interaction_set(G)
    tset = termination_set(G)
    init_state = load_level(G, 1)
    info = Info(init_state, 1)

    g = Goal(AllRef{Butterfly}(), Get())
    @time sgs = decompose(g, info)
    @time for sg = sgs
        evaluate(sg, info)
    end
    @time for sg = sgs
        gradient(sg, info)
    end


    @time g2 = Goal(AllRef{Pinecone}(), Count())
    @time sgs2 = decompose(g2, info)
    @time for sg = sgs2
        evaluate(sg, info)
    end
    @time for sg = sgs2
        gradient(sg, info)
    end





end

test();
