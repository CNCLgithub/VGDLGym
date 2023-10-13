using VGDL
using VGDLGym


function test()
    g = ButterflyGame
    obs_noise = 0.1
    st::GameState = load_level(g, 1)
    wm = VGDLWorldModel(g,
                        1,
                        PixelGraphics(obs_noise, g))
    ws = VGDLWorldState(ws)


end

test();
