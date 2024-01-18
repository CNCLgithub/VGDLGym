using Gen
using VGDL
using VGDLGym
using VGDLGym: render_world_state, render_obs, action_step,
    plan!, perceive!
using ImageIO
using FileIO
using BenchmarkTools
using Profile
using StatProfilerHTML


function render_gym!(gym)
    agent_idx = gym.agent_idx
    path = "test/output"
    isdir(path) || mkpath(path)
    while !isfinished(gym.state, gym.tvec)

        # plan next action
        action = plan!(gym.agent)

        # receive agent's action and generate actions for NPCs
        queues =
            action_step(gym.state, Dict(agent_idx => action))
        # resolve interactions in game state
        next_state  = update_step(gym.state, gym.imap, queues)

        # update agent's percepts
        # also pass the planned action
        obs =
            perceive!(gym.agent, next_state, action)

        obs_img = render_obs(obs)
        state_img = render_obs(render_world_state(gym.agent.perception))

        @show size(obs_img)
        @show size(state_img)
        save("$(path)/gt_$(gym.state.time).png",
             repeat(obs_img, inner = (10,10)))
        save("$(path)/state_$(gym.state.time).png",
             repeat(state_img, inner = (10,10)))

        # update reference to new game state
        gym.state = next_state
    end
    return nothing
end

function test()
    G = ButterflyGame
    imap = compile_interaction_set(G)
    tset = termination_set(G)
    init_state = load_level(G, 3)
    # limit time
    init_state.max_time = 500

    @assert typeof(init_state.scene.dynamic[1]) <: Player

    noise = 0.1 # noise for graphics

    tiles = [VGDL.Ground, VGDL.Pinecone]
    agents = [VGDL.Butterfly]
    wm, ws = init_world_model(VGDLWorldModel,
                              G,
                              PixelGraphics,
                              init_state,
                              tiles,
                              agents,
                              noise)

    args = (100, ws, wm)
    tr = Gen.simulate(vgdl_wm_perceive, args)

    # bm = @benchmark Gen.simulate(vgdl_wm_perceive, $args)
    # display(bm)

    # @profilehtml Gen.simulate(vgdl_wm_perceive, args)
    # Profile.init(;n=Int64(1E6), delay = 1E-7)
    # Profile.clear()
    # @profilehtml Gen.simulate(vgdl_wm_perceive, args)

    ws = VGDLGym.loci_weights(tr)
    # bm = @benchmark $(VGDLGym.loci_weights)($tr)
    # display(bm)

end

tr = test();
