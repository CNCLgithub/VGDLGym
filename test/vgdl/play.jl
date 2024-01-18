using VGDL
using VGDLGym
using VGDLGym: render_world_state, render_obs, action_step,
    plan!, perceive!
using ImageIO
using FileIO


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
    init_state.max_time = 20

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

    tm = map_transfer

    proc_args = (;particles = 100, attention = uniform_attention)
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

    # render_gym!(gym)
    run_gym!(gym)


end

test();
