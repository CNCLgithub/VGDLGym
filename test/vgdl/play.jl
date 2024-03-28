using VGDL
using VGDLGym
using VGDLGym: viz_obs, action_step,
    plan!, perceive!, viz_perception_module,
    viz_planning_module, render_obs
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


        println("Time $(gym.state.time)")
        println("\tWorld State")

        viz_obs(obs)
        save("$(path)/gt_$(gym.state.time).png",
             repeat(render_obs(obs), inner = (10,10)))
        println("\tAgent State")
        println("\t\tPercept")
        viz_perception_module(gym.agent.perception,
                              "$(path)/percept_$(gym.state.time).png")
        println("\t\tHorizon")
        viz_planning_module(gym.agent.planning,
                            "$(path)/horizon_$(gym.state.time).png")

        # update reference to new game state
        gym.state = next_state
    end
    return nothing
end

function test()
    G = ButterflyGame
    imap = compile_interaction_set(G)
    tset = termination_set(G)
    init_state = load_level(G, 2)
    # limit time
    init_state.max_time = 100

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


    proc_args = (;particles = 20)
    q = IncPerceptionModule(vgdl_wm_perceive,
                            wm,
                            ws,
                            proc_args)
    # q = GTPerceptionModule(wm, ws)

    tm = MAPMemory{VGDLWorldModel}()

    goals = [
        Goal(AllRef{Butterfly}(), Get()),
        # Goal(AllRef{Pinecone}(), Count())
    ]
    p = TheoryBasedPlanner{VGDLWorldModel}(
        wm,
        goals,
        5, # search steps
        10, # replan steps
        3, # percept update steps
        7, # extend plan steps
        nothing,
        Goal[],
    )

    att = FactorizedAttention(wm, ws)

    agent = GenAgent(wm, q, p, tm, att)
    agent_idx = 1 # by convention
    gym = SoloGym(wm.imap, wm.tvec, init_state, agent,
                  agent_idx)

    # render_gym!(gym)
    run_gym!(gym)


end

test();
