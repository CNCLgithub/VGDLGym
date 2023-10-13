
function run_gym!(agent::GenAgent,
                  g::Game,
                  state::GameState)
    imap = compile_interaction_set(g)
    tset = termination_set(g)
    action = VGDL.no_action # REVIEW: agent doesn't act at t=1
    while !isfinished(state, tset)
        # perceive and plan next action
        action = process!(agent, state, action)
        # receive agent's action and generate actions for NPCs
        queues = action_step(g, state, [agent => action])
        # resolve interactions in game state
        state  = update_step(state, imap, queues)
    end
end
