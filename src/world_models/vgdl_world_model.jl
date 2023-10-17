struct VGDLWorldModel <: WorldModel
    game::VGDL.Game
    agent_idx::Int64
    imap::InteractionMap
    tvec::Vector{TerminationRule}
    graphics::GraphicsModule
end

struct VGDLWorldState <: WorldState{VGDLWorldModel}
    gstate::VGDL.GameState
end

@gen function vgdl_agent(ai::Int64, prev::VGDLWorldState, wm::VGDLWorldModel)
    agent = prev.gstate.agents[ai]
    # for now, pick a random action
    actions = actionspace(agent)
    a_ws = Fill(1.0 / length(actions))
    action_index = @trace(categorical(a_ws), :action)
    action = actions[action_index]
    return action
end

@gen function vgdl_dynamics(t::Int, prev::VGDLWorldState, wm::VGDLWorldModel)
    # non-self actions
    nagents = length(prev.agents)
    prevs = Fill(prev, nagents)
    wms = Fill(wm, nagents)
    actions = @trace(vgdl_agent_kernel(prevs, wms), :agents)
    queues = aggregate_actions(wm, actions) # TODO
    # deterministic forward step
    # REVIEW: consider incremental computation?
    new_gstate = VGDL.update_step(prev.gstate, queues, wm.imap)
    new_state = VGDLWorldState(new_gstate) # package
end

@gen function vgdl_perceive(t::Int, prev::VGDLWorldState, wm::VGDLWorldModel)
    new_state = @trace(vgdl_dynamics(t, prev, wm), :dynamics)
    # predict observation
    pred_mu, pred_var = render_prediction(wm.graphics, new_gstate)
    obs = @trace(broadcasted_normal(pred_mu, pred_var), :observe)
    return new_state
end

@gen function vgdl_world_init()

end
