struct VGDLWorldModel <: WorldModel
    game::VGDL.Game
    agent_idx::Int64
    graphics::GraphicsModule
end

struct VGDLWorldState <: WorldState{VGDLWorldModel}
    gstate::GameState
end

@gen function vgdl_agent_kernel(ai::Int64, prev::VGDLWorldState, wm::VGDLWorldModel)
    agent = prev.gstate.agents[ai]
    # for now, pick a random action
    actions = actionspace(agent)
    a_ws = Fill(1.0 / length(actions))
    action_index = @trace(categorical(a_ws), :planning)
    action = actions[action_index]
    return action
end

@gen function vgdl_world_kernel(t::Int, prev::VGDLWorldState, a::Action, wm::VGDLWorldModel)
    # non-self actions
    others, prevs, wms = agent_kernel_args(prev, wm)
    other_actions = @trace(vgdl_agent_kernel(prev, wm), :agents)
    queues = aggregate_actions(wm, a, other_actions) # TODO
    # deterministic forward step
    # REVIEW: consider incremental computation?
    new_gstate = VGDL.update_step(prev.gstate, queues)
    # predict observation
    pred_mu, pred_var = render_prediction(wm.graphics, new_gstate)
    obs = @trace(broadcasted_normal(pred_mu, pred_var), :observe)
    new_state = VGDLWorldState(new_gstate) # package
    return new_state
end
