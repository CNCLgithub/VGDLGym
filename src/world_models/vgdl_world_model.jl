struct VGDLWorldModel <: WorldModel
    game::VGDL.Game
    agent_idx::Int64
end

struct VGDLWorldState <: WorldState{VGDLWorldModel}
    gstate::GameState
end

@gen function vgdl_world_kernel(t::Int, prev::VGDLWorldState, a::Action, wm::VGDLWorldModel)
    # non-self actions
    other_actions = @trace(vgdl_agent_kernel(prev, wm), :agents)
    queues = aggregate_actions(a, other_actions)
    # deterministic forward step
    # REVIEW: consider incremental computation?
    new_gstate = VGDL.update_step(prev.gstate, queues)
    # predict observation
    obs = @trace(vgdl_observe(new_gstate), :observe)
    new_state = VGDLWorldState(new_gstate) # package
    return new_state
end
