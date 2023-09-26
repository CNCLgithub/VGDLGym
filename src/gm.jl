@gen function vgdl_wm_init_state(wm::VGDLWorldModel)
end


@gen function vgdl_wm_kernel(t::Int, prev::VGDL.GameState, wm::VGDLWorldModel)
    next_st =

end

@gen function vgdl_world_model(t::Int, wm::VGDLWorldModel)
    initial_state = @trace(vgdl_wm_init_state(wm), :initial_state)
    steps = @trace(Gen.Unfold(vgdl_wm_kernel)(t, initial_state, wm), :steps)
    return steps
end
