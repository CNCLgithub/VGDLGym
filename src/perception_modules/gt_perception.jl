export GTPerceptionModule


mutable struct GTPerceptionModule{T} <: PerceptionModule{T}
    wm::T
    state::WorldState{T}
end


function perceive!(pm::GTPerceptionModule{W},
                   am::AttentionModule,
                   wm::W,
                   st::GameState,
                   action::Int64) where {W<:WorldModel}
    gr = graphics(wm)
    obs = render(gr, st)
    pm.state = WorldState(W)(st)
    return obs
end

function transfer(::MAPMemory, perception::GTPerceptionModule)
    @unpack state, wm = perception
    args = (0, state, wm)
    # model is fixed - only vgdl is valid here
    Gen.simulate(vgdl_wm, (args))
end

function render_world_state(pm::GTPerceptionModule)
    gr = graphics(pm.wm)
    render(gr, pm.state)
end

function viz_perception_module(pm::GTPerceptionModule)
    img = render_world_state(pm)
    viz_obs(img)
    return nothing
end
