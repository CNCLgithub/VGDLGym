#################################################################################
# exports
#################################################################################

export VGDLWorldModel,
    VGDLWorldState,
    vgdl_wm,
    vgdl_wm_perceive,
    init_world_model

#################################################################################
# World model specification
#################################################################################

struct VGDLWorldModel <: WorldModel
    agent_idx::Int64
    imap::InteractionMap
    tvec::Vector{TerminationRule}
    graphics::Graphics
    noise::Float64
    nactions::Int64
end

graphics(wm::VGDLWorldModel) = wm.graphics

struct VGDLWorldState <: WorldState{VGDLWorldModel}
    gstate::VGDL.GameState
end

game_state(ws::VGDLWorldState) = ws.gstate

function init_world_model(::Type{W},
                          ::Type{G},
                          ::Type{U},
                          init_state::GameState,
                          noise::Float64
                          ) where {G<:Game,
                                   W<:VGDLWorldModel,
                                   U<:Graphics}
    imap = compile_interaction_set(G)
    tset = termination_set(G)
    graphics = U(G)

    agent = init_state.scene.dynamic[1]
    nactions = length(actionspace(agent))

    wm = VGDLWorldModel(1, # Player is the first agent
                        imap,
                        tset,
                        graphics,
                        noise,
                        nactions)
    ws = VGDLWorldState(init_state)
    wm, ws
end

function Info(wm::VGDLWorldModel, ws::VGDLWorldState)
    gs = game_state(ws)
    d = affordances(gs)
    Info(gs)
end


function affordances(state::GameState)
    dy, _ = state.scene.bounds
    tiles = state.scene.static
    nv = length(state.scene.static)
    adj_matrix = fill(false, (nv, nv))
    @inbounds for i = 1:(nv -1)
        for j = [i - 1, i + 1, i - dy, i + dy]
            ((j >= 1 && j <= nv) && (tiles[i] == tiles[j])) ||
                continue
            adj_matrix[i, j] = adj_matrix[j, i ] = true
        end
    end
    SimpleGraph(adj_matrix)
end

#################################################################################
# Generative model
#################################################################################

@gen (static) function vgdl_agent(ai::Int64,
                                  prev::VGDLWorldState,
                                  wm::VGDLWorldModel)
    scene = prev.gstate.scene
    agent = scene.dynamic[ai]
    # for now, pick a random action
    actions = actionspace(agent)
    nactions = length(actions)
    a_ws = Fill(1.0 / nactions, nactions)
    action_index::Int64 = @trace(categorical(a_ws), :action)
    return action_index
end

@gen (static) function vgdl_dynamics(t::Int,
                                     prev::VGDLWorldState,
                                     wm::VGDLWorldModel)
    agent_keys = prev.gstate.scene.dynamic.keys
    nagents = length(agent_keys)
    prevs = Fill(prev, nagents)
    wms = Fill(wm, nagents)
    actions = @trace(Gen.Map(vgdl_agent)(agent_keys, prevs, wms),
                     :agents)
    queues = action_step(prev.gstate,
                         Dict(zip(agent_keys, actions)))
    # deterministic forward step
    # REVIEW: consider incremental computation?
    new_gstate = VGDL.update_step(prev.gstate, wm.imap, queues)
    new_state::VGDLWorldState = VGDLWorldState(new_gstate) # package
    return new_state
end

@gen (static) function vgdl_obs(t::Int,
                                prev::VGDLWorldState,
                                wm::VGDLWorldModel)
    new_state = @trace(vgdl_dynamics(t, prev, wm), :dynamics)
    # predict observation
    pred_mu = render(wm.graphics, new_state.gstate)
    pred_var = Fill(wm.noise, size(pred_mu))
    obs = @trace(broadcasted_normal(pred_mu, pred_var), :observe)
    return new_state
end

@gen function vgdl_wm(t::Int, init::VGDLWorldState, wm::VGDLWorldModel)
    # TODO: prior
    states = @trace(Gen.Unfold(vgdl_dynamics)(t, init, wm),
                    :kernel)
    return states
end

@gen (static) function vgdl_wm_perceive(t::Int,
                                        init::VGDLWorldState,
                                        wm::VGDLWorldModel)
    # TODO: prior
    states = @trace(Gen.Unfold(vgdl_obs)(t, init, wm),
                    :kernel)
    return states
end
