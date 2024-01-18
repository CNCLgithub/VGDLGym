#################################################################################
# exports
#################################################################################

export VGDLWorldModel,
    VGDLWorldState,
    vgdl_wm,
    vgdl_wm_perceive,
    init_world_model,
    affordances # TODO: general method

#################################################################################
# World model specification
#################################################################################

struct VGDLWorldModel <: WorldModel
    agent_idx::Int64
    imap::InteractionMap
    tvec::Vector{TerminationRule}
    # Magic params
    # TODO: figure out types
    tile_types::Vector
    birth_types::Vector

    "Graphics parameters"
    graphics::Graphics
    "Pixel noise"
    noise::Float64
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
                          tiles,
                          agents,
                          noise::Float64
                          ) where {G<:Game,
                                   W<:VGDLWorldModel,
                                   U<:Graphics}
    imap = compile_interaction_set(G)
    tset = termination_set(G)
    graphics = U(G)

    agent = init_state.scene.dynamic[1]

    wm = VGDLWorldModel(1, # Player is the first agent
                        imap,
                        tset,
                        tiles,
                        agents,
                        graphics,
                        noise)
    ws = VGDLWorldState(init_state)
    wm, ws
end

function Info(wm::VGDLWorldModel, ws::VGDLWorldState)
    gs = game_state(ws)
    d = affordances(gs)
    Info(gs, d)
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

function extract_ws(wm::VGDLWorldModel, tr::Gen.Trace)
    (t, ws, _) = get_args(tr)
    t == 0 ? ws : last(get_retval(tr))
end

#################################################################################
# Generative model
#################################################################################

struct OrphanBirth{T<:VGDL.DynamicElement} <: VGDL.Rule{BirthEffect, Single}
    args::NamedTuple
end
# id - since not referring to any parent
VGDL.lens(r::OrphanBirth) = x -> x
# discards input - generating new object
VGDL.transform(r::OrphanBirth{T}) where {T} = x -> T(;r.args...)
# REVIEW: need promise?

@gen (static) function no_magic(s::VGDLWorldState, wm::VGDLWorldModel,
                                loci::Int64)
    r::VGDL.Rule = VGDL.no_action
    return r
end

# REVIEW: avoid retiling to same type?
function retile_weights(s::VGDLWorldState, wm::VGDLWorldModel)
    n = length(wm.tile_types)
    Fill(1.0 / n, n)
end

@gen (static) function vgdl_retile(s::VGDLWorldState, wm::VGDLWorldModel,
                                   loci::Int64)
    ws = retile_weights(s, wm)
    tidx = @trace(categorical(ws), :tile_type)
    T = wm.tile_types[tidx]
    ci = loci_to_coord(wm, s, loci)
    r::VGDL.Rule = Retile{T}(nothing, ci)
    return r
end

function birth_weights(s::VGDLWorldState, wm::VGDLWorldModel)
    # what kinds of object to birth
    n = length(wm.birth_types)
    Fill(1.0 / n, n)
end

function loci_to_coord(wm::VGDLWorldModel, s::VGDLWorldState, loci::Int64)
    CartesianIndices(game_state(s).scene.static)[loci]
end

function birth_at(s::VGDLWorldState, wm::VGDLWorldModel, did::Int64, lid::Int64)
    T = wm.birth_types[did]
    ci = loci_to_coord(wm, s, lid)
    pos = SVector{2, Int64}(ci.I)
    OrphanBirth{T}((; position=pos))
end

@gen (static) function vgdl_birth(s::VGDLWorldState, wm::VGDLWorldModel,
                                  loci::Int64)
    obj_ws = birth_weights(s, wm)
    type_id = @trace(categorical(obj_ws), :object_type)
    r::VGDL.Rule = birth_at(s, wm, type_id, loci)
    return r
end

struct Dies <: VGDL.Rule{DeathEffect, Single}
    idx::Int64
    ref
    function Dies(idx::Int64)
        new(idx, VGDL.get_agent(idx))
    end
end
VGDL.lens(r::Dies) = r.ref
VGDL.transform(r::Dies) = x -> x
# REVIEW: need promise?
VGDL.priority(::Dies) = 0
# used in `VGDL.resolve`
function VGDL.pushtoqueue!(r::Dies, ::Dict, ::Dict, d::Dict)
    lr = lens(r)
    # contingent on the target is already dead
    haskey(d, lr) && return false
    tr = transform(r)
    d[lr] = tr
    return true
end


@gen (static) function vgdl_death(s::VGDLWorldState, wm::VGDLWorldModel,
                                  loci::Int64)
    r::VGDL.Rule = death_at(wm, s, loci)
    return r
end

const vgdl_magic_switch = Gen.Switch(no_magic, vgdl_birth, vgdl_death, vgdl_retile)

function loci_weights(s::VGDLWorldState, wm::VGDLWorldModel)
    scene = game_state(s).scene
    dx, dy = scene.bounds
    loc_weights = Vector{Float64}(undef, dx * dy)
    @inbounds for i = 1:length(scene.static)
       # HACK: assuming `Butterfly` game
       # TODO: implement `isnavigable`
       loc_weights[i] = Int64(scene.static[i] != VGDL.obstacle)
    end
    rmul!(loc_weights, 1.0 / sum(loc_weights))
    return loc_weights
end

function magic_weights(s::VGDLWorldState, wm::VGDLWorldModel, loci::Int64)
    # none, birth, death, retile
    ws = zeros(4)
    # TODO: implement `has_agent`
    if has_agent(s, wm, loci)
        # none | death
        ws[1] = ws[3] = 0.5
    else
        # none | birth | retile
        ws[1] = ws[2] = ws[4] = 1.0 / 3.0
    end
    return ws
end

@gen (static) function vgdl_magic(s::VGDLWorldState, wm::VGDLWorldModel)
    # something magical may happen at a location
    lws = loci_weights(s, wm)
    loci = @trace(categorical(lws), :loci)
    # what kind of magic?
    mws = magic_weights(s, wm, loci)
    midx = @trace(categorical(mws), :magic_idx)
    # apply magic
    spell = @trace(vgdl_magic_switch(midx, s, wm, loci), :spell)
    # resolving interactions
    result::GameState = VGDL.resolve(s.gstate, spell)
    new_state::VGDLWorldState = VGDLWorldState(result)
    return new_state
end

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
    queues = action_step(prev.gstate, Dict(zip(agent_keys, actions)))
    # deterministic forward step
    # REVIEW: consider incremental computation?
    new_gstate = VGDL.update_step(prev.gstate, wm.imap, queues)
    new_state::VGDLWorldState = VGDLWorldState(new_gstate) # package
    return new_state
end

@gen (static) function vgdl_render(mus::SVector{3, Float64},
                                   sd::Float64)
    var = Fill(sd, 3)
    pixel::SVector{3, Float64} = @trace(broadcasted_normal(mus, var), :pixel)
    return pixel
end

const vgdl_render_map = Gen.Map(vgdl_render)

@gen (static) function vgdl_obs(t::Int,
                                prev::VGDLWorldState,
                                wm::VGDLWorldModel)
    # birth/death
    temp_state = @trace(vgdl_magic(prev, wm), :birthdeath)
    new_state = @trace(vgdl_dynamics(t, temp_state, wm), :dynamics)
    # predict observation
    pred_mu = predict(wm.graphics, new_state.gstate)
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

#################################################################################
# proposals
#################################################################################

const VGDLObs = Gen.get_trace_type(vgdl_obs)
const VGDLPerceive = Gen.get_trace_type(vgdl_wm_perceive)



@gen (static) function propose_magic(tr::VGDLPerceive)
    # sample region to change
    ws = loci_weights(tr)
    loci::Int64 = @trace(categorical(ws), :loci)
    return loci
end

@transform magic_involution (model_in, aux_in) to (model_out, aux_out) begin
    t = first(get_args(model_in))
    addr = :kernel => t => :magic => :loci
    # forward
    @copy(aux_in[:loci], model_out[addr])
    # backward
    @copy(model_in[addr], aux_out[:loci])
end

Gen.is_involution!(magic_involution)

function select_random_agent(tr::T) where {T<:VGDLPerceive}
    t, _... = get_args(tr)
    # choices = get_choices(tr)
    sub_addr = :kernel => t => :dynamics => :agents
    #NOTE: retval may return newly spawned agents
    nagents = length(tr[sub_addr]) - 1 # not counting player
    selected = categorical(Fill(1.0 / nagents, nagents))
    ai = selected + 1 # from 1 -> N (because of Gen.Map)
    # display(choicemap(get_submap(get_choices(tr), :kernel => t => :dynamics => :agents)))
    addr = :kernel => t => :dynamics => :agents =>
        ai => :action
    Gen.select(addr)
end

magic_translator = SymmetricTraceTranslator(propose_magic,
                                            (),
                                            magic_involution)

function perception_mcmc_kernel(tr::T,
                                magic_steps::Int64,
                                agent_steps::Int64) where {T<:VGDLPerceive}
    new_tr::T = tr
    w::Float64 = 0.0

    t = first(get_args(tr))
    # Magic block
    s = select(
        :kernel => t => :magic => :magic_idx,
        :kernel => t => :magic => :spell,
    )
    for _ = 1:magic_steps
        _tr, d = magic_translator(new_tr, check=false)
        _tr, _d = move_reweight(_tr, s)
        d += _d
        if rand() < d
            new_tr = _tr
            w += d
        end
    end

    # Agent block
    for _ = 1:agent_steps
        s = select_random_agent(new_tr)
        _tr, d = move_reweight(new_tr, s)
        if rand() < d
            new_tr = _tr
            w += d
        end
    end

    return (new_tr, w)
end

#################################################################################
# helpers
#################################################################################

function predict(gr::VGDL.PixelGraphics, state::GameState)
    img = render(gr, state)
    # c, a, b = size(img)
    # mus = Matrix{SVector{3, Float64}}(undef, a, b)
    # @inbounds for x = 1:a, y = 1:b
    #     mus[x, y] = img[:, x, y]
    # end
    # return mus
end

function death_at(wm::VGDLWorldModel, s::VGDLWorldState, loci::Int64)
    cind = loci_to_coord(wm, s, loci)
    pos = SVector{2, Int64}(cind.I)
    tree = s.gstate.scene.kdtree
    idxs = inrange(tree, pos, 0)
    isempty(idxs) && return VGDL.no_action
    idx = first(idxs)
    # don't kill the player
    player = s.gstate.scene.dynamic[1]
    pos == VGDL.position(player) &&  return VGDL.no_action
    Dies(idx)
end

function has_agent(s::VGDLWorldState, wm::VGDLWorldModel, loci::Int64)
    cind = loci_to_coord(wm, s, loci)
    pos = SVector{2, Int64}(cind.I)
    tree = s.gstate.scene.kdtree
    idxs = inrange(tree, pos, 0)
    !isempty(idxs)
end


function loci_weights(tr::T) where {T<:VGDLPerceive}
    t = first(get_args(tr))
    # top-level ir
    top_ir = Gen.get_ir(vgdl_wm_perceive)
    kernel_node = top_ir.call_nodes[1] # :kernel
    ktrace = getproperty(tr, Gen.get_subtrace_fieldname(kernel_node))

    obs_ir = Gen.get_ir(vgdl_obs)
    obs_trace = ktrace.subtraces[t] # :kernel => t
    obs_choice_node = obs_ir.choice_nodes[1] # :observe

    # args
    pred_mu_node = obs_choice_node.inputs[1] # :pred_mu
    pred_mu = getproperty(obs_trace,
                          Gen.get_value_fieldname(pred_mu_node))
    pred_sd_node = obs_choice_node.inputs[2] # :pred_var
    pred_sd = getproperty(obs_trace,
                          Gen.get_value_fieldname(pred_sd_node))

    # image
    img = getproperty(obs_trace, Gen.get_value_fieldname(obs_choice_node))

    nc, nx, ny = size(img)
    ws = Matrix{Float64}(undef, nx, ny)
    @inbounds for x = 1:nx, y=1:ny
        pixel_ls = 0.0
        for c = 1:nc
            pixel_ls += Gen.logpdf(normal, img[c,x,y],
                                   pred_mu[c,x,y], pred_sd[c,x,y])
        end
        ws[x,y] = pixel_ls
    end
    return softmax(vec(ws))
end
