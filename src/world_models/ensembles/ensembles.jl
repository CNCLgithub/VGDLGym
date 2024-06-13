#################################################################################
# exports
#################################################################################

export EnsembleWorldModel,
    EnsembleWorldState,
    ens_wm,
    ens_wm_perceive,
    init_world_model,
    affordances # TODO: general method

using VGDL: DynamicElement, StaticElement
#################################################################################
# World model specification
#################################################################################

struct EnsembleWorldModel <: WorldModel
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

struct EnsembleWorldState <: WorldState{EnsembleWorldModel}
    static::VGDL.GridScene
    individuals::Vector{DynamicElement}
    ensembles::Vector{PoissonPointProcess}
end
