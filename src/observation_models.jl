struct GridObs <: VGDL.Observation
    data::Array{Float64, 3}
end

struct GridObsModel <: ObservationModel
    color_map::Dict #TODO
end

obs_type(Type{GridObsModel}) = GridObs


function render(om::GridObsModel, st::GameState)
    #TODO
    # something like: https://github.com/YoyoZhang24/VGDL/blob/main/src/scene.jl#L69-L90
end
