#################################################################################
# exports
#################################################################################

export PoissonPointProcess

struct PoissonPointProcess{T} <: Ensemble{T}
    rate::Float64
    mu::SVector{2, Float64}
    var::Float64
end

function predict!(img, ppp::PoissonPointProcess, state::GameState)

end
