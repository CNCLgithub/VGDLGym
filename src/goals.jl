export Goal,
    RefType,
    retrieve,
    satisfied,
    SingularRef,
    BroadcastRef,
    AllRef,
    Morphism,
    Injective,
    Surjective,
    Strategy,
    Get,
    Count,
    decompose,
    evaluate,
    gradient,
    WorldMap,
    GoalGradients

using AccessorsExtra: maybe, MaybeOptic

abstract type RefType end

function retrieve end

function satisfied end

const LensUnion = Union{ComposedFunction,
                        IndexLens,
                        PropertyLens,
                        MaybeOptic}

struct SingularRef{X} <: RefType
    l::LensUnion
end

retrieve(r::SingularRef, info) = r.l(info)

abstract type BroadcastRef <: RefType end

struct AllRef{X} <: BroadcastRef end

function broadcast_ref(::AllRef{X}, info) where {X}
    ls = retreive_refs(X, info)
    SingularRef{X}.(ls)
end

function retrieve(r::AllRef{X}, info) where {X}
    ls = broadcast_ref(r, info)
    map(l -> retrieve(l, info), ls)
end

abstract type Morphism end
abstract type Injective <: Morphism end
abstract type Surjective <: Morphism end

abstract type Strategy{X<:Morphism} end

struct Get <: Strategy{Injective} end

struct Count <: Strategy{Surjective} end

struct Goal{R<:RefType, S<:Strategy}
    rt::R
    s::S
end

reference(g::Goal) = g.rt
strategy(g::Goal) = g.s

decompose(g::Goal{R,S}, info) where {R, S} = [g]

function decompose(g::Goal{R,S}, info
                   ) where {R<:BroadcastRef, S<:Strategy{<:Injective}}
    refs = broadcast_ref(reference(g), info)
    map(r -> Goal(r, strategy(g)), refs)
end

#################################################################################
# WorldMap
#################################################################################

struct WorldMap{W<:WorldModel}
    wm::WorldModel
    ws::WorldState{W}
    distances
end

#################################################################################
# Get
#################################################################################

function evaluate(g::Goal{R,S}, info) where {R<:SingularRef, S<:Get}
    iszero(gradient(g, info))
end

function gradient(g::Goal{R,S}, info) where {R<:SingularRef, S<:Get}
    d = distance_to(info, reference(g))
    -d
end

# function evaluate(g::Goal{R,S}, info) where {R<:AllRef, S<:Get}
#     sgs = decompose(g, info)
#     gradient(g, info) != 0
# end

# function gradient(g::Goal{R,S}, info) where {R<:AllRef, S<:Get}
#     sgs = decompose(g, info)
#     sum(map(gradient, sgs))
# end

#################################################################################
# Count
#################################################################################

function evaluate(g::Goal{R,S}, info) where {R<:AllRef, S<:Count}
    iszero(gradient(g, info))
end

function gradient(g::Goal{R,S}, info) where {R<:AllRef, S<:Count}
    n = length(broadcast_ref(reference(g), info))
    -log(1 / (n+1))
end

#################################################################################
# Calculating goal gradients
#################################################################################

struct GoalGradients
    subgoals::Vector{<:Goal}
    affordances::AbstractGraph
end

function (gg::GoalGradients)(tr::T) where {T<:Gen.Trace}
    wm = WorldMap(tr, gg.affordances)
    n = length(gg.subgoals)
    results = Vector{Float64}(undef, n)
    @inbounds for i = 1:n
        results[i] = gradient(gg.subgoals[i], wm)
    end
    results
end

#################################################################################
# Helpers
#################################################################################


function Accessors.IndexLens(i::CartesianIndex{2})
    IndexLens(Tuple(i))
end

function retreive_refs end


function get_vertex(ds::Tuple{Int64, Int64}, pos::SVector{2, Int64})
    y,x = pos
    dy,_ = ds
    (dy * (x -1)) + y
end

function map_position_to_graph end


function distance_to(info::WorldMap, r::SingularRef)
    el = retrieve(r, info)
    isnothing(el) && return 0.0
    v = map_position_to_graph(info, el)
    info.distances[v]
end
