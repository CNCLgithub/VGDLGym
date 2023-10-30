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
    Info

using AccessorsExtra: maybe, MaybeOptic
using VGDL: DynamicElement, StaticElement

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
# Info
#################################################################################

struct Info
    state::GameState
    distances
end

function Info(state::GameState, agent_idx::Int64)
    g = to_graph(state)
    agent = state.scene.dynamic[agent_idx]
    dy, dx = state.scene.bounds
    lpos = (agent.position[2] - 1) * dy + agent.position[2]
    gds = gdistances(g, lpos)
    Info(state, gds)
end

function to_graph(state::GameState)
    dy, _ = state.scene.bounds
    nv = length(state.scene.static)
    adj_matrix = fill(false, (nv, nv))
    for i = 1:(nv -1), j = (i+1):nv
        d = abs(j - i)
        !(d == 1 || d == dy) && continue
        adj_matrix[i, j] = adj_matrix[j, i ] = true
    end
    SimpleGraph(adj_matrix)
end

#################################################################################
# Get
#################################################################################

function evaluate(g::Goal{R,S}, info) where {R<:SingularRef, S<:Get}
    iszero(gradient)
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
# Helpers
#################################################################################

ref_prefix(::Type{X}) where {X<:DynamicElement} = (@optic _.state.scene.dynamic)
ref_prefix(::Type{X}) where {X<:StaticElement} = (@optic _.state.scene.static)

function Accessors.IndexLens(i::CartesianIndex{2})
    IndexLens(Tuple(i))
end

function retreive_refs(::Type{X}, info::Info) where {X}
    rpfx = ref_prefix(X)
    els = rpfx(info)
    # NOTE: works for vector and Dict
    map(i -> maybe(opcompose(rpfx, IndexLens(i))),
        findall(x -> isa(x, X), els))
end

function get_vertex(ds::Tuple{Int64, Int64}, pos::SVector{2, Int64})
    y,x = pos
    dy,_ = ds
    (dy * (x -1)) + y
end

function distance_to(info::Info, r::SingularRef{X}
                     ) where {X<:DynamicElement}
    el = retrieve(r, info)
    isnothing(el) && return 0.0
    v = get_vertex(info.state.scene.bounds, el.position)
    info.distances[v]
end
