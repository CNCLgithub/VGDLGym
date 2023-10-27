abstract type Goal end
abstract type RefType end
abstract type Strategy end


function reference end

function satisfied end

struct SingularRef{X} <: RefType
    x::X
end

reference(r::SingularRef) = r.x

abstract type BroadcastRef <: RefType end

struct AllRef{X} <: BroadcastRef end

reference(r::AllRef{X}) where {X} =

function broadcast_ref(::AllRef{X}, info) where {X}
    els = retreive_all(X, info)
    SingularRef.(els)
end

struct Get <: Strategy end

struct Count <: Strategy end

struct Goal{R<:RefType, S<:Strategy}
    rt::R
    s::S
end

reference(g::Goal) = reference(g.rt)
strategy(g::Goal) = g.s

decompose(g::Goal{R,S}, ::WorldState) where {R<:SingularRef, S} = [g]
function decompose(g::Goal{R,S}, ws::GameState) where {R<:BroadcastRef, S}
    refs = broadcast_ref(g.rt, ws)
    map(r -> Goal(r, strategy(g)), refs)
end

function evaluate(g::Goal{R,S}, info) where {R<:SingularRef, S<:Get}
    distance_to(info, reference(g)) == 0
end

function gradient(g::Goal{R,S}, info) where {R<:SingularRef, S<:Get}
    distance_to(info, reference(g))
end

function evaluate(g::Goal{R,S}, info) where {R<:AllRef{X}, S<:Count, X}
    length(retrieve_all(X, info)) > 0
end

function gradient(g::Goal{R,S}, info) where {R<:SingularRef, S<:Get}
    distance_to(info, reference(g))
end
