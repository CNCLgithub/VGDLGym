

struct IncrementalQuery <: Query
    step::Int
    constraints::Gen.ChoiceMap
    args::Tuple
    argdiffs::Tuple
end

function increment(q::IncrementalQuery, cm::Gen.ChoiceMap,
                   new_args::Tuple)
    _, _, zs... = q.args
    setproperties(q;
                  step = q.step + 1,
                  constraints = cm,
                  args = (new_args..., zs...))
end

include("pf_module.jl")
