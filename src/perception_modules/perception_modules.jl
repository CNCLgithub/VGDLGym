export IncrementalQuery,
    increment

struct IncrementalQuery <: Gen_Compose.Query
    model::Gen.GenerativeFunction
    constraints::Gen.ChoiceMap
    args::Tuple
    argdiffs::Tuple
    step::Int
end

function increment(q::IncrementalQuery, cm::Gen.ChoiceMap,
                   new_args::Tuple)
    # TODO: proceduralize with `argdiff`
    _, zs... = q.args
    args = (first(new_args), zs...)
    setproperties(q;
                  step = q.step + 1,
                  constraints = cm,
                  args = args)
end

include("gt_perception.jl")
include("pf_module.jl")
