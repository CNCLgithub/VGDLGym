export MAPMemory

struct MAPMemory{W} <:MemoryModule{W} end


function transfer(m::M, w::W,
                  perception::V,
                  planning::P
                  ) where {W<:WorldModel,
                           V<:PerceptionModule{W},
                           P<:PlanningModule{W},
                           M<:MAPMemory{W}}
    plan_in = transfer(m, perception) # MAP estimate
    plan_out = plan!(planning, w, plan_in)  # returns an action
end
