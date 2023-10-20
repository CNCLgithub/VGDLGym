export MAPTransfer,
    map_transfer

struct MAPTransfer <:TransferModule end
const map_transfer = MAPTransfer()

function transfer(m::MAPTransfer, w::W,
                  perception::PerceptionModule{W},
                  planning::PlanningModule{W}
                  ) where {W<:WorldModel}
    plan_in = transfer(m, perception) # MAP estimate
    plan_out = plan!(planning, w, plan_in)  # returns an action
end
