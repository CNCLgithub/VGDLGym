abstract type GraphicsModule end

"""
    render_prediction(g::GraphicsModule, ws::WorldState)

Parameterizes a multi-variate distribution in image space.
Returns a tuple `(mu, var)`.
"""
function render_prediction end

struct PixelGraphics <: GraphicsModule
    noise::Float64
    color_map::Dict{Type{VGDL.Element}, SVector{3, Float64}}
end


function render_prediction(g::PixelGraphics, ws::VGDLWorldState)
    gs = ws.gstate

    nx,ny = TODO
    m = Array{Float64}(undef, (3, nx, ny))

    # write agents
    for i in eachindex(gs.agents)
        agent = gs.agents[i]
        x, y = agent.position
        m[:, x, y] = g.color_map[agent]
    end
    # write static elements
    for i in eachindex(gs.items)
        x,y = i.I # REVIEW
        mv = @view m[:, x, y]
        isdefined(mv, 1) && continue
        mv[:] = g.color_map[gs.items[i]]
    end

    sds = Fill(g.noise, (3, nx, ny))

    (m, sds)
end

# TODO
struct SpriteGraphics <: GraphicsModule end
