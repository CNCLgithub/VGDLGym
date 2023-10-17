
"""
    render_prediction(g::GraphicsModule, ws::WorldState)

Parameterizes a multi-variate distribution in image space.
Returns a tuple `(mu, var)`.
"""
function render_prediction end

const F3V = SVector{3, Float64}

struct PixelGraphics <: GraphicsModule
    noise::Float64
    color_map::Dict{Type{VGDL.Element}, F3V}
end

function PixelGraphics(g::VGDL.Game, noise::Float64)
    PixelGraphics(nosie, default_colormap(g))
end

"""
    default_colormap(g::VGDL.Game)

Default colormap for a game.
"""
function default_colormap end

default_colormap(::VGDL.ButterflyGame) =
    Dict{Type{VGDL.Element}, SVector{3, Float64}}(
        VGDL.Ground => F3V(0.8, 0.8, 0.8),
        VGDL.Obstacle => F3V(0., 0., 0.),
        VGDL.PineCone => F3V(0., 0.8, 0.0),
        VGDL.Butterfly => F3V(0.9, 0.7, 0.2),
        VGDL.Player => F3V(0., 0., 0.9),
        GenAgent => F3V(0., 0., 0.9),
    )

function render(g::PixelGraphics, gs::VGDL.GameState)
    nx,ny = gs.bounds
    m = Array{Float64}(undef, (3, nx, ny))

    # write agents
    for i in eachindex(gs.agents)
        agent = gs.agents[i]
        x, y = agent.position
        m[:, x, y] = g.color_map[agent]
    end
    # write static elements
    for i in eachindex(gs.items)
        x,y = i.I # REVIEW: assumes `i` is `CartesianIndex{2}`
        mv = @view m[:, x, y]
        # don't overwrite agents
        isdefined(mv, 1) && continue
        T = typeof(gs.items[i])
        mv[:] = g.color_map[T]
    end
    return m
end

function render_prediction(g::PixelGraphics, gs::VGDL.GameState)
    m = render(g, gs)
    nx,ny = gs.bounds
    sds = Fill(g.noise, (3, nx, ny))
    (m, sds)
end

# TODO
struct SpriteGraphics <: GraphicsModule end
