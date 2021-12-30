using Graphs, SimpleWeightedGraphs
using Plots, GraphRecipes
using AlphaZero
using GraphNeuralNetworks
import AlphaZero.GI


struct GameSpec <: GI.AbstractGameSpec
    gnnGraph
end

function randGraph(numVerticies::Int)
    graph = rand(Float32, numVerticies, numVerticies)
    foreach(enumerate(eachcol(graph))) do (idx, col)
        graph[idx, :] .= col
        graph[idx, idx] = 0
    end
    nodes = collect(1:numVerticies)
    sources = Vector(vec(repeat(nodes, 1, numVerticies - 1)'))
    targets = vcat(map(idx -> filter(val -> val != idx, nodes), nodes)...)
    weights = map(zip(sources, targets)) do (source, target)
        graph[source, target]
    end
    return GNNGraph(sources, targets, weights; ndata = (; x = ones(Float32, 1, numVerticies)))
end

# GameSpec() = GameSpec(randGraph(rand(collect(1:20))))
GameSpec() = GameSpec(randGraph(15))

mutable struct GameEnv <: GI.AbstractGameEnv
    gnnGraph
    maskedActions::Vector{Bool} # Masked actions and visitedVerticies can be derived from graph, but are included for clarity
    visitedVerticies::Vector{Int}
    finished::Bool
end
GI.spec(game::GameEnv) = GameSpec(game.gnnGraph)

function GI.init(spec::GameSpec)
    return GameEnv(spec.gnnGraph, trues(spec.gnnGraph.num_nodes), Vector{Int}(), false)
end

function GI.set_state!(game::GameEnv, state)
    game.gnnGraph = state.gnnGraph
    game.maskedActions = state.availableActions
    game.visitedVerticies = getPath(state.gnnGraph, game.visitedVerticies)
    any(game.maskedActions) || (game.finished = true)
    return
end

function getPath(gnnGraph, visitedVerticies)
    isempty(visitedVerticies) && return Vector{Int}()
    lastVertex = last(visitedVerticies)
    if isone(count(idx -> idx == lastVertex, gnnGraph.graph[1]))
        push!(visitedVerticies, gnnGraph.graph[2][findfirst(ind -> ind == lastVertex, gnnGraph.graph[1])])
    end
    # sources = gnnGraph.graph[1]
    # targets = gnnGraph.graph[2]
    # madeConnection(vertex) = isone(count(val -> val == vertex, sources))
    # isConnected(vertex) = isone(count(val -> val == vertex, targets))
    # startingVertex = filter(idx -> madeConnection(idx) && !isConnected(idx), unique(sources))
    # isnothing(startingVertex) && (startingVertex = 1)
    # path = [startingVertex]
    # foreach(path) do vertex
    #     if madeConnection(vertex)
    #         push!(path, targets[findfirst(idx -> idx == vertex, sources)])
    #     end
    # end
    return visitedVerticies
end

function Base.hash(gnn::GNNGraph, h::UInt64)
    hash(hash(gnn.graph[1]) + hash(gnn.graph[2]) + hash(gnn.graph[3]) + hash(gnn.ndata) + h)
end

GI.two_players(::GameSpec) = false
GI.actions(a::GameSpec) = collect(range(1, length = a.gnnGraph.num_nodes))
GI.clone(g::GameEnv) = GameEnv(g.gnnGraph, deepcopy(g.maskedActions), deepcopy(g.visitedVerticies), g.finished)
GI.white_playing(::GameEnv) = true
GI.game_terminated(g::GameEnv) = g.finished
GI.available_actions(g::GameEnv) = collect(range(1, length = g.gnnGraph.num_nodes))[g.maskedActions]
GI.actions_mask(game::GameEnv) = game.maskedActions

function GI.current_state(g::GameEnv)
    return (gnnGraph = g.gnnGraph, availableActions = g.maskedActions)
    # modifiedAdacencyList = deepcopy(g.graph[3])
    # if length(g.visitedVerticies) > 1
    #     foreach(enumerate(g.visitedVerticies[2:end])) do (idx, node)
    #         prevNode = state.path[idx]
    #         # Values are set to zero because SimpleWeightedGraphs discards edges with weights of zero.
    #         modifiedAdacencyList[prevNode, :] .= 0
    #         modifiedAdacencyList[:, node] .= 0
    #         modifiedAdacencyList[node, prevNode] = 0
    #         modifiedAdacencyList[prevNode, node] = spec.fadjlist[prevNode, node]
    #     end
    # end
    # return (ndata = (data = modifiedAdacencyList, x = ones(2)), availableActions = g.maskedActions)
end

function GI.play!(g::GameEnv, vertex::Int)
    adjMatrix = adjacency_matrix(g.gnnGraph)
    
    maskedActions = deepcopy(g.maskedActions)
    maskedActions[vertex] = false
    if isempty(g.visitedVerticies)
        sources = g.gnnGraph.graph[1]
        targets = g.gnnGraph.graph[2]
        weights = g.gnnGraph.graph[3]
    else
        edgeLength = deepcopy(adjMatrix[last(g.visitedVerticies), vertex])
        adjMatrix[:, vertex] .= 0
        adjMatrix[last(g.visitedVerticies),:] .= 0
        adjMatrix[last(g.visitedVerticies), vertex] = edgeLength

        sources = Vector{Int}()
        targets = Vector{Int}()
        weights = Vector{Float32}()
        foreach(enumerate(eachrow(adjMatrix))) do (i, col)
            foreach(enumerate(col)) do (j, val)
                if !iszero(val)
                    push!(sources, i)
                    push!(targets, j)
                    push!(weights, val)
                end
            end
        end
    end
    graph = GNNGraph(sources, targets, weights; ndata = (; x = ones(Float32, 1, g.gnnGraph.num_nodes)))
    state = (gnnGraph = graph, availableActions = maskedActions)

    push!(g.visitedVerticies, vertex)
    GI.set_state!(g, state)
    return
end

function GI.state_dim(game_spec::GameSpec)
    return size(game_spec.gnnGraph.ndata.x)[1]
end

function GI.white_reward(g::GameEnv)
    isempty(g.visitedVerticies[1:end-1]) && (return -Inf)
    # sources = g.gnnGraph.graph[1]
    # indicies = findall(idx -> idx ∈ g.visitedVerticies[1:end-1], sources)
    return -1 * sum(weights(g.gnnGraph))
end

function GI.heuristic_value(g::GameEnv)
    return GI.white_reward(g)
end

function GI.render(g::GameEnv)
    display(graphplot(adjacency_matrix(g.gnnGraph); curves = false))
end

function GI.graph_state(spec::GameSpec, state)
    return state.gnnGraph
end

function GI.action_string(::GameSpec, a)
    return string(a)
end

function GI.read_state(game::GameSpec)
    nodes = collect(Base.OneTo(game.gnnGraph.num_nodes))
    maskedActions = trues(game.gnnGraph.num_nodes)
    foreach(nodes) do node
        maskedActions[node] = count(vert -> vert == node, game.gnnGraph.graph[2]) == 1 ? false : true
    end
    return (gnnGraph = game.gnnGraph, availableActions = maskedActions)
end

function GI.parse_action(game::GameSpec, input::String)
    try
        p = parse(Int, input)
        return 1 <= p <= game.gnnGraph.num_nodes ? p : nothing
    catch
        return nothing
    end
end