using GraphNeuralNetworks
using Statistics

"""
Super, super demo example

network will be this 
nodeFeature -> number of features on node
innerSize -> how large the inner network layer will be
actionCount -> how many actions plus 1 (network importance)

GNNChain(GCNConv(nodeFeature => innerSize),
                        BatchNorm(innerSize),     # Apply batch normalization on node features (nodes dimension is batch dimension)
                        x -> relu.(x),     
                        GCNConv(64 => innerSize, relu),
                        GlobalPool(mean),  # aggregate node-wise features into graph-wise features
                        Dense(innerSize, actionCount+1)) |> device;
"""
@kwdef struct SimpleGraphNetHP
  innerSize :: Int
end

"""
    SimpleGNN <: FluxGNN

    Something simple
"""

mutable struct SimpleGNN <: FluxGNN
  gspec
  hyper
  common
  vhead 
  phead
end

function SimpleGNN(gspec::AbstractGameSpec, hyper::SimpleGraphNetHP)
    innerSize = hyper.innerSize
    nodeFeature = GI.state_dim(gspec)
    actionCount = GI.num_actions(gspec)
    common = GNNChain(GCNConv(nodeFeature => innerSize),
                        BatchNorm(innerSize),     # Apply batch normalization on node features (nodes dimension is batch dimension)
                        x -> relu.(x),     
                        GCNConv(innerSize => innerSize, relu)
      )
      modelP = GNNChain(Dense(innerSize, 1),softmax)
      modelV = GNNChain( GlobalPool(mean),  # aggregate node-wise features into graph-wise features
                              Dense(innerSize, 1),
                              softmax);

    return SimpleGNN(gspec, hyper, common, modelV, modelP)
end

Network.HyperParams(::Type{<:SimpleGNN}) = SimpleGraphNetHP

function Base.copy(nn::SimpleGNN)
  return SimpleGNN(
    nn.gspec,
    nn.hyper,
    deepcopy(nn.common),
    deepcopy(nn.vhead),
    deepcopy(nn.phead)
  )
end

# function Network.forward(nn::SimpleGNN, state)
#   c = nn.common.(state)
#   applyV(graph) = nn.vhead(graph, graph.ndata.x)
#   resultv = applyV.(c)
#   v = first.(resultv)
#   applyP(graph) = nn.phead(graph)
#   resultp = applyP.(c)
#   @show @__LINE__
#   @show length(resultp[1].ndata.x)
#   @show state[1].num_nodes
#   p = [resultp[ind].ndata.x[indDepth] for indDepth in 1:state[1].num_nodes, ind in 1:length(state) ]
#   return (p, v)
# end

function Network.forward(nn::SimpleGNN, state)
  c = nn.common(state) # no broadcasting
  v = nn.vhead(c, c.ndata.x)   # size: num_features x num_graphs
  p = nn.phead(c, c.ndata.x)   # size: num_features x num_tot_nodes
  return p, v
end