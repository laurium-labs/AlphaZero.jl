"""
This module provides utilities to build neural networks with Flux,
along with a library of standard architectures.
"""
module GraphNeuralNetworkLib

export SimpleGNN, SimpleGraphNetHP

using ..AlphaZero

using CUDA
using Base: @kwdef

import Flux

CUDA.allowscalar(false)
array_on_gpu(::Array) = false
array_on_gpu(::CuArray) = true
array_on_gpu(arr) = error("Usupported array type: ", typeof(arr))

using Flux: relu, softmax, flatten
using Flux: Chain, Dense, Conv, BatchNorm, SkipConnection
import Zygote

# include("graph_network.jl")

#####
##### Flux Networks
#####

"""
    FluxNetwork <: AbstractNetwork

Abstract type for neural networks implemented using the _Flux_ framework.

The `regularized_params_` function must be overrided for all layers containing
parameters that are subject to regularization.

Provided that the above holds, `FluxNetwork` implements the full
network interface with the following exceptions:
[`Network.HyperParams`](@ref), [`Network.hyperparams`](@ref),
[`Network.forward`](@ref) and [`Network.on_gpu`](@ref).
"""
abstract type FluxGNN <: GraphNetwork end

function Base.copy(nn::Net) where Net <: FluxGNN
  #new = Net(Network.hyperparams(nn))
  #Flux.loadparams!(new, Flux.params(nn))
  #return new
  return Base.deepcopy(nn)
end

Network.to_cpu(nn::FluxGNN) = Flux.cpu(nn)

function Network.to_gpu(nn::FluxGNN)
  CUDA.allowscalar(false)
  return Flux.gpu(nn)
end

function Network.set_test_mode!(nn::FluxGNN, mode)
  Flux.testmode!(nn, mode)
end

Network.convert_input(nn::FluxGNN, x) =
  Network.on_gpu(nn) ? Flux.gpu(x) : x

Network.convert_output(nn::FluxGNN, x) = Flux.cpu(x)

Network.params(nn::FluxGNN) = Flux.params(nn)

# This should be included in Flux
function lossgrads(f, args...)
  val, back = Zygote.pullback(f, args...)
  grad = back(Zygote.sensitivity(val))
  return val, grad
end

function Network.train!(callback, nn::FluxGNN, opt::Adam, loss, data, n)
  optimiser = Flux.ADAM(opt.lr)
  params = Flux.params(nn)
  for (i, d) in enumerate(data)
    l, grads = lossgrads(params) do
      loss(d...)
    end
    Flux.update!(optimiser, params, grads)
    callback(i, l)
  end
end

function Network.train!(
    callback, nn::FluxGNN, opt::CyclicNesterov, loss, data, n)
  lr = CyclicSchedule(
    opt.lr_base,
    opt.lr_high,
    opt.lr_low, n=n)
  momentum = CyclicSchedule(
    opt.momentum_high,
    opt.momentum_low,
    opt.momentum_high, n=n)
  optimiser = Flux.Nesterov(opt.lr_low, opt.momentum_high)
  params = Flux.params(nn)
  for (i, d) in enumerate(data)
    l, grads = lossgrads(params) do
      loss(d...)
    end
    Flux.update!(optimiser, params, grads)
    optimiser.eta = lr[i]
    optimiser.rho = momentum[i]
    callback(i, l)
  end
end

regularized_params_(l) = []
regularized_params_(l::Flux.Dense) = [l.weight]
regularized_params_(l::Flux.Conv) = [l.weight]

function Network.regularized_params(net::GraphNetwork)
  return (w for l in Flux.modules(net) for w in regularized_params_(l))
end

function Network.gc(::GraphNetwork)
  GC.gc(true)
  # CUDA.reclaim()
end




# Flux.@functor does not work with abstract types
function Flux.functor(nn::Net) where Net <: FluxGNN
  children = (nn.common, nn.vhead, nn.phead, )
  constructor = cs -> Net(nn.gspec, nn.hyper, cs...)
  return (children, constructor)
end

Network.hyperparams(nn::FluxGNN) = nn.hyper

Network.game_spec(nn::FluxGNN) = nn.gspec

Network.on_gpu(nn::FluxGNN) = array_on_gpu(nn.common[end].bias)

#####
##### Include networks library
#####

include("architectures/demographnet.jl")

end
