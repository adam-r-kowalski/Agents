module AgentsTest

using Agents, Flux, DataStructures, Test
using Flux: OneHotVector, onehot

@testset "Agents" begin

include("gym.jl")
include("networks.jl")
include("cross_entropy.jl")
include("dqn.jl")

end

end
