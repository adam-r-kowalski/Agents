module AgentsTest

include("../src/Agents.jl")

using .Agents

using Flux, DataStructures, Test, Statistics
using Flux: OneHotVector, onehot

@testset "Agents" begin

include("test_gym.jl")
include("test_networks.jl")
include("test_util.jl")
include("test_cross_entropy.jl")
include("test_dqn.jl")

end

end
