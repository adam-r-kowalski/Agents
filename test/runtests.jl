module AgentsTest

using Agents, Flux, DataStructures, Test
using Flux: OneHotVector

@testset "Agents" begin

include("gym.jl")
include("networks.jl")
include("cross_entropy.jl")

end

end
