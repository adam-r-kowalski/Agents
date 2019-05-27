module AgentsTest

using Agents, Flux, Test
using Flux: OneHotVector

@testset "Agents" begin

include("gym.jl")
include("networks.jl")

end

end
