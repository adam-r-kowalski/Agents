@testset "gym" begin

@testset "construct a box using low and high" begin
    box = Box([-1, -2], [1, 2])
    @test box.low == [-1, -2]
    @test box.high == [1, 2]
    @test size(box) == (2,)
end

@testset "construct a box using low and high from python" begin
    box = Box(Agents.gym.spaces.Box([-1, -2], [1, 2]))
    @test box.low == [-1, -2]
    @test box.high == [1, 2]
    @test size(box) == (2,)
end

@testset "cannot construct a box with differnt shape for low and high" begin
    @test_throws AssertionError Box([-1, -2], [1, 2, 3])
end

@testset "construct a box using low, high, and shape" begin
    box = Box(-1, 1, (3, 2))
    @test box.low == fill(-1, (3, 2))
    @test box.high == fill(1, (3, 2))
    @test size(box) == (3, 2)
end

@testset "construct a box using low, high, and shape from python" begin
    box = Box(Agents.gym.spaces.Box(-1, 1, (3, 2)))
    @test box.low == fill(-1, (3, 2))
    @test box.high == fill(1, (3, 2))
    @test size(box) == (3, 2)
end

@testset "construct a discrete using an integer" begin
    discrete = Discrete(5)
    @test discrete.n == 5
end

@testset "cannot construct a discrete using a negative integer" begin
    @test_throws AssertionError Discrete(-1)
end

@testset "construct a discrete from python" begin
    discrete = Discrete(Agents.gym.spaces.Discrete(5))
    @test discrete.n == 5
end

@testset "get observation space from python gym env" begin
    env = Agents.gym.make("CartPole-v0")
    @test Agents.ObservationSpace(env) isa Box{1}
    env = Agents.gym.make("FrozenLake-v0")
    @test Agents.ObservationSpace(env) isa Discrete
end

@testset "get action space from python gym env" begin
    env = Agents.gym.make("CartPole-v0")
    @test Agents.ActionSpace(env) isa Discrete
end

@testset "construct environment by name" begin
    env = Environment("CartPole-v0")
    @test observation_space(env) isa Box{1}
    @test action_space(env) == Discrete(2)
end

@testset "resetting environment gives back observation" begin
    env = Environment("CartPole-v0")
    observation = reset(env)
    @test observation isa Vector{Float32}
    @test size(observation) == (4,)

    env = Environment("FrozenLake-v0")
    observation = reset(env)
    @test observation isa OneHotVector
    @test size(observation) == (16,)
end

@testset "onehot discrete" begin
    encoded = Agents.onehot_discrete(4, Discrete(16))
    @test encoded isa OneHotVector
    @test size(encoded) == (16,)
    @test all(encoded[:3] .== 0)
    @test encoded[4] == 1
    @test all(encoded[5:end] .== 0)
end

@testset "stepping a environment" begin
    env = Environment("CartPole-v0")
    reset(env)
    observation, reward, done = step(env, 1)
    @test observation isa Vector{Float32}
    @test size(observation) == (4,)
    @test reward isa Float32
    @test done isa Bool

    env = Environment("FrozenLake-v0")
    reset(env)
    observation, reward, done = step(env, 1)
    @test observation isa OneHotVector
    @test size(observation) == (16,)
    @test reward isa Float32
    @test done isa Bool
end

end
