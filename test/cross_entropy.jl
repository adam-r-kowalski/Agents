@testset "cross entropy agent tests" begin

@testset "construct cross entropy agent with all defaults" begin
    env = Environment("CartPole-v0")
    agent = CrossEntropy(env)
    actions = action_space(env).n
    @test size(agent.π[1].W, 2) == size(observation_space(env))[1]
    @test size(agent.π[2].W, 1) == actions
    @test agent.optimizer isa ADAM
    @test agent.transitions isa Vector{<:Agents.Transition}
    @test length(agent.transitions) == 0
    @test agent.episodes isa CircularBuffer{<:Agents.Episode}
    @test length(agent.episodes) == 0
    @test agent.batch_size == 16
    @test agent.percentile ≈ 0.7
    @test agent.actions == actions
end

@testset "cross entropy agent adapts to environment type" begin
    env = Environment("FrozenLake-v0")
    agent = CrossEntropy(env)
    actions = action_space(env).n
    @test size(agent.π[1].W, 2) == observation_space(env).n
    @test size(agent.π[2].W, 1) == actions
    @test agent.actions == actions
end

@testset "cross entropy agent can have custom optimizer" begin
    env = Environment("CartPole-v0")
    agent = CrossEntropy(env, optimizer=RMSProp())
    @test agent.optimizer isa RMSProp
end

@testset "cross entropy agent can have custom batch size" begin
    env = Environment("CartPole-v0")
    agent = CrossEntropy(env, batch_size=32)
    @test agent.batch_size == 32
end

@testset "cross entropy agent can have custom percentile" begin
    env = Environment("CartPole-v0")
    agent = CrossEntropy(env, percentile=0.5)
    @test agent.percentile ≈ 0.5
end

@testset "cross entropy agent can have custom policy constructor" begin
    env = Environment("CartPole-v0")

    custom_π(observation_space::Box{1}, action_space::Discrete) =
        Dense(size(observation_space)[1], action_space.n)

    agent = CrossEntropy(env, construct_π=custom_π)
    @test size(agent.π.W, 2) == 4
    @test size(agent.π.W, 1) == 2
end

end
