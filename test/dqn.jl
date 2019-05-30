@testset "dqn agent tests" begin

@testset "construct dqn agent with all defaults" begin
    env = Environment("CartPole-v0")
    agent = DQN(env)
    function same_weights(Q1, Q2)
        b1 = Q1[1].W.data ≈ Q2[1].W.data
        b2 = Q1[1].b.data ≈ Q2[1].b.data
        b3 = Q1[2].W.data ≈ Q2[2].W.data
        b4 = Q1[2].b.data ≈ Q2[2].b.data
        b1 && b2 && b3 && b4
    end
    actions = action_space(env).n
    @test size(agent.Q[1].W, 2) == size(observation_space(env))[1]
    @test size(agent.Q[2].W, 1) == actions
    @test same_weights(agent.Q, agent.Q̂)
    @test agent.optimizer isa ADAM
    @test agent.replay_buffer isa CircularBuffer{<:Agents.Transition}
    @test length(agent.replay_buffer) == 0
    @test agent.batch_size == 32
    @test agent.ε ≈ 1.0
    @test agent.ε_decay ≈ 0.999
    @test agent.γ ≈ 0.9
    @test agent.sync_every == 1000
    @test agent.iterations == 0
end

@testset "dqn agent adapts to environment type" begin
    env = Environment("FrozenLake-v0")
    agent = DQN(env)
    actions = action_space(env).n
    @test size(agent.Q[1].W, 2) == observation_space(env).n
    @test size(agent.Q[2].W, 1) == actions
    @test agent.actions == actions
end

@testset "dqn agent can have custom optimizer" begin
    env = Environment("CartPole-v0")
    agent = DQN(env, optimizer=RMSProp())
    @test agent.optimizer isa RMSProp
end

@testset "dqn agent can have custom batch size" begin
    env = Environment("CartPole-v0")
    agent = DQN(env, batch_size=64)
    @test agent.batch_size == 64
end

@testset "dqn agent can have custom capacity" begin
    env = Environment("CartPole-v0")
    agent = DQN(env, capacity=1000)
    @test agent.replay_buffer.capacity == 1000
end

@testset "dqn agent can have custom ε decay" begin
    env = Environment("CartPole-v0")
    agent = DQN(env, ε_decay=0.9)
    @test agent.ε_decay ≈ 0.9
end

@testset "dqn agent can have custom γ" begin
    env = Environment("CartPole-v0")
    agent = DQN(env, γ=0.5)
    @test agent.γ ≈ 0.5
end

@testset "dqn agent can have custom sync interval" begin
    env = Environment("CartPole-v0")
    agent = DQN(env, sync_every=10_000)
    @test agent.sync_every == 10_000
end

@testset "dqn agent can have custom Q constructor" begin
    env = Environment("CartPole-v0")

    custom_Q(observation_space::Box{1}, action_space::Discrete) =
        Dense(size(observation_space)[1], action_space.n)

    agent = DQN(env, construct_Q=custom_Q)
    @test size(agent.Q.W, 2) == 4
    @test size(agent.Q.W, 1) == 2
end

@testset "dqn agent can take an action in the environment" begin
    env = Environment("CartPole-v0")
    agent = DQN(env)
    observation = reset(env)
    action = select_action!(agent, observation)
    @test action ∈ 1:action_space(env).n
end

@testset "dqn agent can remember transitions in replay_buffer" begin
    env = Environment("CartPole-v0")
    agent = DQN(env)
    @test length(agent.replay_buffer) == 0
    @test agent.iterations == 0
    ε = agent.ε
    observation = reset(env)
    action = select_action!(agent, observation)
    next_observation, reward, done = step(env, action)
    transition = Transition(observation, action, reward, next_observation, done)
    remember!(agent, transition)
    @test length(agent.replay_buffer) == 1
    @test agent.replay_buffer[1] == transition
    @test agent.ε ≈ ε * agent.ε_decay
    @test agent.iterations == 1
end

@testset "dqn agent replay buffer removes oldest transitions" begin
    env = Environment("CartPole-v0")
    agent = DQN(env, capacity=100)
    ε = agent.ε
    observation = reset(env)
    transitions = eltype(agent.replay_buffer)[]
    for i ∈ 1:100
        action = select_action!(agent, observation)
        next_observation, reward, done = step(env, action)
        transition = Transition(
            observation, action, reward, next_observation, done)
        remember!(agent, transition)
        observation = done ? reset(env) : next_observation
        push!(transitions, transition)
    end
    @test agent.ε ≈ ε * agent.ε_decay^100
    @test all(transitions .== agent.replay_buffer)
    action = select_action!(agent, observation)
    next_observation, reward, done = step(env, action)
    transition = Transition(
        observation, action, reward, next_observation, done)
    remember!(agent, transition)
    push!(transitions, transition)
    @test agent.ε ≈ ε * agent.ε_decay^101
    @test all(transitions[2:end] .== agent.replay_buffer)
end

@testset "dqn agent improves" begin
    env = Environment("CartPole-v0")
    agent = DQN(env, capacity=100)
    Q = deepcopy(agent.Q)
    function same_weights(Q1, Q2)
        b1 = Q1[1].W.data ≈ Q2[1].W.data
        b2 = Q1[1].b.data ≈ Q2[1].b.data
        b3 = Q1[2].W.data ≈ Q2[2].W.data
        b4 = Q1[2].b.data ≈ Q2[2].b.data
        b1 && b2 && b3 && b4
    end
    @test same_weights(Q, agent.Q)
    observation = reset(env)
    for i ∈ 1:agent.batch_size-1
        action = select_action!(agent, observation)
        next_observation, reward, done = step(env, action)
        transition = Transition(
            observation, action, reward, next_observation, done)
        remember!(agent, transition)
        observation = done ? reset(env) : next_observation
        @test length(agent.replay_buffer) == i
        @test same_weights(Q, agent.Q)
    end
    action = select_action!(agent, observation)
    next_observation, reward, done = step(env, action)
    transition = Transition(
        observation, action, reward, next_observation, done)
    remember!(agent, transition)
    @test length(agent.replay_buffer) == agent.batch_size
    @test !same_weights(Q, agent.Q)
end

end
