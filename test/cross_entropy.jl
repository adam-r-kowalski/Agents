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

@testset "cross entropy agent can have custom π constructor" begin
    env = Environment("CartPole-v0")

    custom_π(observation_space::Box{1}, action_space::Discrete) =
        Dense(size(observation_space)[1], action_space.n)

    agent = CrossEntropy(env, construct_π=custom_π)
    @test size(agent.π.W, 2) == 4
    @test size(agent.π.W, 1) == 2
end

@testset "cross entropy agent can take an action in the environment" begin
    env = Environment("CartPole-v0")
    agent = CrossEntropy(env)
    observation = reset(env)
    action = select_action!(agent, observation)
    @test action ∈ 1:action_space(env).n
end

@testset "cross entropy agent can remember transitions" begin
    env = Environment("CartPole-v0")
    agent = CrossEntropy(env)
    @test length(agent.transitions) == 0
    observation = reset(env)
    action = select_action!(agent, observation)
    next_observation, reward, done = step(env, action)
    transition = Transition(observation, action, reward, next_observation, done)
    remember!(agent, transition)
    @test length(agent.transitions) == 1
end

@testset "cross entropy agent stores transitions" begin
    env = Environment("CartPole-v0")
    agent = CrossEntropy(env)
    @test length(agent.transitions) == 0
    @test length(agent.episodes) == 0
    done = false
    observation = reset(env)
    transitions = agent.transitions
    for i ∈ 1:200
        action = select_action!(agent, observation)
        next_observation, reward, done = step(env, action)
        transition = Transition(
            observation, action, reward, next_observation, done)
        remember!(agent, transition)
        observation = next_observation
        if done
            @test length(agent.transitions) == 0
            @test length(agent.episodes) == 1
            @test agent.episodes[1].transitions == transitions
            break
        else
            @test length(agent.transitions) == i
        end
    end
end

@testset "cross entropy agent improves" begin
    env = Environment("CartPole-v0")
    agent = CrossEntropy(env)
    π = deepcopy(agent.π)
    function same_weights(π1, π2)
        b1 = π1[1].W.data ≈ π2[1].W.data
        b2 = π1[1].b.data ≈ π2[1].b.data
        b3 = π1[2].W.data ≈ π2[2].W.data
        b4 = π1[2].b.data ≈ π2[2].b.data
        b1 && b2 && b3 && b4
    end
    @test same_weights(π, agent.π)
    @test length(agent.episodes) == 0
    for i ∈ 1:agent.batch_size-1
        done = false
        observation = reset(env)
        while !done
            action = select_action!(agent, observation)
            next_observation, reward, done = step(env, action)
            transition = Transition(
                observation, action, reward, next_observation, done)
            remember!(agent, transition)
            observation = next_observation
        end
        @test length(agent.episodes) == i
        @test same_weights(π, agent.π)
    end
    done = false
    observation = reset(env)
    while !done
        action = select_action!(agent, observation)
        next_observation, reward, done = step(env, action)
        transition = Transition(
            observation, action, reward, next_observation, done)
        remember!(agent, transition)
        observation = next_observation
    end
    @test length(agent.episodes) == agent.batch_size
    @test !same_weights(π, agent.π)
end

end
