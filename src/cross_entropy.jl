mutable struct CrossEntropy{Observation, Policy, Optimizer}
    π::Policy
    optimizer::Optimizer
    experiences::Vector{Experience{Observation}}
    episodes::CircularBuffer{Episode{Observation}}
    batch_size::Int
    percentile::Float32
    actions::Int
end

construct_policy(::Type{CrossEntropy},
                 observations::Box, actions::Discrete, hidden::Integer) =
    Chain(Dense(size(observations)[1], hidden, selu),
          Dense(hidden, n(actions)),
          softmax)

construct_policy(::Type{CrossEntropy},
                 observations::Discrete, actions::Discrete, hidden::Integer) =
    Chain(Dense(n(observations), hidden, selu),
          Dense(hidden, n(actions)),
          softmax)

observation_type(::Box) = Vector{Float32}
observation_type(::Discrete) = OneHotVector

function CrossEntropy(env::Environment;
                      η=0.01, batch_size=16, percentile=0.7, hidden=128)
    observations = observation_space(env)
    actions = action_space(env)
    π = construct_policy(CrossEntropy, observations, actions, hidden)
    optimizer = ADAM(η)
    Observation = observation_type(observations)
    Policy = typeof(π)
    Optimizer = typeof(optimizer)
    experiences = Experience{Observation}[]
    episodes = CircularBuffer{Episode{Observation}}(batch_size)
    CrossEntropy{Observation, Policy, Optimizer}(
        π, optimizer, experiences, episodes, batch_size,
        Float32(percentile), n(actions))
end

select_action!(agent::CrossEntropy{Observation},
               observation::Observation) where Observation =
    Int32(rand(Categorical(agent.π(observation))))

function improve!(agent::CrossEntropy{Observation}) where Observation
    length(agent.episodes) < agent.batch_size && return nothing
    rewards = [episode.reward for episode ∈ agent.episodes]
    reward_bound = quantile(rewards, agent.percentile)
    observations = Observation[]
    actions = OneHotVector[]
    for episode ∈ agent.episodes
        if episode.reward >= reward_bound
            for experience ∈ episode.experiences
                push!(observations, experience.observation)
                push!(actions, onehot(experience.action, 1:agent.actions))
            end
        end
    end
    ŷ = agent.π(reduce(hcat, observations))
    y = reduce(hcat, actions)
    θ = params(agent.π)
    Δ = gradient(() -> crossentropy(ŷ, y), θ)
    update!(agent.optimizer, θ, Δ)
end

function remember!(agent::CrossEntropy{Observation},
                   experience::Experience{Observation}) where Observation
    push!(agent.experiences, experience)
    if experience.done
        experiences = agent.experiences
        reward = sum(experience.reward for experience ∈ experiences)
        push!(agent.episodes, Episode(experiences, reward))
        agent.experiences = []
        improve!(agent)
    end
end
