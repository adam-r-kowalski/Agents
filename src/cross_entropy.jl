mutable struct CrossEntropy{Observation, Policy, Optimizer}
    π::Policy
    optimizer::Optimizer
    experiences::Vector{Experience{Observation}}
    episodes::CircularBuffer{Episode{Observation}}
    batch_size::Int
    percentile::Float64
    action_space::Int
end

function CrossEntropy(observation_space::Integer, action_space::Integer)
    hidden = 128
    π = Chain(Dense(observation_space, hidden, selu),
              Dense(hidden, action_space),
              softmax)
    optimizer = ADAM(0.01)
    Observation = Vector{Float64}
    Policy = typeof(π)
    Optimizer = typeof(optimizer)
    experiences = Experience{Observation}[]
    batch_size = 16
    episodes = CircularBuffer{Episode{Observation}}(batch_size)
    percentile = 0.7
    CrossEntropy{Observation, Policy, Optimizer}(
        π, optimizer, experiences, episodes, batch_size,
        percentile, action_space)
end

select_action!(agent::CrossEntropy{Observation},
               observation::Observation) where Observation =
    rand(Categorical(agent.π(observation)))

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
                push!(actions, onehot(experience.action, 1:agent.action_space))
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
