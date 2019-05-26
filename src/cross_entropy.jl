mutable struct CrossEntropy{Observation, Policy, Optimizer}
    π::Policy
    optimizer::Optimizer
    transitions::Vector{Transition{Observation}}
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

function CrossEntropy(env::Environment;
                      η=0.01, batch_size=16, percentile=0.7, hidden=128)
    observations = observation_space(env)
    actions = action_space(env)
    π = construct_policy(CrossEntropy, observations, actions, hidden)
    optimizer = ADAM(η)
    Observation = typeof(reset(env))
    Policy = typeof(π)
    Optimizer = typeof(optimizer)
    transitions = Transition{Observation}[]
    episodes = CircularBuffer{Episode{Observation}}(batch_size)
    CrossEntropy{Observation, Policy, Optimizer}(
        π, optimizer, transitions, episodes, batch_size,
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
            for transition ∈ episode.transitions
                push!(observations, transition.observation)
                push!(actions, onehot(transition.action, 1:agent.actions))
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
                   transition::Transition{Observation}) where Observation
    push!(agent.transitions, transition)
    if transition.done
        transitions = agent.transitions
        reward = sum(transition.reward for transition ∈ transitions)
        push!(agent.episodes, Episode(transitions, reward))
        agent.transitions = []
        improve!(agent)
    end
end
