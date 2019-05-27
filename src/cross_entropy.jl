export CrossEntropy, select_action!, remember!

mutable struct CrossEntropy{Observation, Policy, Optimizer}
    π::Policy
    optimizer::Optimizer
    transitions::Vector{Transition{Observation}}
    episodes::CircularBuffer{Episode{Observation}}
    batch_size::Int
    percentile::Float32
    actions::Int
end

function CrossEntropy(env::Environment;
                      optimizer=ADAM(0.01), batch_size=16, percentile=0.7,
                      construct_π=construct_π)
    observations = observation_space(env)
    actions = action_space(env)
    π = construct_π(observations, actions)
    Observation = typeof(reset(env))
    Policy = typeof(π)
    Optimizer = typeof(optimizer)
    transitions = Transition{Observation}[]
    episodes = CircularBuffer{Episode{Observation}}(batch_size)
    CrossEntropy{Observation, Policy, Optimizer}(
        π, optimizer, transitions, episodes, batch_size,
        Float32(percentile), actions.n)
end

select_action!(agent::CrossEntropy{Observation},
               observation::Observation) where Observation =
    Int32(rand(Categorical(agent.π(observation))))

function training_data(agent::CrossEntropy{Observation}) where Observation
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
    reduce(hcat, observations), reduce(hcat, actions)
end

function improve!(agent::CrossEntropy)
    length(agent.episodes) < agent.batch_size && return nothing
    observations, actions = training_data(agent)
    θ = params(agent.π)
    Δ = gradient(θ) do
        crossentropy(agent.π(observations), actions)
    end
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
