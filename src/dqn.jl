mutable struct DQN{Observation, Quality, Optimizer}
    Q::Quality
    Q̂::Quality
    optimizer::Optimizer
    replay_buffer::CircularBuffer{Transition{Observation}}
    minibatch_size::Int32
    ε::Float32
    ε_decay::Float32
    γ::Float32
    actions::Int
    sync_every::Int
    iterations::Int
end

construct_Q(::Type{DQN},
            observations::Box, actions::Discrete, hidden::Integer) =
    Chain(Dense(size(observations)[1], hidden, selu),
          Dense(hidden, n(actions)))

construct_Q(::Type{DQN},
            observations::Discrete, actions::Discrete, hidden::Integer) =
    Chain(Dense(n(observations), hidden, selu),
          Dense(hidden, n(actions)))

function DQN(env::Environment;
             η=0.01, minibatch_size=32, hidden=128, capacity=10_000,
             ε_decay=0.999, γ=0.9, sync_every=1000)
    observations = observation_space(env)
    actions = action_space(env)
    Q = construct_Q(DQN, observations, actions, hidden)
    Q̂ = deepcopy(Q)
    optimizer = ADAM(η)
    Observation = typeof(reset(env))
    Quality = typeof(Q)
    Optimizer = typeof(optimizer)
    replay_buffer = CircularBuffer{Transition{Observation}}(capacity)
    ε = Float32(1)
    iterations = 0
    DQN{Observation, Quality, Optimizer}(
        Q, Q̂, optimizer, replay_buffer, Int32(minibatch_size),
        ε, Float32(ε_decay), Float32(γ), n(actions),
        sync_every, iterations)
end

select_action!(agent::DQN{Observation},
               observation::Observation) where Observation =
    Int32(rand() ≤ agent.ε ?
        rand(1:agent.actions) :
        argmax(agent.Q(observation)))

function improve!(agent::DQN{Observation}) where Observation
    length(agent.replay_buffer) < agent.minibatch_size && return nothing
    minibatch = sample(agent.replay_buffer, agent.minibatch_size, replace=false)
    observations = Observation[]
    actions = OneHotVector[]
    rewards = Float32[]
    next_observations = Observation[]
    dones = Bool[]
    for transition ∈ minibatch
        push!(observations, transition.observation)
        push!(actions, onehot(transition.action, 1:agent.actions))
        push!(rewards, transition.reward)
        push!(next_observations, transition.next_observation)
        push!(dones, transition.done)
    end
    ŷ = agent.Q(reduce(hcat, observations))[reduce(hcat, actions)]
    logits = data(agent.Q̂(reduce(hcat, next_observations)))
    Q_values = reshape(maximum(logits, dims=1), :)
    Q_values[dones] .= 0
    y = rewards + agent.γ * Q_values
    θ = params(agent.Q)
    Δ = gradient(() -> mse(ŷ, y), θ)
    update!(agent.optimizer, θ, Δ)
    agent.ε *= agent.ε_decay
    agent.iterations % agent.sync_every == 0 && (agent.Q̂ = deepcopy(agent.Q))
end


function remember!(agent::DQN{Observation},
                   transition::Transition{Observation}) where Observation
    push!(agent.replay_buffer, transition)
    improve!(agent)
    agent.iterations += 1
end
