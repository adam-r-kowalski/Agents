mutable struct DQN{Observation, Quality, Optimizer}
    Q::Quality
    Q̂::Quality
    optimizer::Optimizer
    replay_buffer::CircularBuffer{Transition{Observation}}
    batch_size::Int32
    ε::Float32
    ε_decay::Float32
    γ::Float32
    actions::Int
    sync_every::Int
    iterations::Int
end

function DQN(env::Environment;
             optimizer=ADAM(0.01), batch_size=32, capacity=10_000,
             ε_decay=0.999, γ=0.9, sync_every=1000,
             construct_Q=construct_Q)
    actions = action_space(env)
    Q = construct_Q(observation_space(env), actions)
    Q̂ = deepcopy(Q)
    Observation = typeof(reset(env))
    Quality = typeof(Q)
    Optimizer = typeof(optimizer)
    replay_buffer = CircularBuffer{Transition{Observation}}(capacity)
    ε = Float32(1)
    iterations = 0
    DQN{Observation, Quality, Optimizer}(
        Q, Q̂, optimizer, replay_buffer, Int32(batch_size),
        ε, Float32(ε_decay), Float32(γ), actions.n,
        sync_every, iterations)
end

select_action!(agent::DQN{Observation},
               observation::Observation) where Observation =
    Int32(rand() ≤ agent.ε ?
        rand(1:agent.actions) :
        argmax(agent.Q(observation)))

function training_data(agent::DQN{Observation}) where Observation
    minibatch = sample(agent.replay_buffer, agent.batch_size, replace=false)
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
    logits = data(agent.Q̂(reduce(hcat, next_observations)))
    Q_next = reshape(maximum(logits, dims=1), :)
    Q_next[dones] .= 0
    Q_values = rewards + agent.γ * Q_next
    reduce(hcat, observations), reduce(hcat, actions), Q_values
end

function improve!(agent::DQN{Observation}) where Observation
    length(agent.replay_buffer) < agent.batch_size && return nothing
    observations, actions, Q_values = training_data(agent)
    θ = params(agent.Q)
    Δ = gradient(θ) do
        mse(agent.Q(observations)[actions], Q_values)
    end
    update!(agent.optimizer, θ, Δ)
end


function remember!(agent::DQN{Observation},
                   transition::Transition{Observation}) where Observation
    push!(agent.replay_buffer, transition)
    improve!(agent)
    agent.iterations += 1
    agent.ε *= agent.ε_decay
    agent.iterations % agent.sync_every == 0 && (agent.Q̂ = deepcopy(agent.Q))
end
