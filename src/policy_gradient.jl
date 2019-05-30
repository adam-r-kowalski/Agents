mutable struct PolicyGradient{Observation, Policy, Optimizer}
    π::Policy
    optimizer::Optimizer
    log_probabilities::Vector{TrackedReal{Float32}}
    rewards::Vector{Float32}
    γ::Float32
    actions::Int
end

function PolicyGradient(env::Environment;
                        optimizer=ADAM(0.01), γ=0.9, construct_π=construct_π)
    observations = observation_space(env)
    actions = action_space(env)
    π = construct_π(observations, actions)
    Observation = typeof(reset(env))
    log_probabilities = TrackedReal{Float32}[]
    rewards = Float32[]
    Policy = typeof(π)
    Optimizer = typeof(optimizer)
    PolicyGradient{Observation, Policy, Optimizer}(
        π, optimizer, log_probabilities, rewards, Float32(γ), actions.n)
end

function select_action!(agent::PolicyGradient{Observation},
                        observation::Observation) where Observation
    distribution = Categorical(agent.π(observation))
    action = rand(distribution)
    push!(agent.log_probabilities, loglikelihood(distribution, [action]))
    Int32(action)
end

function discount(rewards::AbstractVector{T}, γ::T) where {T<:AbstractFloat}
    discounted = similar(rewards)
    running_sum = zero(T)
    for i ∈ length(rewards):-1:1
        running_sum = running_sum * γ + rewards[i]
        discounted[i] = running_sum
    end
    discounted
end

normalize(xs::AbstractVector{<:AbstractFloat}) =
    (xs .- mean(xs)) / (std(xs) + eps(eltype(xs)))

function improve!(agent::PolicyGradient)
    returns = normalize(discount(agent.rewards, agent.γ))
    θ = params(agent.π)
    Δ = gradient(θ) do
        sum(-agent.log_probabilities .* returns)
    end
    update!(agent.optimizer, θ, Δ)
    empty!(agent.log_probabilities)
    empty!(agent.rewards)
end

function remember!(agent::PolicyGradient{Observation},
                   transition::Transition{Observation}) where Observation
    push!(agent.rewards, transition.reward)
    transition.done && improve!(agent)
end
