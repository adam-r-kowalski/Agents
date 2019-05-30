module Agents

export Transition, DQN, CrossEntropy, select_action!, remember!

using PyCall, Flux, Distributions, DataStructures, Statistics, Plots, Juno
using Flux: OneHotVector, onehot, mse, crossentropy, params, data
using Flux.Tracker: gradient, update!, TrackedReal

struct Transition{Observation}
    observation::Observation
    action::Int32
    reward::Float32
    next_observation::Observation
    done::Bool
end

struct Episode{Observation}
    transitions::Vector{Transition{Observation}}
    reward::Float32
end

include("gym.jl")
include("networks.jl")
include("cross_entropy.jl")
include("dqn.jl")
include("policy_gradient.jl")

function simulate!(agent, env;
                   episodes=1, graph_rewards=false, render_environment=false)
    rewards = Float64[]
    @progress "simulate!" for _ ∈ 1:episodes
        done = false
        episode_reward = 0f0
        observation = reset(env)
        while !done
            action = select_action!(agent, observation)
            next_observation, reward, done = step(env, action)
            transition = Transition(
                observation, action, reward, next_observation, done)
            remember!(agent, transition)
            episode_reward += reward
            observation = next_observation
            render_environment && render(env)
        end
        push!(rewards, episode_reward)
    end
    graph_rewards ? plot(rewards) : mean(rewards)
end

end
