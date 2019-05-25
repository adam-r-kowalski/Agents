module Agents

using PyCall, Flux, Distributions, DataStructures, Statistics, Plots
using Flux: OneHotVector, onehot, crossentropy, params
using Flux.Tracker: gradient, update!

struct Experience{Observation}
    observation::Observation
    action::Int
    reward::Float64
    next_observation::Observation
    done::Bool
end

struct Episode{Observation}
    experiences::Vector{Experience{Observation}}
    reward::Float64
end

include("cross_entropy.jl")

function simulate!(agent, env; episodes=1, graph=false, render=false)
    rewards = Float64[]
    for _ ∈ 1:episodes
        done = false
        episode_reward = 0.0
        observation = env.reset()
        while !done
            action = select_action!(agent, observation)
            next_observation, reward, done, _ = env.step(action - 1)
            experience = Experience(
                observation, action, reward, next_observation, done)
            remember!(agent, experience)
            episode_reward += reward
            observation = next_observation
            render && env.render()
        end
        push!(rewards, episode_reward)
    end
    graph ? plot(rewards) : mean(rewards)
end

end
