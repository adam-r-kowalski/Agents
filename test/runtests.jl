include("../src/Agents.jl")

env = Agents.gym.make("CartPole-v0")

agent = Agents.CrossEntropy(4, 2)

Agents.simulate!(agent, env; episodes=100, graph=true)

Agents.simulate!(agent, env; episodes=5, render=true)

env.close()
