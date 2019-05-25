include("../src/Agents.jl")

env = Agents.Environment("CartPole-v0")
env = Agents.Environment("FrozenLake-v0")

agent = Agents.CrossEntropy(env)

Agents.simulate!(agent, env; episodes=100, graph_rewards=true)

Agents.simulate!(agent, env; episodes=5, render_environment=true)

close(env)
