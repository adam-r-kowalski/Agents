env = Environment("CartPole-v0")
env = Environment("LunarLander-v2")
env = Environment("Ant-v2")
env = Environment("HalfCheetah-v2")

agent = CrossEntropy(env)
agent = DQN(env)
agent = PolicyGradient(env)
agent = Odin(env)

simulate!(agent, env; episodes=1, graph_rewards=false)
simulate!(agent, env; episodes=5, render_environment=true)

Agents.improve!(agent)

close(env)


function episode(env)
    env.reset()
    for _ in 1:1000
        _, _, done, _ = env.step(env.action_space.sample())
        env.render()
    end
end


episode(env.p)
