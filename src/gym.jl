const gym = PyNULL()

struct Box
    p::PyObject
end

Base.size(box::Box) = box.p.shape


struct Discrete
    p::PyObject
end

n(discrete::Discrete) = discrete.p.n


function observation_space_type(p::PyObject)
    obs_type = pytypeof(p.observation_space)
    obs_type == gym.spaces.Discrete && return Discrete
    obs_type == gym.spaces.Box && return Box
end

action_space_type(p::PyObject) = Discrete


struct Environment{ObservationSpace, ActionSpace}
    p::PyObject
end

function Environment(name::AbstractString)
    p = gym.make(name)
    ObservationSpace = observation_space_type(p)
    ActionSpace = action_space_type(p)
    Environment{ObservationSpace, ActionSpace}(p)
end

onehot_discrete(i::Integer, discrete::Discrete) =
    onehot(i + 1, 1:n(discrete))

Base.reset(env::Environment{Box}) = Float32.(env.p.reset())

Base.reset(env::Environment{Discrete}) =
    onehot_discrete(convert(Int, env.p.reset()), observation_space(env))

function Base.step(env::Environment{Box, Discrete}, action::Integer)
    observation, reward, done, _ = env.p.step(action - 1)
    Float32.(observation), Float32(reward), Bool(done)
end

function Base.step(env::Environment{Discrete, Discrete}, action::Integer)
    observation, reward, done, _ = env.p.step(action - 1)
    encoded = onehot_discrete(observation, observation_space(env))
    encoded, Float32(reward), Bool(done)
end

Base.close(env::Environment) = env.p.close()

render(env::Environment) = env.p.render()

observation_space(env::Environment{Box}) = Box(env.p.observation_space)
observation_space(env::Environment{Discrete}) =
    Discrete(env.p.observation_space)

action_space(env::Environment{O, Discrete}) where O =
    Discrete(env.p.action_space)


__init__() = copy!(gym, pyimport("gym"))
