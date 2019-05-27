export gym, Box, Discrete, Environment, observation_space, action_space

const gym = PyNULL()

struct Box{Rank}
    low::Array{Float32, Rank}
    high::Array{Float32, Rank}

    function Box(low::Array{Float32, Rank}, high::Array{Float32, Rank}) where Rank
        @assert(size(low) == size(high))
        new{Rank}(low, high)
    end
end

Box(low::AbstractArray{<:Number, Rank},
    high::AbstractArray{<:Number, Rank}) where Rank =
    Box(Float32.(low), Float32.(high))

Box(low::Number, high::Number, shape::NTuple{Rank, Integer}) where Rank =
    Box(fill(low, shape), fill(high, shape))

function Box(p::PyObject)
    @assert pytypeof(p) == gym.spaces.Box
    Box(p.low, p.high)
end

Base.size(box::Box) = size(box.low)


struct Discrete
    n::Int32

    function Discrete(n::Integer)
        @assert n > 0
        new(n)
    end
end

function Discrete(p::PyObject)
    @assert pytypeof(p) == gym.spaces.Discrete
    Discrete(p.n)
end


function ObservationSpace(p::PyObject)
    obs = p.observation_space
    obs_type = pytypeof(obs)
    obs_type == gym.spaces.Discrete && return Discrete(obs)
    obs_type == gym.spaces.Box && return Box(obs)
end

ActionSpace(p::PyObject) = Discrete(p.action_space)


struct Environment{ObservationSpace, ActionSpace}
    p::PyObject
    observation_space::ObservationSpace
    action_space::ActionSpace
end

function Environment(name::AbstractString)
    p = gym.make(name)
    Environment(p, ObservationSpace(p), ActionSpace(p))
end

onehot_discrete(i::Integer, discrete::Discrete) = onehot(i, 1:discrete.n)

Base.reset(env::Environment{<:Box}) = Float32.(env.p.reset())

Base.reset(env::Environment{Discrete}) =
    onehot_discrete(env.p.reset() + 1, observation_space(env))

function Base.step(env::Environment{<:Box, Discrete}, action::Integer)
    observation, reward, done, _ = env.p.step(action - 1)
    Float32.(observation), Float32(reward), Bool(done)
end

function Base.step(env::Environment{Discrete, Discrete}, action::Integer)
    observation, reward, done, _ = env.p.step(action - 1)
    encoded = onehot_discrete(observation + 1, observation_space(env))
    encoded, Float32(reward), Bool(done)
end

Base.close(env::Environment) = env.p.close()

render(env::Environment) = env.p.render()

observation_space(env::Environment) = env.observation_space

action_space(env::Environment) = env.action_space

__init__() = copy!(gym, pyimport("gym"))
