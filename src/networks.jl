export construct_π, construct_Q

construct_π(observation_space::Box{1}, action_space::Discrete) =
    Chain(Dense(size(observation_space)[1], 128, selu),
          Dense(128, action_space.n),
          softmax)

construct_π(observation_space::Discrete, action_space::Discrete) =
    Chain(Dense(observation_space.n, 128, selu),
          Dense(128, action_space.n),
          softmax)

construct_Q(observation_space::Box{1}, action_space::Discrete) =
    Chain(Dense(size(observation_space)[1], 128, selu),
          Dense(128, action_space.n))

construct_Q(observation_space::Discrete, action_space::Discrete) =
    Chain(Dense(observation_space.n, 128, selu),
          Dense(128, action_space.n))
