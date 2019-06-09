@testset "networks" begin

@testset "construct π box discrete" begin
    policy = construct_π(Box(-1, 1, (4,)), Discrete(2))
    manual = Chain(Dense(4, 128, selu), Dense(128, 2), softmax)

    @test size(policy[1].W) == size(manual[1].W)
    @test size(policy[1].b) == size(manual[1].b)
    @test policy[1].σ == manual[1].σ

    @test size(policy[2].W) == size(manual[2].W)
    @test size(policy[2].b) == size(manual[2].b)
    @test policy[2].σ == manual[2].σ

    @test policy[3] == manual[3]
end

@testset "construct π discrete discrete" begin
    policy = construct_π(Discrete(6), Discrete(2))
    manual = Chain(Dense(6, 128, selu), Dense(128, 2), softmax)

    @test size(policy[1].W) == size(manual[1].W)
    @test size(policy[1].b) == size(manual[1].b)
    @test policy[1].σ == manual[1].σ

    @test size(policy[2].W) == size(manual[2].W)
    @test size(policy[2].b) == size(manual[2].b)
    @test policy[2].σ == manual[2].σ

    @test policy[3] == manual[3]
end

@testset "construct Q box discrete" begin
    policy = construct_Q(Box(-1, 1, (4,)), Discrete(2))
    manual = Chain(Dense(4, 128, selu), Dense(128, 2))

    @test size(policy[1].W) == size(manual[1].W)
    @test size(policy[1].b) == size(manual[1].b)
    @test policy[1].σ == manual[1].σ

    @test size(policy[2].W) == size(manual[2].W)
    @test size(policy[2].b) == size(manual[2].b)
end

@testset "construct Q discrete discrete" begin
    policy = construct_Q(Discrete(6), Discrete(2))
    manual = Chain(Dense(6, 128, selu), Dense(128, 2))

    @test size(policy[1].W) == size(manual[1].W)
    @test size(policy[1].b) == size(manual[1].b)
    @test policy[1].σ == manual[1].σ

    @test size(policy[2].W) == size(manual[2].W)
    @test size(policy[2].b) == size(manual[2].b)
end

end
