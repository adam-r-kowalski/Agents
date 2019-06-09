@testset "util" begin

@testset "discount rewards" begin
    rewards = [1.0, 2, 3, 2, 0, 1]
    γ = 0.9
    discounted = Agents.discount(rewards, γ)
    @test discounted[1] ≈ rewards[1] + γ * discounted[2]
    @test discounted[2] ≈ rewards[2] + γ * discounted[3]
    @test discounted[3] ≈ rewards[3] + γ * discounted[4]
    @test discounted[4] ≈ rewards[4] + γ * discounted[5]
    @test discounted[5] ≈ rewards[5] + γ * discounted[6]
    @test discounted[6] ≈ rewards[6]
end

@testset "normalize values" begin
    values = [1.0, 5, 10, 6, -3, 2]
    normalized = Agents.normalize(values)
    @test std(normalized) ≈ 1.0
    @test isapprox(mean(normalized), 0.0, atol=0.000000001)
end

end
