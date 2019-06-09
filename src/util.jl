function discount(rewards::AbstractVector{T}, γ::T) where {T<:AbstractFloat}
    discounted = similar(rewards)
    running_sum = zero(T)
    for i ∈ length(rewards):-1:1
        running_sum = running_sum * γ + rewards[i]
        discounted[i] = running_sum
    end
    discounted
end

normalize(xs::AbstractVector{<:AbstractFloat}) =
    (xs .- mean(xs)) / (std(xs) + eps(eltype(xs)))
