# Here we write out Newton Raphson and test integration with LineSearch.jl. Main tests are
# over at NonlinearSolve.jl and SimpleNonlinearSolve.jl
# Note: Enzyme tests are in a separate test group (test/enzyme/)
@testsetup module RootFinding

using SciMLBase, DifferentiationInterface, ForwardDiff
using SciMLBase: AbstractNonlinearProblem
using MaybeInplace: @bb
const DI = DifferentiationInterface

function newton_raphson(prob::AbstractNonlinearProblem, ls)
    if SciMLBase.isinplace(prob)
        return newton_raphson_iip(prob, ls)
    else
        return newton_raphson_oop(prob, ls)
    end
end

function newton_raphson_oop(prob::AbstractNonlinearProblem, ls)
    u = copy(prob.u0)
    fu = prob.f(u, prob.p)

    ls_cache = init(prob, ls, fu, u)

    alphas = Float64[]
    iter = 0
    for _ in 1:100
        iter += 1

        maximum(abs, fu) < 1.0e-8 && return true, fu, u, iter, alphas

        J = DI.jacobian(prob.f, AutoForwardDiff(), u, Constant(prob.p))
        δu = -J \ fu

        ls_sol = solve!(ls_cache, u, δu)

        push!(alphas, ls_sol.step_size)
        @bb @. u = u + ls_sol.step_size * δu

        fu = prob.f(u, prob.p)
    end

    return false, fu, u, iter, alphas
end

function newton_raphson_oop_no_alphas(prob::AbstractNonlinearProblem, ls)
    u = copy(prob.u0)
    fu = prob.f(u, prob.p)

    ls_cache = init(prob, ls, fu, u)

    iter = 0
    for _ in 1:100
        iter += 1

        maximum(abs, fu) < 1.0e-8 && return true, fu, u, iter

        # J = DI.jacobian(prob.f, AutoForwardDiff(), u, Constant(prob.p))
        J = DI.jacobian(prob.f, AutoForwardDiff(), u, Constant(prob.p))
        δu = -J \ fu

        ls_sol = solve!(ls_cache, u, δu)

        @bb @. u = u + ls_sol.step_size * δu

        fu = prob.f(u, prob.p)
    end

    return false, fu, u, iter
end

function newton_raphson_track_allocs(prob::AbstractNonlinearProblem, ls)
    converged, fu, u, iter = newton_raphson_oop_no_alphas(prob, ls)
    allocs = @allocated newton_raphson_oop_no_alphas(prob, ls)
    return converged, fu, u, iter, allocs
end

function newton_raphson_iip(prob::AbstractNonlinearProblem, ls)
    u = copy(prob.u0)
    fu = similar(u)
    fu2 = similar(u)
    prob.f(fu, u, prob.p)

    ls_cache = init(prob, ls, fu, u)

    alphas = Float64[]
    iter = 0
    for _ in 1:100
        iter += 1

        maximum(abs, fu) < 1.0e-8 && return true, fu, u, iter, alphas

        J = DI.jacobian(prob.f, fu2, AutoForwardDiff(), u, Constant(prob.p))
        δu = -J \ fu

        ls_sol = solve!(ls_cache, u, δu)

        push!(alphas, ls_sol.step_size)
        @. u = u + ls_sol.step_size * δu

        prob.f(fu, u, prob.p)
    end

    return false, fu, u, iter, alphas
end

export newton_raphson, newton_raphson_track_allocs

end

@testitem "LineSearches.jl: Newton Raphson" tags = [:linesearchesjl] setup = [RootFinding] begin
    using LineSearches, SciMLBase
    using ADTypes, Tracker, ForwardDiff, Zygote, ReverseDiff, FiniteDiff

    @testset "OOP Problem" begin
        nlf(x, p) = x .^ 2 .- p
        nlp = NonlinearProblem(nlf, [-1.0, 1.0], [3.0])

        @testset for autodiff in (
                AutoTracker(), AutoForwardDiff(), AutoZygote(),
                AutoReverseDiff(), AutoFiniteDiff(),
            )
            @testset "method: $(nameof(typeof(method)))" for method in (
                    LineSearches.BackTracking(; order = 3),
                    StrongWolfe(),
                    HagerZhang(),
                    MoreThuente(),
                    Static(),
                )
                linesearch = LineSearchesJL(; method, autodiff)
                converged, fu, u, iter, alphas = newton_raphson(nlp, linesearch)

                @test fu ≈ [0.0, 0.0] atol = 1.0e-3
                @test abs.(u) ≈ sqrt.([3.0, 3.0]) atol = 1.0e-3
            end
        end
    end

    @testset "In-Place Problem" begin
        nlf(dx, x, p) = (dx .= x .^ 2 .- p)
        nlp = NonlinearProblem(nlf, [-1.0, 1.0], [3.0])

        @testset for autodiff in (
                AutoForwardDiff(), AutoReverseDiff(), AutoFiniteDiff(),
            )
            @testset "method: $(nameof(typeof(method)))" for method in (
                    LineSearches.BackTracking(; order = 3),
                    StrongWolfe(),
                    HagerZhang(),
                    MoreThuente(),
                    Static(),
                )
                linesearch = LineSearchesJL(; method, autodiff)
                converged, fu, u, iter, alphas = newton_raphson(nlp, linesearch)

                @test fu ≈ [0.0, 0.0] atol = 1.0e-3
                @test abs.(u) ≈ sqrt.([3.0, 3.0]) atol = 1.0e-3
            end
        end
    end
end

@testitem "Native Line Search: Newton Raphson" tags = [:core] setup = [RootFinding] begin
    using SciMLBase
    using ADTypes, Tracker, ForwardDiff, Zygote, ReverseDiff, FiniteDiff
    using StaticArraysCore: SVector
    using DifferentiationInterface

    @testset "OOP Problem" begin
        nlf(x, p) = x .^ 2 .- p
        nlp = NonlinearProblem(nlf, [-1.0, 1.0], [3.0])

        @testset "method: $(nameof(typeof(method)))" for method in (
                LiFukushimaLineSearch(),
                NoLineSearch(0.5),
            )
            converged, fu, u, iter, alphas = newton_raphson(nlp, method)

            @test fu ≈ [0.0, 0.0] atol = 1.0e-1
            @test abs.(u) ≈ sqrt.([3.0, 3.0]) atol = 1.0e-1
        end

        @testset for autodiff in (
                AutoTracker(), AutoForwardDiff(), AutoZygote(),
                AutoReverseDiff(), AutoFiniteDiff(),
            )
            @testset "method: $(nameof(typeof(method)))" for method in (
                    BackTracking(; order = Val(3), autodiff),
                    BackTracking(; order = Val(2), autodiff),
                )
                converged, fu, u, iter, alphas = newton_raphson(nlp, method)

                @test fu ≈ [0.0, 0.0] atol = 1.0e-3
                @test abs.(u) ≈ sqrt.([3.0, 3.0]) atol = 1.0e-3
            end
        end

    end

    @testset "OOP Problem, static" begin
        nlf(x, p) = x .^ 2 .- p
        x0 = SVector{2,Float64}(-1.0, 1.0)
        params = 3.0
        nlp = NonlinearProblem(nlf, x0, params)
        @testset for autodiff in (
                AutoForwardDiff(), AutoZygote(),
                AutoReverseDiff(), AutoFiniteDiff(),
                DifferentiationInterface.AutoForwardFromPrimitive(AutoForwardDiff()),
            )
            @testset "method: $(nameof(typeof(method)))" for method in (
                    BackTracking(; order = Val(3), autodiff),
                    BackTracking(; order = Val(2), autodiff),
                )
                converged, fu, u, iter, alphas = newton_raphson(nlp, method)

                @test fu ≈ [0.0, 0.0] atol = 1.0e-3
                @test abs.(u) ≈ sqrt.([3.0, 3.0]) atol = 1.0e-3

                converged, fu, u, iter, allocs = newton_raphson_track_allocs(nlp, method)
                println("$(autodiff), $(nameof(typeof(method))): allocs = $(allocs)")
                @test allocs == 0
                # if autodiff in (AutoForwardDiff(), AutoFiniteDiff())
                #     @test allocs == 0
                # end
                @test fu ≈ [0.0, 0.0] atol = 1.0e-3
                @test abs.(u) ≈ sqrt.([3.0, 3.0]) atol = 1.0e-3
            end
        end
    end

    @testset "In-Place Problem" begin
        nlf(dx, x, p) = (dx .= x .^ 2 .- p)
        nlp = NonlinearProblem(nlf, [-1.0, 1.0], [3.0])

        @testset "method: $(nameof(typeof(method)))" for method in (
                LiFukushimaLineSearch(),
                NoLineSearch(0.5),
            )
            converged, fu, u, iter, alphas = newton_raphson(nlp, method)

            @test fu ≈ [0.0, 0.0] atol = 1.0e-1
            @test abs.(u) ≈ sqrt.([3.0, 3.0]) atol = 1.0e-1
        end

        @testset for autodiff in (
                AutoForwardDiff(), AutoReverseDiff(), AutoFiniteDiff(),
            )
            @testset "method: $(nameof(typeof(method)))" for method in (
                    BackTracking(; order = Val(3), autodiff),
                    BackTracking(; order = Val(2), autodiff),
                )
                converged, fu, u, iter, alphas = newton_raphson(nlp, method)

                @test fu ≈ [0.0, 0.0] atol = 1.0e-3
                @test abs.(u) ≈ sqrt.([3.0, 3.0]) atol = 1.0e-3
            end
        end
    end
end
