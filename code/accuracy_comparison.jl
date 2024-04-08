using Random
using QuadGK
using FiniteDifferences
using LegendrePolynomials #for evaluating legendre polynomials
using Polynomials, SpecialPolynomials #to change basis from standard to legendre
using TensorOperations #used to compute B
using LinearAlgebra
using Permutations, Kronecker #to do Andrew's magic
using DifferentialEquations #integrate ODEs
using ProgressLogging
using Plots
using DelimitedFiles

include("multipledimensions_integration.jl")
# : sampleA, get_weights, get_vander, build_F, build_T, computeB, computeQ, computeQ_r, getbasispx, getlegendre, buildG, evaluatebvec, popdist, quad_int, re, re_discrete


function simulation(n, k, N, threshold, r, use_threshold, tspan, a, b, A, times)
    evalpoints = collect(range(a, b, N))
    #resolution window size
    deltax = (b - a) / (N - 1)
    wvec = get_weights(deltax, N)

    #construct F
    T = buildT(n)
    B = computeB(A, T, k)

    ##reshape B
    Blong = reshape(B, (n^k, n^k))
	B = nothing

    ##do SVD
    if use_threshold
        Q, r = computeQ(Blong, threshold)
    else
        Q = computeQ_r(Blong, r)
    end

	Blong = nothing

    ##get basis of monomials
    monomialbasis = getbasispx(n)
    legendrebasis = getlegendre(T, monomialbasis)

	T = nothing
    ##grid of points evaluating each b
    G = buildG(Q, n, k)

    big_eval_points = collect(Iterators.product(ntuple(i -> evalpoints, k)...))
    big_wvec = map(prod, collect(Iterators.product(ntuple(i -> wvec, k)...)))

    bmat = evaluatebvec(G, legendrebasis, big_eval_points, n, r, k)
    big_wvec = vec(big_wvec)

    initial_parameters = repeat([0.1], r)
    p0 = popdist(initial_parameters, bmat)
    mass = quad_int(p0, big_wvec)

    normalizing_row = Q \ I[1:n^k, 1]
    initial_parameters -= normalizing_row * log(mass)

    # p0 = popdist(initial_parameters, bmat)
    # mass = quad_int(p0, big_wvec)
    # print(mass)

    W = buildW(r)
    par_lat = (W, bmat, big_wvec)
    problemlatent = ODEProblem(re!, initial_parameters, tspan, par_lat)
    # println("Integrating in latent space ")
    @time sollatent = DifferentialEquations.solve(problemlatent) #, progress = true, progress_steps = 10)
    print("Latent Done\n")

    # reshape F to (N^k, N^k)
    F = buildF(get_vander(evalpoints, n), A, k)

    F = reshape(permutedims(F, collect([k:-1:1; 2*k:-1:k+1])), (N^k, N^k))
    par_ori = (F, big_wvec, N, k) #vector of parameters
    p0 = popdist(initial_parameters, bmat)
    #set up ODE problem and solve it
    problemdisc = ODEProblem(re_discrete!, p0, tspan, par_ori)
    # println("Integrating in original space")
    @time soldisc = DifferentialEquations.solve(problemdisc) #, progress = true, progress_steps = 10)

    print("Discrete Done\n")
    # times = sort(unique(cat(sollatent.t, soldisc.t, dims = 1)))

    # end_time = minimum([sollatent.t[end], soldisc.t[end]])
    # times = times[times.<=end_time]

    disc_sols = soldisc(times)

    lat_sols = [popdist(sollatent(t), bmat) for t in times]

    return disc_sols, lat_sols
end

function step(n, k, N, threshold, r, use_threshold, tspan, a, b, trials, times, step_num)
    total_diff = zeros(trials, length(times))
    for i in 1:trials
        A = sampleA(n, k)

        disc_sols, lat_sols = simulation(n, k, N, threshold, r, use_threshold, tspan, a, b, A, times)

        dmat = disc_sols - lat_sols
        total_diff[i, :] += [norm(dmat[:, i]) for i in axes(dmat, 2)]
    end

    print(step_num)
    print("\n")
    return sum(total_diff, dims=1) / trials
end


function main()
    n_vals = (6)
    k_vals = (3)
    N_vals = (10,20,30,40)

    trials = 5

    threshold = 1e-8
    use_threshold = true
    r = 1
    tspan = (1, 10)
    a, b = -1, 1
    times = [1, 3, 5, 7, 9]

    results = Dict((n, k, N) => step(n, k, N, threshold, r, use_threshold, tspan, a, b, trials, times, step_num) for (step_num, (n, k, N)) in enumerate(Iterators.product(n_vals, k_vals, N_vals)))
    return results
end

results = main()
