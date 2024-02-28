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

"""
Get weights for quadrature integration
"""
function get_weights(deltax, N)
    wvec = deltax * ones(N)
    wvec[1] = 0.5 * wvec[1]
    wvec[end] = 0.5 * wvec[end]
    return wvec
end

function sampleA(n, k)
    randT = rand(repeat([n], 2 * k)...)
    permutevec = [k+1:2*k; 1:k]
    A = 1 / 2 * (randT - permutedims(randT, permutevec))
    return A
end

function quad_int(samples, weights)
    return transpose(samples)*weights
    return transpose(samples) * weights
end

function popdist(theta, bmat)
    return exp.(bmat * theta)
end

function append0s(vectofill, finaldimension)
    return append!(vectofill, zeros(finaldimension - length(vectofill)))
end

function getlegendrecoeffs(order)
    coeffbin = zeros(order)
    coeffbin[order] = 1
    p = Legendre(coeffbin) #ARE THIS NORMALIZED?
    #transform to standard basis of monomials
    pst = convert(Polynomial, p)
    #get legendre coefficients
    lcoeffs = coeffs(pst)
    return lcoeffs
end

function buildT(spandimension)
    #initialize T
    T = zeros(spandimension, spandimension)
    for n in 1:spandimension
        lcoeffn = getlegendrecoeffs(n)
        tostore = append0s(lcoeffn, spandimension)
        T[n, :] = tostore
    end
    return T
end

function invertT(T)
    #create a triangular matrix
    UTT = LowerTriangular(T)
    return inv(UTT)
end

function computeB(A, T, k)
    B = deepcopy(A)
    inv_T = Matrix(invertT(T))
    for i in 1:2*k
        IA = collect([1:i-1; 0; i+1:2*k])
        IB = collect([0; i])
        IC = collect(1:2*k)
        B .= tensorcontract(IC, B,
            IA, inv_T,
            IB)
    end
    return B
end

function buildPvec(r)
    vec = Array{Int64}(undef, r)
    for i in 1:r
        if i <= r / 2
            vec[i] = 2 * i
        else
            vec[i] = 2 * (r - i + 1) - 1
        end
    end
    return vec
end

function buildP(r)
    transpose(Matrix(Permutation(buildPvec(r))))
end

function computeQ(B, threshold)
    eigens, U = eigen(B)
    n = length(eigens)

    r = sum(abs.(imag.(eigens)) .> threshold)
    r = Integer(floor(r / 2) * 2)

    filter = I(n)[setdiff(1:end, (r÷2+1):(n-r÷2)), :]
    indsort = sortperm(imag(eigens))
    Psort = Matrix(Permutation(indsort))
    Usorted = U * Psort
    eigenssorted = transpose(Psort) * eigens
    Usfilt = Usorted * transpose(filter)
    eigensfilt = filter * eigenssorted
    P = buildP(r)
    M = 1 / sqrt(2) .* [1 -1im; 1 1im]
    #return V
    V = Usfilt * P * (I(r ÷ 2) ⊗ M)
    Q = V * sqrt.(diagm(abs.(transpose(P) * eigensfilt)))
    return real(Q), r
end

function getbasispx(spandimension)
    return [Polynomial(coeffbinary(i, spandimension)) for i in 1:spandimension]
end

function getlegendre(T, basis)
    return T * basis #return Q*basis*
end

function evaluatebi(G, x, basis, n, k)
    # Evaluated at x, which is a point
    L = Array{Float64}(undef, n, k)
    for polyi in 1:n
        for coordj in 1:k
            L[polyi, coordj] = basis[polyi](x[coordj])
        end
    end

    tensor_L = L[:, 1]
    for i in 2:k
        tensor_L = tensorproduct(1:i, tensor_L, 1:i-1, L[:, i], [i])
    end
    return tensorcontract(G, 0:k, tensor_L, 1:k)
end

function buildG(Q, n, k)
    dimensions = [n^k; repeat([n], k)]
    F = Array{Float64}(undef, dimensions...) .= 0
    inds = collect(Iterators.product(Iterators.repeated(1:n, k)...))
    iterator = map(collect, reduce(vcat, inds))
    for i in 1:n^k
        ivec = iterator[i]
        j = (ivec .- 1)' * n .^ collect(k-1:-1:0) + 1
        F[j, ivec...] = 1
    end
    return tensorcontract(Q, [0, 1], F, [0; 2:k+1])
end

function coeffbinary(term, order)
    vec = zeros(order)
    vec[term] = 1
    return vec
end

function buildW(r::Int64)
    #build rotation matrix
    R = [0 1; -1 0]
    dim = r ÷ 2 #get dimension of W
    #initialize W
    W0 = [zeros(Int64, 2, 2) for i in 1:dim, j in 1:dim]
    W0[diagind(W0)] .= [R]
    #construct W
    W = reduce(vcat, [reduce(hcat, W0[i, :]) for i in 1:dim])
    return W
end

function evaluatebvec(G, basis, big_eval_points, n, r, k)
    # Evaluate b_i's at all points x
    npoints = length(big_eval_points)
    bmat = Array{Float64}(undef, npoints, r)

    for i in 1:npoints
        bmat[i, :] = evaluatebi(G, big_eval_points[i], basis, n, k)
    end
    return bmat # N^k x n^k
end

function getbasispx(spandimension)
    return [Polynomial(coeffbinary(i, spandimension)) for i in 1:spandimension]
end

function re!(dtheta, theta, p, t)
    W, bmat, weights = p #coeffs=A
    #integrate samples of population density for quadrature weights
    P(theta) = quad_int(popdist(theta, bmat), weights)
    #evaluate gradient
    gradP = grad(central_fdm(5, 1), P, theta)[1] #
    # gradP = transpose(transpose(exp.(bmat * theta)) * bmat) * P(theta)
    dtheta .= W * gradP
    return dtheta
end

###########################################################################################
#FUNCTIONS FOR INTEGRATION IN DISCRETIZED ORIGINAL SPACE
###########################################################################################

function get_vander(xvec, n)
    Vmat = collect(ntuple(i -> xvec::Array{Float64,1} .^ (i - 1)::Int64, n))
    return transpose(mapreduce(permutedims, vcat, Vmat))
end

function buildF(V, A, k)
    F = deepcopy(A)
    for i in 1:2*k
        IA = collect([1:i-1; 0; i+1:2*k])
        IB = collect([i; 0])
        IC = collect(1:2*k)
        F = tensorcontract(IC, F,
            IA, V,
            IB)
    end
    return F
end

function re_discrete!(dpdt, p, pars, t)
    #unpack parameters: 
    #payoff coefficients (F), 
    #integration weights (wvec)
    F, wvec, N, k = pars
    # pw = reshape(p .* wvec, repeat([N], k)...)

    # #compute differential change
    # IA = collect(1:2*k)
    # IB = collect(k+1:2*k)
    # IC = collect(1:k)
    # Fpw = tensorcontract(IC, F,
    #     IA, pw,
    #     IB)

    dpdt .= p .* (F * (p .* wvec))

    return dpdt
end

function main()
    n = 4
    k = 4
    # r = n^k - 2
    threshold = 1e-8
    N = 10 #integration resolution
    tspan = (1, 10) #integration time span
    evalpoints = collect(range(-1, 1, N))
    a, b = (-1, 1) #trait domain
    #resolution window size
    deltax = (b - a) / (N - 1)
    wvec = get_weights(deltax, N)
    A = sampleA(n, k)
    # A = zeros(repeat([n], 2 * k)...)

    # A[3,1,1,1] = 0
    # A[2,2,1,1] = 0
    # A[1,3,1,1] = 1
    # A = permutedims(A,(3,4,1,2)) - A 
    V = get_vander(evalpoints, n)
    F = buildF(V, A, k)
    #construct F
    T = buildT(n)
    B = computeB(A, T, k)

    ##reshape B
    Blong = reshape(B, (n^k, n^k))

    ##do SVD
    Q, r = computeQ(Blong, threshold)

    ##get basis of monomials
    monomialbasis = getbasispx(n)
    legendrebasis = getlegendre(T, monomialbasis)
    ##grid of points evaluating each b
    G = buildG(Q, n, k)

    big_wvec = deepcopy(wvec)
    big_eval_points = deepcopy(evalpoints)

    big_eval_points = collect(Iterators.product(ntuple(i -> evalpoints, k)...))
    big_wvec = map(prod, collect(Iterators.product(ntuple(i -> wvec, k)...)))

    # for i=2:k
    #     big_eval_points = Iterators.product(big_eval_points, evalpoints)
    #     big_wvec = map(prod, Iterators.product(big_wvec, wvec) |> collect )
    # end
    # big_eval_points = collect(Iterators.flatten(big_eval_points))
    bmat = evaluatebvec(G, legendrebasis, big_eval_points, n, r, k)
    big_wvec = vec(big_wvec)

    initial_parameters = repeat([1.0], r)
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
    println("Integrating in latent space ")
    sollatent = DifferentialEquations.solve(problemlatent, progress=true, progress_steps=10)

    # reshape F to (N^k, N^k)
    # old_F = deepcopy(F)
    F = reshape(permutedims(F, collect([k:-1:1; 2*k:-1:k+1])),(N^k, N^k))
    par_ori = (F, big_wvec, N, k) #vector of parameters
    p0 = popdist(initial_parameters, bmat)
    #set up ODE problem and solve it
    problemdisc = ODEProblem(re_discrete!, p0, tspan, par_ori)
    println("Integrating in original space")
    soldisc = DifferentialEquations.solve(problemdisc, progress=true, progress_steps=10)
    return sollatent, bmat, N, k, r, soldisc, evalpoints
end

sollatent, bmat, N, k, r, soldisc, evalpoints = main()

# disc_solution = reshape(soldisc.u[end], repeat([N], k)...)
# lat_solution = reshape(popdist(sollatent.u[end], bmat), repeat([N], k)...)

times = sort(unique(cat(sollatent.t,soldisc.t, dims=1)))

end_time = minimum([sollatent.t[end],soldisc.t[end]])
times = times[times.<=end_time]

disc_sols = soldisc(times)

lat_sols = [popdist(sollatent(t), bmat) for t in times]

dmat = disc_sols - lat_sols

norms = [norm(dmat[:, i]) for i in 1:size(dmat, 2)]

comparison = @animate for i in 1:length(sollatent.t)
    fig = plot(layout = grid(1,2), legend=true)
    
    lat_plot = abs.(reshape(lat_sols[i], repeat([N], k)...))[repeat([1], k-2)..., :, :]
    lat_plot = log.(lat_plot)

    disc_plot = abs.(reshape(disc_sols[i], repeat([N], k)...))[repeat([1], k-2)..., :, :]
    disc_plot = log.(disc_plot)

    min_val = minimum(minimum.([lat_plot,disc_plot]))
    max_val = maximum(maximum.([lat_plot,disc_plot]))
    colorrange = (min_val, max_val)

    heatmap!(fig[1], evalpoints, evalpoints,lat_plot, title="Latent, t=$(floor(sollatent.t[i]))", clim=colorrange)
    heatmap!(fig[2], evalpoints, evalpoints,disc_plot, title="Discretized, t=$(floor(sollatent.t[i]))", clim=colorrange)
end

gif(comparison, "comparison.gif", fps=5)