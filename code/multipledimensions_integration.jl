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
end

function popdist(theta, bmat)
    return exp.(bmat*theta)
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

function computeQ(B, r)
    eigens, U = eigen(B)
    n = length(eigens)
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
    return real(Q)
end

function getbasispx(spandimension)
    return [Polynomial(coeffbinary(i, spandimension)) for i in 1:spandimension]
end

function getlegendre(T, basis)
    return T * basis #return Q*basis*
end

function evaluatebi(G, x, basis, n, k)
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
    R = [0 1;-1 0]
    dim = r ÷ 2 #get dimension of W
    #initialize W
    W0 = [zeros(Int64, 2, 2) for i in 1:dim, j in 1:dim]
    W0[diagind(W0)] .= [R]
    #construct W
    W = reduce(vcat, [reduce(hcat, W0[i, :]) for i in 1:dim])
    return W
end

function evaluatebvec(G, basis, big_eval_points, n, r, k)
    npoints = length(big_eval_points)
    bmat = Array{Float64}(undef, npoints, r)
    for i in 1:npoints
	    bmat[i,:] = evaluatebi(G, big_eval_points[i], basis, n, k)
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
    #gradP = expectedvalue(evalpoints, theta, r, coeffss, weights)
    dtheta .= W*gradP
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
    p = reshape(p, repeat([N], k)...)
    #re-normalize p using quadrature rule
    #T = quad_int(p, wvec)
    #p = p ./ T
    elementwisemult = p .* wvec
    #compute differential change
    IA = collect(1:2*k)
    IB = collect([k + 1; 2 * k])
    IC = collect(1:k)
    tmp = tensorcontract(IC, F,
        IA, elementwisemult,
        IB)

    dpdt .= vec(p .* tmp)
    return dpdt
end

function main()
    n = 4
    k = 2
    r = n^k
    N = 100 #integration resolution
    tspan = (1, 1e2) #integration time span
    initial_parameters = repeat([1.0], r)
    evalpoints = collect(range(-1, 1, N))
    a, b = (-1, 1) #trait domain
    #resolution window size
    deltax = (b - a) / (N - 1)
    wvec = get_weights(deltax, N)
    A = sampleA(n, k)
    V = get_vander(evalpoints, n)
    F = buildF(V, A, k)
    #construct F
    T = buildT(n)
    B = computeB(A, T, k)

    ##reshsape B
    Blong = reshape(B, (n^k, n^k))

    ##do SVD
    Q = computeQ(Blong, r)

    ##get basis of monomials
    monomialbasis = getbasispx(n)
    legendrebasis = getlegendre(T, monomialbasis)
    ##grid of points evaluating each b
    G = buildG(Q, n, k)

    big_wvec = deepcopy(wvec)
    big_eval_points = deepcopy(evalpoints)
    for i=2:k
        big_eval_points = Iterators.product(big_eval_points, evalpoints)
        big_wvec = map(prod, Iterators.product(big_wvec, wvec) |> collect )
    end
    big_eval_points = big_eval_points |> collect
    bmat = evaluatebvec(G, legendrebasis, big_eval_points, n, r, k)
    big_wvec = vec(big_wvec)
    
    W = buildW(r)
    par_lat = (W, bmat, big_wvec)
    problemlatent = ODEProblem(re!, initial_parameters, tspan, par_lat)
    println("Integrating in latent space ")
    sollatent = DifferentialEquations.solve(problemlatent, progress=true, progress_steps=10)


    par_ori = (F, wvec, N, k) #vector of parameters
    p0 = popdist(initial_parameters, bmat)
    #set up ODE problem and solve it
    problemdisc = ODEProblem(re_discrete!, p0, tspan, par_ori)
    println("Integrating in original space")
    soldisc = DifferentialEquations.solve(problemdisc, progress=true, progress_steps=10)
    return sollatent, bmat, soldisc, n, k, N
end
sollatent, bmat, soldisc, n, k, N = main()

disc_solution = reshape(soldisc.u[end],repeat([N], k)...)
lat_solution = reshape(popdist(sollatent.u[end],bmat), repeat([N], k)...) 
