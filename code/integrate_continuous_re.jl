using LegendrePolynomials #for evaluating legendre polynomials
using QuadGK #for numerical integration of a function
using DifferentialEquations #integrate ODEs
using LinearAlgebra #to build rotation matrices
using FiniteDifferences #to calculate gradients numerically
using Random
using Distributions #to go from parameters to distributions
using Polynomials, SpecialPolynomials #to change basis from standard to legendre


"""
    getu(x, theta, r)

compute numerical value of the function u given a value, a rank, and the parameters
"""
function getu(x, theta, r, coefficients)
    u = 0
    for i in 1:r
        #evaluate legendre polynomial of degree i at x
        b_i = coefficients[i]*Pl(x, i)/sqrt(2/(2*i + 1))
        #calcule ith term
        u += theta[i]*b_i
    end
    return u
end

"""
    popdist(x, theta, r)

compute the numerical value of the population distribution at x, theta
"""
function popdist(x, theta, r, coefficients)
    u = getu(x, theta, r, coefficients)
    return exp(u)
end

"""
    totalpopulation(f, a, b, N, theta, r)

calculate total population by numerically integrating the population distribution
"""
function totalpopulation(f, a, b, N, theta, r, coeffs) 
    h = (b-a)/N
    int = h * ( f(a, theta, r, coeffs) + f(b, theta, r, coeffs) ) / 2
    for k=1:N-1
        xk = (b-a) * k/N + a
        int = int + h*f(xk, theta, r, coeffs)
    end
    return int
end

"""
    buildW(r::Int64)

build block-diagonal matrix of rotational matrices
"""
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

"""
    re!(dtheta, theta, p, t)

replicator equation to integrate
"""
function re!(dtheta, theta, p, t)
    r, a, b, N, coeffs = p
    W = buildW(r)
    #write gradient function
    P(theta) = totalpopulation(popdist, a, b, N, theta, r, coeffs)
    #evaluate gradient
    gradP = grad(central_fdm(5, 1), P, theta)[1]
    dtheta .= W*gradP
    return dtheta
end

function test_integration()
    seed = 1
    rng = MersenneTwister(seed)
    r = 2 #rank of the approximation 
    n = 2 #number of moments to describe the distribution
    a, b = (-1, 1) #trait domain
    initial = rand(n) #initial conditions
    tspan = (1, 1e3) #integration time span
    N = 1000 #integration resolution
    #sample coefficients of normal polynomial
    #get coefficients of Legendre basis
    ######################################################
    parameters = (r, a, b, N, coeffs) #vector of parameters
    #set up the problem and solve it
    problem = ODEProblem(re!, initial, tspan, parameters)
    sol = DifferentialEquations.solve(problem, Tsit5())
    return sol
end

function thetas2pi(theta, a, b, dx)
    d = Normal(theta[1], theta[2])
    dtrunc = truncated(d, a, b)
    pdfvec = pdf.(dtrunc, a:dx:b)/(sum(pdf.(dtrunc, a:dx:b)))
    return pdfvec
end

function getlegendrecoefficients(coefficients)
    #form polynomial in basis of monomials
    p = Polynomial(coefficients)
    pL = convert(Legendre, p)
    return coeffs(pL)
end

###

##function to solve replicator directly in original space
    ###discretize distribution and do numerical integration there
    ###sample from current distribution to get expected value with montecarlo integration
##realistic f (project to legendere polynomials)--> get matrix of change of basis
