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
function getu(x, theta, r, A)
    T = buildT(r)
    Q = A2Q(A, r)
    b_vec = evaluatebi(Q, T, getbasispx(r), x, r)
    u = dot(b_vec, theta)
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
    A = sampleA(r)
    parameters = (r, a, b, N, A) #vector of parameters
    #set up the problem and solve it
    problem = ODEProblem(re!, initial, tspan, parameters)
    sol = DifferentialEquations.solve(problem, Tsit5())
    return sol
end

"""
    thetas2pi(x, theta, r, coefficients)

provide value of distribution at x given the moments theta
"""
function thetas2pi(x, theta, r, coefficients)
    return popdist(x, theta, r, coefficients)
end

### Code to get at the bs
#1. Sample A as skew symmetric matrix of coefficients of original basis
function sampleA(spandimension)
    randmat = rand(spandimension, spandimension)
    A = 1/2*(randmat-transpose(randmat))
    return A
end

#2. Construct T: The rows of T are the coefficients of the legendre polynomials
function getlegendrecoeffs(order)
    coeffbin = zeros(order)
    coeffbin[order] = 1
    p = Legendre(coeffbin) 
    #transform to standard basis of monomials
    pst = convert(Polynomial, p)
    #get legendre coefficients
    lcoeffs = coeffs(pst)
    return lcoeffs
end

function append0s(vectofill, finaldimension)
    return append!(vectofill, zeros(finaldimension-length(vectofill)))
end

function buildT(spandimension)
    #initialize T
    T = zeros(spandimension, spandimension)
    for n in 1:spandimension
        lcoeffn = getlegendrecoeffs(n)
        tostore = append0s(lcoeffn, spandimension)
        T[n,:] = tostore
    end
    return T
end

#3. Write code to invert T efficiently, since it is a triangular  matrix
function invertT(T)
    #create a triangular matrix
    UTT = LowerTriangular(T)
    return inv(UTT)
end

#4. Compute B by doing T-1AT
function computeB(A, T)
    return invertT(T)*A*T
end

#5. Compute Q by doing the SVD of B, as Q = U sqrt(S)
function computeQ(B)
    U, S, V = svd(B)
    return U*sqrt.(diagm(S))
end
#Wrapper for all these functions
function A2Q(A, spandimension)
    T = buildT(spandimension)
    B = computeB(A, T)
    return computeQ(B)
end

function coeffbinary(term, order)
    vec = zeros(order)
    vec[term] = 1
    return vec
end
#6. Use b(x) in the integration pipeline
function getbasispx(spandimension)
    return [Polynomial(coeffbinary(i, spandimension)) for i in 1:spandimension]
end
function getbi(Q, T, basis)
    return Q*T*basis
end
function evaluatebi(Q, T, basis, x, spandimension)
    pxeval =  [basis[i](x) for i in 1:spandimension]
    return getbi(Q, T, pxeval)
end

##function to solve replicator directly in original space
    ###discretize distribution and do numerical integration there
    ###sample from current distribution to get expected value with montecarlo integration
