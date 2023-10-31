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

replicator equation to integrate in parameter space
"""
function re!(dtheta, theta, p, t)
    r, a, b, N, coeffs = p
    W = buildW(r)
    #write gradient function
    P(theta) = totalpopulation(popdist, a, b, N, theta, r, coeffs)
    #evaluate gradient
    gradP = grad(central_fdm(5, 1), P, theta)[1] #
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
    #initial = [0, 0]
    #initial = [0, 1]
    tspan = (1, 1e3) #integration time span
    N = 1000 #integration resolution
    #sample coefficients of normal polynomial
    #get coefficients of Legendre basis
    A = sampleA(r)
    parameters = (r, a, b, N, A) #vector of parameters
    #set up the problem and solve it
    problem = ODEProblem(re!, initial, tspan, parameters)
    sol = DifferentialEquations.solve(problem, Tsit5()) #what is this doing? Do same time discretization as this method
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
    p = Legendre(coeffbin) #ARE THIS NORMALIZED?
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
    U, S, V = svd(B) #check if I can outputing only U and S cuts signfificant time
    return U*sqrt.(diagm(S))
end
#Wrapper for all these functions
function A2Q(A, spandimension) #feed T as argument
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
function getbi(Q, T, basis) #basis* = legendre basis of polynomials instead of monomials
    return Q*T*basis #return Q*basis*
end
function evaluatebi(Q, T, basis, x, spandimension)
    pxeval =  [basis[i](x) for i in 1:spandimension]
    return getbi(Q, T, pxeval)
end

###########################################################################################
#TEST CODE
###########################################################################################
#what should f be? the product of f(x, y) and pi. 
function evaluatefxy(A)
    return fxy
end

function discretize(dist, moment_vec, point_eval)
    #evaluate density of dist with moments moment_vec at point_eval
    #normalize (half weight in the boundaries)
    return dist_vec

#what distribution do I choose?
        #random variance
        #uniform 
        #in the latent space it is [0,0]
        #scale by total population (add constant (1/widthofinterval) to basis functions)
    #how do I ensure normalization?
    #nomalize giving half weight to boundary points
    #we have to match this with the approximation

function growthrate(A, x, y, w)
    #evaluate fxy at (x, y)
    fxy = x*A*y
    #multiply by w
    
    return
end

#function to calculate total population by direct integration in discretized space
function int_growth_rate(f, a, b, N, r)
    #WHY IS r APPEARING HERE IF I AM NOT DOING AN APPROXIMATION?
    h = (b-a)/N
    int = h * ( f(a, r) + f(b, r) ) / 2
    #integrate numerically over y
    for k=1:N-1
        yk = (b-a) * k/N + a
        int = int + h*f(yk, r)
    end
    return int
end

function test_integration_discrete()
    #set seed for reproducibility
    seed = 1
    rng = MersenneTwister(seed)
    #SHOULD THIS BE R? BUT WHY, IF R IS FOR THE APPROXIMATION?
    n = 2 #number of players
    a, b = (-1, 1) #domain
    N = 1000 #integration resolution
    tspan = (1, 1e3) #integration time span
    #write vector of points where to evaluate distribution
    evalpoints = collect(range(-1,1, N))
    #moments of distribution
    mu=0
    sigma=1
    #set a gaussian distribution with mean mu and variance sigma
    dist = Normal(mu, sigma)
    #initial conditions w0 (discretize initial distribution)
    w0 = discretize(dist, moments, evalpoints)
    A = sampleA(n)
    parameters = (A, moments, dist, a, b, N, r) #vector of parameters
    #set up the problem and solve it
    problem = ODEProblem(re_discrete!, w0, tspan, parameters)
    sol = DifferentialEquations.solve(problem, Tsit5()) #what is this doing? Do same time discretization as this method
    return sol
end

function re_discrete!(dw, w, p, t)
    #unpack parameters (payoff coefficients, vector of moments, distribution,
    #evaluation sparsity)
    A, moments, dist, a, b, N, r = p
    #calculate dwdt
    dw = w*int_growth_rate(growt_rate, a, b, N, r)
    return dw
end
###discretize distribution and do numerical integration there
###sample from current distribution to get expected value with montecarlo integration