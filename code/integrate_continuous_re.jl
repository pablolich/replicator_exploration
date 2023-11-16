using LegendrePolynomials #for evaluating legendre polynomials
using QuadGK #for numerical integration of a function
using DifferentialEquations #integrate ODEs
using LinearAlgebra #to build rotation matrices
using FiniteDifferences #to calculate gradients numerically
using Random
using Distributions #to go from parameters to distributions
using Polynomials, SpecialPolynomials, PolynomialMatrices #to change basis from standard to legendre

###########################################################################################
#INTEGRATION IN LATENT SPACE
###########################################################################################

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
    popdist(x, theta, r, coefficients)

compute the numerical value of the population distribution at x, theta
"""
function popdist(x, theta, r, coefficients)
    u = getu(x, theta, r, coefficients)
    return exp(u)
end

"""
    get_popdist_samples(evalpoints, theta, r coeffs)

compute the numerical value of the population distribution at multiple points
"""
function get_popdist_samples(evalpoints, theta, r, coeffs)
    [popdist(i, theta, r, coeffs) for i in evalpoints]
end

"""
Get weights for quadrature integration
"""
function get_weights(deltax, N)
    wvec = deltax*ones(N)
    wvec[1] = 0.5*wvec[1]
    wvec[end] = 0.5*wvec[end]
    return wvec
end

"""
Perform quadrature integration
"""
function quad_int(samples, weights)
    return dot(samples, weights)
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
    r, evalpoints N, coeffs = p
    W = buildW(r)
    #write function to calculate total population
    #integrate samples of population density for quadrature weights
    #population density are computed as a function of theta
    P(theta) = quad_int(get_popdist_samples(evalpoints, theta, r, coefficients), weights)
    #evaluate gradient
    gradP = grad(central_fdm(5, 1), P, theta)[1] #
    dtheta .= W*gradP
    return dtheta
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
#INTEGRATION IN DISCRETIZED ORIGINAL SPACE
###########################################################################################

function get_vander(xvec, n)
    Vmat =  collect(ntuple(i -> xvec::Array{Float64,1} .^ i::Int64, n))
    return transpose(mapreduce(permutedims, vcat, Vmat))
end

function get_F(V, A)
    return V*A*transpose(V)
end

function normalize(density_vec, weights)
    #get normalizing factor (half weight in the boundaries)
    #apply trapezoidal integration to the set of points, and divide by that
    T = quad_int(density_vec, weights)
    return density_vec/T
end

function discretize(dist, point_eval, weights)
    #evaluate density of dist with moments moment_vec at point_eval
    pvec = [pdf(dist, i) for i in point_eval] #height of points interpolated with linear spline
    #normalizing factor
    T = quad_int(pvec, weights)
    return pvec/T
end

function re_discrete!(dpdt, p, pars, t)
    #unpack parameters: 
    #payoff coefficients (F), 
    #integration weights (wvec)
    F, wvec = pars
    #re-normalize p using quadrature rule
    T = quad_int(p, wvec)
    p = p/T
    #compute differential change
    dpdt = diagm(p)*(F*diagm(p)*wvec)
    return dpdt
end


###########################################################################################
#COMPARE THE TWO APPROACHES
###########################################################################################

#general parameters for the two approaches
seed = 1
rng = MersenneTwister(seed)
r = 2 #rank of the approximation 
n = 2 #degree of polynomials in original basis
a, b = (-1, 1) #trait domain
N = 1000 #integration resolution
#resolution window size
deltax = (b-a)/(N-1)
#write vector of points where to evaluate distribution
evalpoints = collect(range(-1,1, N))
#get weights of quadrature integration
wvec = get_weights(deltax, N)
tspan = (1, 1e3) #integration time span
#sample coefficients of polynomial in standard basis of monomials
A = sampleA(n) #IS THIS TRUE?
#initial distribution
initial_moments = [0, 1]

#specific parameters for integration in latent space
par_lat = (r, evalpoints N, A)
#set up ODE problem and solve it
problem = ODEProblem(re!, initial_moments, tspan, par_lat)
sol = DifferentialEquations.solve(problem, Tsit5()) #what is this doing? Do same time discretization as this method

#specific parameters for integration in original space
#set a gaussian distribution with mean mu and variance sigma
dist = Normal(initial_moments[1], initial_moments[2])
#initial conditions w0 (discretize initial distribution)
p0 = discretize(dist, evalpoints, wvec)
#compute vandermonde matrix of monomial basis
V = get_vander(evalpoints, n)
F = get_F(V, A)
par_ori = (F, wvec) #vector of parameters
#set up ODE problem and solve it
problem = ODEProblem(re_discrete!, p0, tspan, par_ori)
sol = DifferentialEquations.solve(problem, Tsit5())


"""
Evaluate distribution with given moments at given evaluation points
"""
function moments2densities(evalpoints, moment_vec, dist)
end

"""
Create vector discretized distributions for each time step
"""
function multiplediscretizations(evalpoints, moment_mat, tvec)
end

"""
Compare two vectors using some norm
"""
function comparevector(vector1, vector2, norm)
end

"""
Compare distributions at each time step
"""
function comparedynamics(solution1, solution2, norm)
end
