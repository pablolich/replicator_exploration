using LegendrePolynomials #for evaluating legendre polynomials
using QuadGK #for numerical integration of a function
using DifferentialEquations #integrate ODEs
using LinearAlgebra #to build rotation matrices
using FiniteDifferences #to calculate gradients numerically
using Random
using Distributions #to go from parameters to distributions
using Polynomials, SpecialPolynomials, PolynomialMatrices #to change basis from standard to legendre

###########################################################################################
#FUNCTIONS USED ACCROSS ALL THE CODE
###########################################################################################

#sample coefficients of polynomial in standard basis of monomials
function sampleA(spandimension)
    randmat = rand(spandimension, spandimension)
    A = 1/2*(randmat-transpose(randmat))
    return A
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

###########################################################################################
#FUNCTIONS FOR INTEGRATION IN LATENT SPACE
###########################################################################################

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

#5. Compute Q by doing the SVD of B, as Q = U sqrt(S)
function computeQ(B)
    U, S, V = svd(B) #check if I can outputing only U and S cuts signfificant time
    return U*sqrt.(diagm(S))
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

#Wrapper for all these functions
function A2Q(A, spandimension, T)
    B = computeB(A, T)
    return computeQ(B)
end

function append0s(vectofill, finaldimension)
    return append!(vectofill, zeros(finaldimension-length(vectofill)))
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

"""
    getu(x, theta, r)

compute numerical value of the function u given a value, a rank, and the parameters
"""
function getu(x, theta, r, A)
    T = buildT(r)
    Q = A2Q(A, r, T)
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
    r, evalpoints, N, coeffs, weights = p #coeffs=A
    W = buildW(r)
    #write function to calculate total population
    #integrate samples of population density for quadrature weights
    #population density are computed as a function of theta
    P(theta) = quad_int(get_popdist_samples(evalpoints, theta, r, coeffs), weights)
    #evaluate gradient
    gradP = grad(central_fdm(5, 1), P, theta)[1] #
    #in case there is a bug, try expected values
    dtheta .= W*gradP
    return dtheta
end

###########################################################################################
#FUNCTIONS FOR INTEGRATION IN DISCRETIZED ORIGINAL SPACE
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
#FUNCTIONS TO COMPARE THE TWO INTEGRATIONS
###########################################################################################

"""
Create vector discretized distributions for each time step
"""
function multiplediscretizations(evalpoints, par_mat, r, coefficients)
    ntpoints = size(par_mat, 1)
    nevals = length(evalpoints)
    dist_mat = Array{Float64}(undef, (ntpoints, nevals))
    for i in 1:ntpoints
        #get vector of moments
        par_vec_i = par_mat[i,:]
        #evaluate distribution
        density_vec_i = get_popdist_samples(evalpoints, par_vec_i, r, coefficients) #coefficients = A
        #store
        dist_mat[i,:]
    end
    return dist_mat
end

"""
Compare two vectors using some norm
"""
function comparevectors(vector1, vector2, lnorms)
    return [norm(vector1-vector2, lnorms[i]) for i in 1:length(lnorms)]
end

"""
Compare distributions at each time step
"""
function comparedynamics(solution1, solution2, norm_vec)
    #get number of rows of solutions
    tsteps = size(solution1, 1)
    nnorms = length(norm_vec)
    #initialize storing matrix
    norm_mat = Array{Float64}(undef, (tsteps, nnorms))
    for i in 1:tsteps
        #get solutions at time i
        solution1ti = solution1[i,:]
        solution2ti = solution2[i,:]
        #compare them with each norm
        norm_mat[i,:] = comparevectors(solution1ti, solution2ti, [1,2,Inf])
    end
    return norm_mat
end

"""
plotting
distributions
space points
norms
"""

#COMPARE WITH DIFFERETN DISCRETIZATION FINNESS

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
A = sampleA(n) #IS THIS TRUE? 
#initial distribution
initial_moments = [1, 1]

#try n large, and replace A with SVD approximation of rank r
#talk to klausmeier
#dynamic fitness landscapes


#specific parameters for integration in latent space
par_lat = (r, evalpoints, N, A, wvec)
#set up ODE problem and solve it
problemlatent = ODEProblem(re!, initial_moments, tspan, par_lat)
sollatent = DifferentialEquations.solve(problemlatent) #what is this doing? Do same time discretization as this method
 
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
problemdisc = ODEProblem(re_discrete!, p0, tspan, par_ori)
soldisc = DifferentialEquations.solve(problemdisc)

#evaluate dense solutions at same timepoints
if length(soldisc.t)<length(sollatent.t)
    #evaluate soldisc at the times of sollatent and reshape
    solution2 = mapreduce(permutedims, vcat, soldisc(sollatent.t).u)
    #discretize latent space solution
    sollatent_mat = mapreduce(permutedims, vcat, sollatent.u)
    solution1 = multiplediscretizations(evalpoints, sollatent_mat, r, A)
else
    #evaluate sollat at the times of solldisc
    sollat_tdisc = sollat(soldisc.t)
    #discretize solution in latent space on the distribution space, evaluated at the time points of the discrete
    sollat_tdisc_mat = mapreduce(permutedims, vcat, sollat_tdisc.u) #first, reshape
    solution1 = multiplediscretizations(evalpoints, sollat_tdisc_mat, r, A)
    solution2 = mapreduce(premutedims, vcat, soldisc.u) #reshape
end

#compare solutions
norm_mat = comparedynamics(solution1, solution2, [1, 2, Inf])
#plot