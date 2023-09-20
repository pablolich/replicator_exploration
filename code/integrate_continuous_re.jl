using LegendrePolynomials #for evaluating legendre polynomials
using QuadGK #for numerical integration of a function
using DifferentialEquations #integrate ODEs
using LinearAlgebra #to build rotation matrices
using FiniteDifferences #to calculate gradients numerically

function getu(x, theta, r)
    u = 0
    for i in 1:r
        #evaluate legendre polynomial of degree i at x
        b_i = Pl(x, i) #divide by sqrt
        #calcule ith term
        u += theta[i]*b_i
    end
    return u
end

function popdist(x, theta, r)
    u = getu(x, theta, r)
    return exp(u)
end


function totalpopulation(f, a, b, N, theta, r) 
    h = (b-a)/N
    int = h * ( f(a, theta, r) + f(b, theta, r) ) / 2
    for k=1:N-1
        xk = (b-a) * k/N + a
        int = int + h*f(xk)
    end
    return int
end

function buildW(r::Int64)
    #build rotation matrix
    R = [0 1;-1 0]
    #initialize W
    W0 = [zeros(Int64, 2, 2) for i in 1:r, j in 1:r]
    W0[diagind(W0)] .= [R]
    #construct W
    W = reduce(vcat, [reduce(hcat, W0[i, :]) for i in 1:r])
    return W
end

function re!(dtheta, theta, p, t)
    r, a, b, N = p
    W = getW(r)
    #write gradient function
    P(theta) = totalpopulation(popdist, a, b, N, theta, r)
    #evaluate gradient
    gradP = grad(central_fdm(5, 1), P, theta)[1]
    dtheta .= W*gradP
    return dtheta
end

###
###
##get the ODE running
##thetas to distributions
##function to solve replicator directly in original space
    ###discretize distribution and do numerical integration there
##realistic f (project to legendere polynomials)--> get matrix of change of basis
