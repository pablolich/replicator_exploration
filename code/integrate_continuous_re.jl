using LegendrePolynomials #for evaluating legendre polynomials
using QuadGK #for numerical integration of a function
using DifferentialEquations #integrate ODEs
using LinearAlgebra #to build rotation matrices

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

function totalpopulation(x, theta, r)
    u = getu(x, theta, r)
    return exp(u)
end

function quad_trap(f, a, b, N, theta, r) 
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

function getgradP()
end

function re!(dtheta, theta, p, t)
    r, a, b, N = p
    W = getW()
    P = quad_trap(totalpopulation, a, b, N, theta, r)
    #get gradient of P
    gradP = getgradP(P, theta)
    dtheta .= W*gradP
    return dtheta
end