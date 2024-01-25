using Random
using LegendrePolynomials #for evaluating legendre polynomials
using Polynomials, SpecialPolynomials #to change basis from standard to legendre
using TensorOperations #used to compute B
using LinearAlgebra
using Permutations, Kronecker #to do Andrew's magic

function sampleA(n, k)
    randT = rand(repeat([n], 2*k)...)
    permutevec = [k+1:2*k; 1:k]
    A = 1/2*(randT - permutedims(randT, permutevec))
    return A
end

function append0s(vectofill, finaldimension)
    return append!(vectofill, zeros(finaldimension-length(vectofill)))
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
	T[n,:] = tostore
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
	IB = collect([0;i])
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
	if i <= r/2
	    vec[i] = 2*i
	else
	    vec[i] = 2*(r-i+1)-1
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
    filter = I(n)[setdiff(1:end, (r÷2+1):(n-r÷2)),:]
    indsort = sortperm(imag(eigens))
    Psort = Matrix(Permutation(indsort))
    Usorted = U*Psort
    eigenssorted = transpose(Psort)*eigens
    Usfilt = Usorted*transpose(filter)
    eigensfilt = filter*eigenssorted
    P = buildP(r)
    M = 1/sqrt(2) .* [1 -1im; 1 1im]
    #return V
    V = Usfilt*P*(I(r÷2) ⊗ M)
    Q = V*sqrt.(diagm(abs.(transpose(P)*eigensfilt)))
    return real(Q)
end

function getbasispx(spandimension)
    return [Polynomial(coeffbinary(i, spandimension)) for i in 1:spandimension]
end

function getlegendre(T, basis) 
    return T*basis #return Q*basis*
end

function evaluatebvec(G, x, basis, n, k)
    L = Array{Float64}(undef, n, k)
    for polyi in 1:n
    	for pointj in 1:k
	    L[polyi, pointj] = basis[polyi](x[pointj])
    	end
    end
	
    tensor_L = L[:,1]
    for i in 2:k
    	tensor_L = tensorproduct(1:i,tensor_L, 1:i-1, L[:,i], [i])
    end
    return tensorcontract(G, 0:k, tensor_L, 1:k)
end

function buildG(Q, n, k)
    dimensions = [n^k ; repeat([n], k)]
    F = Array{Float64}(undef, dimensions...)
    inds = collect(Iterators.product(Iterators.repeated(1:n, k)...))
    iterator = map(collect,reduce(vcat, inds))
    for i in 1:n^k
    	ivec = iterator[i]
    	j = (ivec .- 1)' * n .^ collect(k-1:-1:0) + 1
    	F[j,ivec...] = 1
    end
    return tensorcontract(Q, [0,1], F, [0;2:k+1])
end

function coeffbinary(term, order)
    vec = zeros(order)
    vec[term] = 1
    return vec
end

function getbasispx(spandimension)
    return [Polynomial(coeffbinary(i, spandimension)) for i in 1:spandimension]
end

function main()
    n=4
    k=3
    r=n^k
    N = 100 #integration resolution
    evalpoints = collect(range(-1,1, N))
    xeval = randn(k)
    A = sampleA(n, k)
    T = buildT(n)
    B = computeB(A, T, k)

    #reshsape B
    Blong = reshape(B, (n^k, n^k))

    #do SVD
    Q = computeQ(Blong, r)

    #get basis of monomials
    monomialbasis = getbasispx(n)
    legendrebasis = getlegendre(T, monomialbasis)
    #grid of points evaluating each b
    G = buildG(Q, n, k)
    bvec = evaluatebvec(G, xeval, legendrebasis, n, k)
    return bvec
end

main()
