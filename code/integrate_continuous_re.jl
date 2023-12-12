    using LegendrePolynomials #for evaluating legendre polynomials
    using QuadGK #for numerical integration of a function
    using DifferentialEquations #integrate ODEs
    using LinearAlgebra #to build rotation matrices
    using FiniteDifferences #to calculate gradients numerically
    using Random #sample random numbers
    using Distributions #to go from parameters to distributions
    using Polynomials, SpecialPolynomials #to change basis from standard to legendre
    using Plots #plot norms between two solutions

    ###########################################################################################
    #FUNCTIONS USED ACCROSS ALL THE CODE
    ###########################################################################################

    #sample coefficients of polynomial in standard basis of monomials
    function sampleA(spandimension)
        #SET A TO BE THE SAME EVERY TIME
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
        return transpose(samples)*weights
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

    """
    orthogonal basis functions b(x) such that f(x,y) = b(x)' W b(y)
    """
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
        return transpose(invertT(T))*A*invertT(T) #CORRECTED FROM invertT(T)*A*T
    end

    #Wrapper for all these functions
    function A2Q(A, T)
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
        Q = A2Q(A, T)
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

    function func(Q, T, r, x, theta, coefficients)
        evaluatebi(Q, T, getbasispx(r), x, r)*popdist(x, theta, r, coefficients)
    end
    function expectedvalue(x, theta, r, coefficients, weights)
        T = buildT(r)
        Q = A2Q(A, T)
        samplesfvec = [func(Q, T, r, i, theta, coefficients) for i in x]
        samplesf = mapreduce(permutedims, vcat, samplesfvec)
        quad_int(samplesf, weights)
    end

    """
        re!(dtheta, theta, p, t)

    replicator equation to integrate in parameter space
    """
    function re!(dtheta, theta, p, t)
        r, evalpoints, N, coeffss, weights = p #coeffs=A
        W = buildW(r)
        #write function to calculate total population
        #integrate samples of population density for quadrature weights
        #population density are computed as a function of theta
        P(theta) = quad_int(get_popdist_samples(evalpoints, theta, r, coeffss), weights)
        #evaluate gradient
        gradP = grad(central_fdm(5, 1), P, theta)[1] #
        expectedvals = expectedvalue(evalpoints, theta, r, coeffss, weights)
        println(norm(gradP .- expectedvals, 2))
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

    function re_discrete!(dpdt, p, pars, t)
        #unpack parameters: 
        #payoff coefficients (F), 
        #integration weights (wvec)
        F, wvec = pars
        #re-normalize p using quadrature rule
        T = quad_int(p, wvec)
        p = p/T
        #compute differential change
        dpdt .= diagm(p)*(F*diagm(p)*wvec)
        return dpdt
    end

    ###########################################################################################
    #FUNCTIONS VISUALIZE/COMPARE OUTPUTS FROM THE TWO INTEGRATIONS
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
            #evaluate distribution and store
            dist_mat[i, :] = get_popdist_samples(evalpoints, par_vec_i, r, coefficients) #coefficients = A
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
    Plot the evolution of a distribution over time
    """
    function plotsolution(solutionmat, evalpoints, inds::Array{Int64, 1}, title::String)
        #loop over each line
        nlines = size(inds, 1)
        ntimes = size(solutionmat, 1)
        #select x points to plot
        xpoints = evalpoints[inds]
        #set palette
        palette_rb = cgrad(:thermal, ntimes, categorical=true)
        #plot first line
        p=plot(evalpoints, solutionmat[inds[1],:], c=palette_rb[inds[1]], 
            xlabel = "x",
            ylabel = "density",
            title = title)
        #add the rest
        for i in inds[2:end]
            plot!(evalpoints, solutionmat[i,:], c=palette_rb[i])
        end
        #display/save plot
        #savefig()
        return p
    end

    #COMPARE WITH DIFFERENT DISCRETIZATION FINNESS

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
    initial_parameters = repeat([1.0], r)

    #specific parameters for integration in latent space
    par_lat = (r, evalpoints, N, A, wvec)
    #set up ODE problem and solve it
    problemlatent = ODEProblem(re!, initial_parameters, tspan, par_lat)
    sollatent = DifferentialEquations.solve(problemlatent) #what is this doing? Do same time discretization as this method
    
    #specific parameters for integration in original space
    #set a gaussian distribution with mean mu and variance sigma
    p0 = get_popdist_samples(evalpoints, initial_parameters, r, A)
    #initial conditions w0 (discretize initial distribution)
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
        tpoints = sollatent.t
    else
        #evaluate sollat at the times of solldisc
        sollat_tdisc = sollatent(soldisc.t)
        #discretize solution in latent space on the distribution space, evaluated at the time points of the discrete
        sollat_tdisc_mat = mapreduce(permutedims, vcat, sollat_tdisc.u) #first, reshape
        solution1 = multiplediscretizations(evalpoints, sollat_tdisc_mat, r, A)
        solution2 = mapreduce(permutedims, vcat, soldisc.u) #reshape
        tpoints = soldisc.t
    end
    #solution1 --> solution in the latent space
    #solution2 --> solution in the discretized space

    #compare solutions
    norm_mat = comparedynamics(solution1, solution2, [1, 2, Inf])
    #plot solutions
    lineinds = collect(1:length(tpoints)) #length(tpoints)
    psol1 = plotsolution(solution1, evalpoints, lineinds, "latent")
    psol2 = plotsolution(solution2, evalpoints, lineinds, "discretized")
    plot(psol1, psol2, layout=2, 
        ylimits = (-0.1, 4.1),
        legend=false,
        size=(600, 300))
    savefig("../replicator_exploration/figures/integrations.pdf")
#plot distances
plot(1:size(norm_mat, 1), log.(norm_mat), label=["1" "2" "Inf"])


#THINGS TO DO
#try n large, and replace A with SVD approximation of rank r
#dynamic fitness landscapes