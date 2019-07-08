module NKModel

#%% cell
#=
# Solving a New Keynesian model with Julia

This file is part of a computational appendix that accompanies the paper.

> MATLAB, Python, Julia: What to Choose in Economics?
>
> Coleman, Lyon, Maliar, and Maliar (2017)


For some details regarding the model solved in this file, please see the
[companion notebook that describes the model](http://bookshelf.quantecon.org/submission/59fa1b45145fc3772b0cef82).

In order to run the codes in this file you will need to install and configure a
few Julia packages. We recommend downloading
[JuliaPro](https://juliacomputing.com/products/juliapro.html) and/or following
the instructions on
[quantecon.org](https://lectures.quantecon.org/py/getting_started.html).

Once your Julia installation is up and running, there are a few additional packages
you will need in order to run the code here. We use the
[InstantiateFromURL](https://github.com/QuantEcon/InstantiateFromURL.jl) pacakge to make
it easy to get all required packages set up in your Julia environment.  If you have not
previously installed the `InstantiateFromURL` package you will need to uncomment the line
`pkg"add InstantiateFromURL"` and run the cell below.
=#

#%% cell
using Pkg
# pkg"add InstantiateFromURL"
using InstantiateFromURL: activate_github_path
activate_github_path("sglyon/CLMMJuliaPythonMatlab", path="NewKeynesian/julia", activate=true, force=true)

#%% cell
#=
## Julia Code

The Julia version of our algorithm is implemented as a few functions defined on
a core type named `Model`. This type is itself composed of three different types
that hold the model parameters, steady state, and grids needed to describe the
numerical model. Before we get to the types, we need to bring in some
dependencies:
=#

#%% cell
# for plotting the output
using PlotlyJS

# for constructing Sobol sequences
using Sobol
# for basis matrix of complete monomials and monomial quadrature rules
using BasisMatrices, QuantEcon
using MAT
using Parameters

# standard library components
using Printf  # printing messages
using Statistics  # computing `mean`
using Random: seed!
using LinearAlgebra: diagm, cholesky, dot
using InteractiveUtils: versioninfo
using DelimitedFiles: writedlm

seed!(42);  # set seed for reproducibility

include("model.jl")
include("solution.jl")

#%% cell
#=
### Running the code

Now that we've done all the hard work to define the model, its solution and
simulation, and accuracy checks, let's put things together and run the code!
=#

#%% cell
function ensurefile(url, localpath)
    if !isfile(localpath)
        println("Downloading $url to $localpath")
        download(url, localpath)
    end
end


function main(m=Model(); io::IO=stdout)
    ensurefile("https://github.com/sglyon/CLMMJuliaPythonMatlab/raw/master/NewKeynesian/Sobol_grids.mat", "Sobol_grids.mat")
    ensurefile("https://github.com/sglyon/CLMMJuliaPythonMatlab/raw/master/NewKeynesian/epsi_test_NK.mat", "epsi_test_NK.mat")
    ensurefile("https://github.com/sglyon/CLMMJuliaPythonMatlab/raw/master/NewKeynesian/random_grids.mat", "random_grids.mat")
    coefs, solve_time = solve(m);

    # simulate the model
    ts = time(); sim = Simulation(m, coefs); simulation_time = time() - ts

    # check accuracy
    tr = time(); resids = Residuals(m, coefs, sim); resids_time = time() - tr

    err_by_eq = max_E(resids)
    l1 = mean(resids)
    l∞ = max(resids)
    tot_time = solve_time + simulation_time + resids_time
    round3(x) = round(x, digits=3)
    round2(x) = round(x, digits=2)

    println(io, "Solver time (in seconds): $(solve_time)")
    println(io, "Simulation time (in seconds): $(simulation_time)")
    println(io, "Residuals time (in seconds): $(resids_time)")
    println(io, "total time (in seconds): $(tot_time)")
    println(io, "\nAPPROXIMATION ERRORS (log10):");
    println(io, "\ta) mean error in the model equations: $(round3(l1))");
    println(io, "\tb) max error in the model equations: $(round3(l∞))");
    println(io, "\tc) max error by equation:$(round3.(err_by_eq))");

    solve_time, simulation_time, resids_time, coefs, sim, resids

    t = 1:100
    p1 = plot([scatter(x=t, y=sim.S[t], name="S"), scatter(x=t, y=sim.F[t], name="F")],
              Layout(title="Figure 1a. S and F"))

    p2 = plot([scatter(x=t, y=sim.Y[t], name="Y"), scatter(x=t, y=sim.Yn[t], name="Yn")],
              Layout(title="Figure 1b. Output and natural output"))

    p3 = plot([scatter(x=t, y=sim.C[t], name="C"), scatter(x=t, y=sim.L[t], name="L")],
              Layout(title="Figure 1c. Consumption and labor"))

    p4 = plot([scatter(x=t, y=sim.δ[t], name="δ"), scatter(x=t, y=sim.R[t], name="R"),
               scatter(x=t, y=sim.π[t], name="π")],
              Layout(title="Figure 1d. Distortion, interest rate and inflation"))

    p = [p1 p2; p3 p4]
    p.plot.layout["width"] = 1000
    p.plot.layout["height"] = 600

    p, solve_time, simulation_time, resids_time, coefs, sim, resids, l1, l∞
end

#%% cell
function build_paper_table()
    # call main once to precompile all routines
    main()

    to_csv = []
    push!(to_csv, ["pi_star", "zlb", "degree", "solve_time", "l_1", "l_inf"])

    this_dir = @__DIR__

    open(joinpath(this_dir, "output.log"), "w") do f
        for params in [Dict(:πstar=>1.0, :zlb=>false),
                       Dict(:πstar=>1.0, :zlb=>true),
                       Dict(:πstar=>1 + 0.0598/4, :zlb=>false)]
            for deg in 1:5
                println("Starting $params and deg=$deg")
                m = Model(;grid_kind=:sobol, deg=deg, params...)

                println(f, "Starting $params and deg=$deg")
                solve_time, l1, linf = main(m, io=f)[[2, 8, 9]]
                push!(to_csv, [params[:πstar], params[:zlb], deg, solve_time, l1, linf])
                @show to_csv
                println(f, "\n"^5)
                flush(f)
            end
        end
        versioninfo(f, verbose=true)
    end

    writedlm(
        joinpath(this_dir, "output_osx.csv"),
        permutedims(hcat(to_csv...), (2, 1)), ','
    )
end


#%% cell
# if "table" in ARGS
#     build_paper_table()
# else
#     results = main()
#     results[1]  # show plot
# end

end  # module
