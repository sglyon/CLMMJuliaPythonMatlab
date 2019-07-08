# NewKeynesian

MATLAB, Python and Julia codes for a stylized new Keynesian model

## Installation instructions

### Python

To get up and running with the python version of the code you should follow
these steps:

- Download and install the [Anaconda python distribution](https://www.continuum.io/downloads) for python 3.5 (NOTE: python 2.x will not work). A download link and installation directions can be found at the linked website
- Download and install the `dolo` python package using the command `pip install dolo`.

The suggested way to run the python code is:

1. Start `ipython`  in the python directory of this repository. To do this navigate your command prompt or terminal shell to the python folder and run the commadn `ipython`.
2. Once inside IPython run the following commands: `run main`

### Julia

To get started with the Julia version, do the following:

- Download the current released version of Julia for your platform from the [Julia website](http://julialang.org/downloads/). Follow any installation instructions found at that site
- Start julia and enter the following commands (one at a time) from the `julia>` prompt:

```julia
using Pkg
pkg"add InstantiateFromURL"
using InstantiateFromURL: activate_github_path
activate_github_path("sglyon/CLMMJuliaPythonMatlab", path="NewKeynesian/julia", activate=true, force=true)
```

Then in the same Julia session run the following:


```julia
include("main.jl")
NKModel.main();
```