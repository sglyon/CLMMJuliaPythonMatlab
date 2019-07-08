# NewKeynesian

MATLAB, Python and Julia codes for a stylized new Keynesian model

## Installation instructions

### Python

In order to run the codes in this file you will need to install and
configure a few Python packages. We recommend following the instructions
on
[quantecon.org](https://lectures.quantecon.org/jl/getting_started.html)
for getting a base python installation set up. Then to acquire
additional packages used for this project, please run the following commands
at your terminal or command line prompt:

```shell
pip install git+https://github.com/EconForge/interpolation.py.git
pip install git+https://github.com/naught101/sobol_seq.git
pip install requests
```

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