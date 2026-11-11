"""
Quantum Reservoir Computing module, contains every submodules and functions used in the project.
- `Qrcutils`: contains the functions to compute the Quantum Reservoir procedure, and in general the functions to compute and execute the hamiltonians.
- `ExperimentUtils`: contains execution support functions.
- `HardwareUtils`: contains the functions to submit the task to the hardware, and to retrieve the results.
"""

module qrc_module
#include("utils.jl")
#include("experiments.jl")
include("qrcutils.jl")

#using .Qrcutils
#using .HardwareUtils
#using .ExperimentUtils

#export Qrcutils, HardwareUtils, ExperimentUtils
export Qrcutils
end
