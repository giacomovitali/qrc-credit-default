# Classical (mean-field) reservoir on the 12 aggregated features of both groups.
#
#   julia --project=. classical_rc/CRC.jl [output_dir]      (default: dataset_aquila)
#
# Writes <output_dir>/{qpu,emulation}_samples_crc_labeled.csv: one column per datapoint, named by ID.

using Pkg
Pkg.activate(".")
Pkg.instantiate()
include("utils.jl")

# import required libraries
using OneHotArrays
using Flux
using Bloqade
using CSV
using DataFrames
using Random

output_dir = isempty(ARGS) ? "dataset_aquila" : ARGS[1]
mkpath(output_dir)

Xqpu = DataFrame(CSV.File("dataset_aquila/qpu_samples.csv", header=true))
ids_qpu = string.(Xqpu.ID)
DataFrames.select!(Xqpu, Not(:ID))
Xemu = DataFrame(CSV.File("dataset_aquila/emulation_samples.csv", header=true))
ids_emu = string.(Xemu.ID)
DataFrames.select!(Xemu, Not(:ID))

# convert the dataframes into arrays
Xqpu = Matrix(Xqpu)'
Xemu = Matrix(Xemu)'

# rescale the dataset features to feasible values of detuning
Δ_max = 4.5
spectral = max(abs(maximum(Xemu)), abs(minimum(Xemu)))
Xemu = Xemu/spectral * Δ_max
Xqpu = Xqpu/spectral * Δ_max

d = 10.0
nsites=size(Xemu,1)

locs = [[i*d,0.0] for i in 1:nsites] # put atoms in a chain with 10 micron spacing
u0 =zeros(3*nsites)
for i in 1:nsites
    u0[3*i] = -1 
end

pre_layer = ClassicalDetuningLayer(;
    atoms=locs, 
    Ω = 2π, 
    t_start = 0.0, 
    t_end = 2.5, 
    step = 0.5, 
    state=u0,
    readout="zz"
);

embeddings_emu_crc = apply_classical_layer(pre_layer, Xemu)
embeddings_qpu_crc = apply_classical_layer(pre_layer, Xqpu)

CSV.write(joinpath(output_dir, "emulation_samples_crc_labeled.csv"), DataFrame(embeddings_emu_crc, ids_emu))
CSV.write(joinpath(output_dir, "qpu_samples_crc_labeled.csv"), DataFrame(embeddings_qpu_crc, ids_qpu))
