# Reduce the bitstrings sampled by the hardware backend to features at other shot counts.
#
#   julia --project=. -t auto quantum_evolution/reduce_bitstrings.jl \
#       input=results/emulation/<stem>_hwbits.jld2 shots=10,20,40,100,240 \
#       output=results/emulation/<stem>_hwshots_{shots}.csv
#
# Features at n shots use the first n of the stored samples of each datapoint and
# readout time, so the counts of one sweep are nested subsamples and the result at
# n equals what run_reservoir.jl writes for shot_counts containing n. Output
# layout as for every reservoir output: one column per datapoint, named by ID.

include("cli.jl")

using qrc_module
using CSV, DataFrames, JLD2

const DEFAULTS = Dict{String, Any}(
    "input" => "",      # *_hwbits.jld2 written by run_reservoir.jl (backend = "hardware")
    "shots" => [240],   # shot counts, each at most the number of stored samples
    "output" => "",     # output path; `{shots}` is replaced by the shot count
)

function main()
    s, _ = load_settings(DEFAULTS)
    shots = Int.(as_list(s["shots"]))
    s["shots"] = shots
    isempty(s["input"]) && error("No input given: pass input=<file>_hwbits.jld2")
    isempty(s["output"]) && error("No output given: pass output=PATH, with {shots} if several shot counts are requested")
    length(shots) > 1 && !occursin("{shots}", s["output"]) && error("Several shot counts requested but `output` has no {shots} placeholder")

    d = load(s["input"])
    ids, bits, nsites = d["ids"], d["bits"], d["nsites"]
    @info "Loaded $(size(bits, 3)) datapoints × $(size(bits, 2)) readouts × $(size(bits, 1)) samples from $(s["input"])"
    for n in shots
        cols = Vector{Vector{Float64}}(undef, size(bits, 3))
        Threads.@threads for k in 1:size(bits, 3)
            cols[k] = Qrcutils.shot_features(view(bits, :, :, k), nsites, n)
        end
        out = replace(s["output"], "{shots}" => string(n))
        mkpath(dirname(abspath(out)))
        CSV.write(out, DataFrame(reduce(hcat, cols), string.(ids)))
        Qrcutils.write_manifest(replace(out, r"\.csv$" => "") * ".manifest.json", s;
            shots = n, output = out, inputs = Dict(s["input"] => file_sha256(s["input"])))
        @info "Wrote $(length(cols[1])) features × $(length(cols)) datapoints at $n shots to $out"
    end
end

main()
