# Reduce shot-based experiments to feature CSVs, one per shot count.
#
#   julia --project=. quantum_evolution/reduce_experiment.jl [CONFIG.toml] [key=value ...]
#
# Example (from the repository root):
#
#   julia --project=. quantum_evolution/reduce_experiment.jl \
#       experiments=data/experiments/QRC_12_emu_true_position_2026-09-24-10.00.00 \
#       shots=100,240,1000 seed=7 output=datasets/position_shots_{shots}shots.csv
#
# For each shot count, every datapoint's stored measurement records are
# subsampled to that many shots (seeded per datapoint, so the subsample of a
# datapoint does not depend on which other datapoints are reduced with it) and
# turned into ⟨Zᵢ⟩ and ⟨ZᵢZⱼ⟩ for each readout time. Several experiments —
# e.g. disjoint datapoint ranges run in parallel — are merged into one file.
#
# The output has one column per datapoint, named by datapoint ID, in ascending
# ID order. The classification stage aligns columns to labels by ID, so no
# separate reordering step is needed. A manifest is written next to each file.

include("cli.jl")

using qrc_module
using CSV, DataFrames

const DEFAULTS = Dict{String, Any}(
    "experiments" => String[],   # experiment directories to merge
    "shots" => [-1],             # shot counts; -1 = every stored measurement
    "seed" => 1234,              # subsampling seed
    "output" => "",              # output path; `{shots}` is replaced by the shot count
    "duplicates" => "error",     # datapoint in several experiments: "error", or "first" to keep the first listed
)

shots_label(n) = n > 0 ? string(n) : "all"

function main()
    s, _ = load_settings(DEFAULTS)
    dirs = string.(as_list(s["experiments"]))
    shots = Int.(as_list(s["shots"]))
    s["experiments"], s["shots"] = dirs, shots
    isempty(dirs) && error("No experiments given: pass experiments=DIR[,DIR...]")
    isempty(s["output"]) && error("No output given: pass output=PATH, with {shots} if several shot counts are requested")
    length(shots) > 1 && !occursin("{shots}", s["output"]) && error("Several shot counts requested but `output` has no {shots} placeholder")

    experiments = [Qrcutils.resume_experiment(d) for d in dirs]
    stored = unique(e.nshots for e in experiments)
    @info "Loaded $(length(experiments)) experiment(s), $(sum(length(e.tasks_id) for e in experiments)) datapoints, $(join(stored, "/")) stored shots per datapoint"
    for n in shots
        n > minimum(stored) && error("Requested $n shots but an experiment stores only $(minimum(stored)) per datapoint")
    end

    s["duplicates"] in ("error", "first") || error("duplicates must be \"error\" or \"first\"")
    seen = Dict{Int, String}()
    take = [Int[] for _ in experiments]          # datapoints taken from each experiment
    dropped = Dict{String, Vector{Int}}()
    for (k, (e, d)) in enumerate(zip(experiments, dirs)), id in sort(collect(keys(e.tasks_id)))
        if haskey(seen, id)
            s["duplicates"] == "error" && error("Datapoint $id appears in both $(seen[id]) and $d; pass duplicates=first to keep the first")
            push!(get!(dropped, d, Int[]), id)
            continue
        end
        seen[id] = d
        push!(take[k], id)
    end
    isempty(dropped) || @warn "Duplicated datapoints kept from the first experiment listing them" dropped

    for n in shots
        ids_all = Int[]
        blocks = Matrix{Float64}[]
        skipped_all = Int[]
        for (e, wanted) in zip(experiments, take)
            ids, F, skipped = Qrcutils.experiment_features(e; n_shots = n, seed = s["seed"], ids = wanted)
            append!(skipped_all, skipped)
            isempty(ids) && continue
            append!(ids_all, ids)
            push!(blocks, F)
        end
        isempty(blocks) && error("No datapoint of the given experiments has usable results")
        F = reduce(hcat, blocks)
        order = sortperm(ids_all)
        ids_all, F = ids_all[order], F[:, order]

        out = replace(s["output"], "{shots}" => shots_label(n))
        mkpath(dirname(abspath(out)))
        CSV.write(out, DataFrame(F, string.(ids_all)))
        Qrcutils.write_manifest(replace(out, r"\.csv$" => "") * ".manifest.json", s;
            shots = n, output = out, n_datapoints = length(ids_all), n_features_out = size(F, 1),
            skipped_datapoints = sort(skipped_all), duplicates_dropped = dropped)
        isempty(skipped_all) || @warn "$(length(skipped_all)) datapoint(s) had no usable results and were skipped; see the manifest"
        @info "Wrote $(size(F, 1)) features × $(length(ids_all)) datapoints at $(shots_label(n)) shots to $out"
    end
end

main()
