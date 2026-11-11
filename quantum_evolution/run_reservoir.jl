# Run the quantum reservoir on preprocessed feature sets, in emulation.
#
#   julia --project=. quantum_evolution/run_reservoir.jl CONFIG.toml [key=value ...]
#
# Examples (from the repository root):
#
#   julia --project=. -t auto quantum_evolution/run_reservoir.jl configs/reservoir/position_statevector.toml
#   julia --project=. quantum_evolution/run_reservoir.jl configs/reservoir/position_shots.toml first=1 last=1000
#
# Backends (what each one evolves: docs/architecture.md):
#   statevector  exact expectation values of the continuous pulse; writes one
#                feature CSV to `output_dir`. `readout_timing = "exact"` reads out
#                at exactly t_start+step, …, t_end; "integrator" reads out at the
#                integrator's step endpoints at or just after those times.
#   hardware     the Aquila protocol, noiseless and single-chain: a fresh program
#                per readout time with the waveforms executed on Aquila. Writes the exact
#                features, shot features at each of `shot_counts` (sampled from
#                the exact states, seeded per datapoint) and, optionally, the
#                sampled bitstrings for later reduction at other shot counts.
#   shots        record-based shot emulation (not used for the article's results), recorded as an
#                experiment under `experiment_root`; reduce it to features with
#                quantum_evolution/reduce_experiment.jl.
# The statevector and hardware backends distribute datapoints over Julia
# threads (start Julia with -t); results do not depend on the thread count.
#
# Hardware submission is deliberately not offered here; see docs/hardware.md.
# Every setting is listed in DEFAULTS below and documented in docs/pipeline.md.
# A JSON manifest recording the resolved settings, the git commit and input
# checksums is written next to every output.

include("cli.jl")

# Progress lines must reach a redirected log while the run is going: stderr is
# block-buffered when it is a file, so flush after every message.
progress(msg) = (@info msg; flush(stderr))

using qrc_module
using CSV, DataFrames
using Random
using Printf
using JLD2
import Dates

const DEFAULTS = Dict{String, Any}(
    "encoding" => "position",               # "position" | "detuning"
    "backend" => "statevector",             # "statevector" | "hardware" | "shots"
    "readout_timing" => "exact",            # statevector: "exact" | "integrator" (step-endpoint readouts)
    "shot_counts" => [240, 1000],           # hardware: shot counts to write features for
    "save_bitstrings" => true,              # hardware: also store the n_shots sampled bitstrings
    "input_dir" => "dataset_aquila",        # directory holding the preprocessed CSVs
    "sets" => ["qpu_samples"],              # input files (without .csv) to process
    "scale_reference" => "",                # set fixing the feature scaling; "" = first of `sets`
    "first" => 1,                           # datapoint range within each set, 1-based, inclusive
    "last" => -1,                           #   -1 = to the end of the set
    "n_shots" => 1000,                      # hardware, shots: samples per datapoint and readout time
    "seed" => 1234,                         # hardware, shots: per-datapoint seeds derive from this
    "output_dir" => "datasets",             # statevector: where the feature CSV is written
    "experiment_root" => "data/experiments",# shots: where the experiment directory is created
    "resume" => "",                         # shots: existing experiment directory to append to
    "checkpoint_every" => 100,              # shots: save the experiment every N datapoints
    "tag" => "",                            # appended to output and experiment names
    "dry_run" => false,                     # print the resolved settings and exit
    "layer" => Dict{String, Any}(
        "t_start" => 0.0,                   # µs
        "t_end" => 2.5,                     # µs
        "step" => 0.5,                      # µs between readouts
        "omega" => 2π,                      # Rabi frequency, rad/µs
        "scale" => 10.0,                    # µm, base spacing of the atom chain
        "delta" => 2.0,                     # position: uniform detuning; detuning: offset Δ
        "lambda" => 1.0,                    # position only: displacement scale λ
        "delta_max" => -9.0,                # detuning only: detuning window Δ_max
    ),
)

# ---------------------------------------------------------------------------
# Inputs and scaling
# ---------------------------------------------------------------------------

"Read a preprocessed set: returns (ids, X) with X of size n_features × n_datapoints."
function load_set(input_dir, name)
    path = joinpath(input_dir, name * ".csv")
    df = DataFrame(CSV.File(path; header = true))
    "ID" in names(df) || error("$path has no ID column")
    ids = Vector{Int}(df.ID)
    X = permutedims(Matrix{Float64}(select(df, Not(:ID))))
    return ids, X, path
end

# Scaling as in the runs reported in the paper: each set is shifted by its own
# global minimum, then divided by the largest magnitude of the shifted
# reference set, so that the reference set spans [0, 1].
shift_to_zero(X) = X .- minimum(X)
scale_factor(X_reference) = maximum(abs, shift_to_zero(X_reference))

function datapoint_range(n, first, last)
    stop = last == -1 ? n : last
    (1 <= first <= stop <= n) || error("Invalid datapoint range first=$first last=$last for a set of $n datapoints")
    return first:stop
end

# ---------------------------------------------------------------------------
# Reservoir
# ---------------------------------------------------------------------------

function build_layer(s, n_features)
    L = s["layer"]
    common = (; t_start = L["t_start"], t_end = L["t_end"], step = L["step"], Ω = L["omega"], scale = L["scale"])
    if s["encoding"] == "position"
        return Qrcutils.PositionLayer(; dim = n_features, Δ = L["delta"], λ = L["lambda"], common...)
    elseif s["encoding"] == "detuning"
        return Qrcutils.DetuningLayer(; dim = n_features, Δ_max = L["delta_max"], Δ = L["delta"], common...)
    else
        error("Unknown encoding `$(s["encoding"])`: expected \"position\" or \"detuning\"")
    end
end

# The seed of one datapoint's shots depends only on the run seed and the
# datapoint ID, so a run can be split into ranges, interrupted and resumed
# without changing any datapoint's measurement records.
seed_datapoint!(seed, id) = Random.seed!(UInt32[seed, id])

function output_stem(s, set, n_features, range, n_total)
    L = s["layer"]
    stem = @sprintf("%s_encoding_%s_%d_delta_%0.1f", s["encoding"], set, n_features, L["delta"])
    s["encoding"] == "position" && (stem *= @sprintf("_lambda_%0.1f", L["lambda"]))
    s["encoding"] == "detuning" && (stem *= @sprintf("_deltamax_%0.1f", L["delta_max"]))
    length(range) == n_total || (stem *= "_$(first(range))-$(last(range))")
    isempty(s["tag"]) || (stem *= "_" * s["tag"])
    return stem
end

function run_statevector(s, set, ids, X, n_features, range, inputs)
    s["readout_timing"] in ("exact", "integrator") || error("readout_timing must be \"exact\" or \"integrator\"")
    exact = s["readout_timing"] == "exact"
    n = length(range)
    columns = Vector{Vector{Float64}}(undef, n)
    nt = Threads.nthreads()
    progress("Statevector run on $n datapoints of $set with $nt thread(s)")
    done = Threads.Atomic{Int}(0)
    t0 = time()
    # One layer per task: a layer owns the register the evolution writes into.
    @sync for chunk in Iterators.partition(1:n, cld(n, nt))
        Threads.@spawn begin
            layer = build_layer(s, n_features)
            for k in chunk
                i = range[k]
                columns[k] = Qrcutils.ideal_features(layer, X[:, i]; exact_times = exact)
                c = Threads.atomic_add!(done, 1) + 1
                if c % 100 == 0 || c == n
                    progress(@sprintf("  %d/%d datapoints, %.0f s elapsed", c, n, time() - t0))
                end
            end
        end
    end
    F = reduce(hcat, columns)
    mkpath(s["output_dir"])
    stem = output_stem(s, set, n_features, range, length(ids))
    stem *= exact ? "_sv_exact" : "_sv"
    out = joinpath(s["output_dir"], stem * ".csv")
    CSV.write(out, DataFrame(F, string.(ids[range])))
    Qrcutils.write_manifest(joinpath(s["output_dir"], stem * ".manifest.json"), s;
        set = set, output = out, n_datapoints = n, n_features_out = size(F, 1), inputs = inputs,
        wall_time_s = round(time() - t0, digits = 1))
    progress("Wrote $(size(F, 1)) features × $n datapoints to $out")
    return out
end

function run_hardware(s, set, ids, X, n_features, range, inputs)
    counts = Int.(as_list(s["shot_counts"]))
    s["shot_counts"] = counts
    n_shots = s["n_shots"]
    all(1 .<= counts .<= n_shots) || error("shot_counts must lie in 1:n_shots (n_shots = $n_shots)")
    n = length(range)
    exact_cols = Vector{Vector{Float64}}(undef, n)
    shot_cols = [Vector{Vector{Float64}}(undef, n) for _ in counts]
    nsites = n_features + (s["encoding"] == "position" ? 1 : 0)
    n_times = length(Qrcutils.readout_times(build_layer(s, n_features)))
    bits_all = s["save_bitstrings"] ? zeros(UInt32, n_shots, n_times, n) : nothing
    nt = Threads.nthreads()
    progress("Hardware-protocol run on $n datapoints of $set with $nt thread(s), $n_shots samples per readout")
    done = Threads.Atomic{Int}(0)
    t0 = time()
    @sync for chunk in Iterators.partition(1:n, cld(n, nt))
        Threads.@spawn begin
            layer = build_layer(s, n_features)
            for k in chunk
                i = range[k]
                states = Qrcutils.hardware_protocol_states(layer, X[:, i])
                exact_cols[k] = reduce(vcat, [Qrcutils.readout_features(layer, r) for r in states])
                # seeded per datapoint: independent of thread count, ranges and order
                bits = Qrcutils.sample_bitstrings(states, n_shots, MersenneTwister(UInt32[s["seed"], ids[i]]))
                for (c, m) in enumerate(counts)
                    shot_cols[c][k] = Qrcutils.shot_features(bits, nsites, m)
                end
                isnothing(bits_all) || (bits_all[:, :, k] .= bits)
                c = Threads.atomic_add!(done, 1) + 1
                if c % 100 == 0 || c == n
                    progress(@sprintf("  %d/%d datapoints, %.0f s elapsed", c, n, time() - t0))
                end
            end
        end
    end
    mkpath(s["output_dir"])
    stem = output_stem(s, set, n_features, range, length(ids)) * "_hw"
    names = string.(ids[range])
    outputs = Dict{String, String}()
    outputs["exact"] = joinpath(s["output_dir"], stem * ".csv")
    CSV.write(outputs["exact"], DataFrame(reduce(hcat, exact_cols), names))
    for (c, m) in enumerate(counts)
        outputs["shots_$m"] = joinpath(s["output_dir"], stem * "shots_$m.csv")
        CSV.write(outputs["shots_$m"], DataFrame(reduce(hcat, shot_cols[c]), names))
    end
    if !isnothing(bits_all)
        outputs["bitstrings"] = joinpath(s["output_dir"], stem * "bits.jld2")
        jldsave(outputs["bitstrings"]; ids = ids[range], bits = bits_all, nsites = nsites,
                readout_times = Qrcutils.readout_times(build_layer(s, n_features)),
                layout = "bits[shot, readout, datapoint]; bit k (least significant = 1) is atom k")
    end
    Qrcutils.write_manifest(joinpath(s["output_dir"], stem * ".manifest.json"), s;
        set = set, outputs = outputs, n_datapoints = n, n_features_out = length(exact_cols[1]),
        inputs = inputs, wall_time_s = round(time() - t0, digits = 1))
    progress("Wrote hardware-protocol features ($(length(exact_cols[1])) × $n) and shot counts $(counts) under $(s["output_dir"])/$(stem)*")
    return outputs
end

function run_shots(s, set, ids, X, n_features, range, inputs)
    layer = build_layer(s, n_features)
    if isempty(s["resume"])
        name = "QRC_$(n_features)_emu_true_$(s["encoding"])" * (isempty(s["tag"]) ? "" : "_" * s["tag"])
        experiment = Qrcutils.create_experiment(name; n_shots = s["n_shots"],
            timesteps = Qrcutils.layer_timesteps(layer), root_dir = s["experiment_root"])
    else
        experiment = Qrcutils.resume_experiment(s["resume"])
        experiment.nshots == s["n_shots"] || error("n_shots=$(s["n_shots"]) differs from the resumed experiment's $(experiment.nshots)")
        progress("Resuming $(experiment.experiment_dir): $(length(experiment.tasks_id)) datapoints already done")
    end
    Qrcutils.write_manifest(joinpath(experiment.experiment_dir, "run_manifest_$(Dates.format(Dates.now(), "yyyy-mm-dd-HH.MM.SS")).json"), s;
        set = set, experiment_dir = experiment.experiment_dir, datapoint_range = [first(range), last(range)], inputs = inputs)

    todo = [i for i in range if !haskey(experiment.tasks_id, ids[i])]
    progress("Shot-based run on $(length(todo)) datapoints of $set ($(length(range) - length(todo)) already in the experiment)")
    t0 = time()
    for (k, i) in enumerate(todo)
        seed_datapoint!(s["seed"], ids[i])
        Qrcutils.apply_layer(layer, X[:, i], true, experiment, ids[i])
        if k % s["checkpoint_every"] == 0 || k == length(todo)
            Qrcutils.save_experiment(experiment)
            progress(@sprintf("  %d/%d datapoints, %.0f s elapsed, checkpoint saved", k, length(todo), time() - t0))
        end
    end
    isempty(todo) && Qrcutils.save_experiment(experiment)
    progress("Experiment stored in $(experiment.experiment_dir)")
    return experiment.experiment_dir
end

# ---------------------------------------------------------------------------

function main()
    s, config_file = load_settings(DEFAULTS)
    s["backend"] in ("statevector", "hardware", "shots") || error("Unknown backend `$(s["backend"])`: expected \"statevector\", \"hardware\" or \"shots\"")
    sets = as_list(s["sets"])
    s["sets"] = sets
    s["backend"] == "shots" && !isempty(s["resume"]) && length(sets) > 1 &&
        error("resume= appends to one experiment, but $(length(sets)) sets were given; resume one set at a time")
    if s["dry_run"]
        println("Resolved settings$(isnothing(config_file) ? "" : " from $config_file"):")
        TOML.print(s; sorted = true)
        return
    end
    reference = isempty(s["scale_reference"]) ? first(sets) : s["scale_reference"]
    _, X_reference, reference_path = load_set(s["input_dir"], reference)
    spectral = scale_factor(X_reference)
    progress("Config $(something(config_file, "(defaults)")): $(s["encoding"]) encoding, $(s["backend"]) backend, scaling by $reference (factor $(round(spectral, digits = 6)))")

    for set in sets
        ids, X, path = load_set(s["input_dir"], set)
        X = shift_to_zero(X) ./ spectral
        range = datapoint_range(length(ids), s["first"], s["last"])
        if length(range) > 5000 && Base.JLOptions().heap_size_hint == 0 && s["backend"] != "shots"
            @warn """Large run ($(length(range)) datapoints) without a heap-size cap: the garbage collector
                     falls behind on long multithreaded runs (observed: RSS ~85 GB, under half the threads busy).
                     Start Julia with e.g. --heap-size-hint=40G."""
        end
        inputs = Dict(path => file_sha256(path), reference_path => file_sha256(reference_path))
        if s["backend"] == "statevector"
            run_statevector(s, set, ids, X, size(X, 1), range, inputs)
        elseif s["backend"] == "hardware"
            run_hardware(s, set, ids, X, size(X, 1), range, inputs)
        else
            run_shots(s, set, ids, X, size(X, 1), range, inputs)
        end
    end
end

main()
