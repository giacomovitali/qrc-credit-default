# Export measurement records to a flat layout that numpy reads directly.
#
#   julia --project=. analysis/hardware_noise/export_bits.jl OUT_PREFIX DIR [DIR ...]
#   julia --project=. analysis/hardware_noise/export_bits.jl OUT_PREFIX FILE_hwbits.jld2
#
# Inputs are either hardware experiment directories (the filtered per-readout
# bitstrings in results/task_<id>/task_<id>_step_<t>.jld2, exactly what the
# published Aquila features were reduced from) or a *_hwbits.jld2 written by
# run_reservoir.jl (backend = "hardware"). Writes
#
#   OUT_PREFIX_bits.u32    every record as a UInt32, bit k-1 = atom k (1 = Rydberg)
#   OUT_PREFIX_index.csv   id, readout (1…5 = 0.5…2.5 µs), offset, count, source
#
# A datapoint present in several directories is taken from the first one listed
# and reported. Datapoints missing any readout file are skipped and reported.

using qrc_module
using JLD2, CSV, DataFrames

const STEP_LABELS = ["0.0", "0.5", "1.0", "1.5", "2.0"]   # file labels of readouts 1…5

function export_dirs(prefix, dirs)
    bits = UInt32[]
    rows = NamedTuple[]
    seen = Dict{Int, String}()
    nsites = 0
    for dir in dirs
        rdir = joinpath(dir, "results")
        isdir(rdir) || (@warn "No results in $dir"; continue)
        for task in sort(readdir(rdir))
            id = parse(Int, replace(task, "task_" => ""))
            files = [joinpath(rdir, task, "$(task)_step_$(l).jld2") for l in STEP_LABELS]
            if !all(isfile, files)
                @warn "Datapoint $id in $dir: missing readout files, skipped"
                continue
            end
            if haskey(seen, id)
                @warn "Datapoint $id in $dir already taken from $(seen[id]); skipped"
                continue
            end
            seen[id] = dir
            for (r, f) in enumerate(files)
                m = load_object(f)
                nsites = length(m[1])
                push!(rows, (id = id, readout = r, offset = length(bits), count = length(m), source = basename(dir)))
                append!(bits, UInt32[UInt32(x.buf) for x in m])
            end
        end
    end
    return bits, DataFrame(rows), nsites
end

function export_hwbits(prefix, file)
    d = load(file)
    ids, B = d["ids"], d["bits"]          # samples × readouts × datapoints
    rows = NamedTuple[]
    for k in axes(B, 3), r in axes(B, 2)
        push!(rows, (id = ids[k], readout = r, offset = ((k - 1) * size(B, 2) + (r - 1)) * size(B, 1),
                     count = size(B, 1), source = basename(file)))
    end
    return vec(UInt32.(B)), DataFrame(rows), d["nsites"]
end

function main()
    length(ARGS) >= 2 || error("usage: export_bits.jl OUT_PREFIX DIR [DIR ...] | OUT_PREFIX FILE_hwbits.jld2")
    prefix, inputs = ARGS[1], ARGS[2:end]
    bits, index, nsites = endswith(inputs[1], ".jld2") ? export_hwbits(prefix, inputs[1]) : export_dirs(prefix, inputs)
    mkpath(dirname(abspath(prefix)))
    write(prefix * "_bits.u32", bits)
    index[!, :nsites] .= nsites
    CSV.write(prefix * "_index.csv", index)
    @info "Wrote $(length(bits)) records, $(length(unique(index.id))) datapoints, $nsites sites to $(prefix)_*"
end

main()
