module Qrcutils
export register_embedding!, DetuningLayer, PositionLayer, apply_layer, resubmit_single_timestep

include("experiments.jl")

using .ExperimentUtils
using PythonCall
using Bloqade, BloqadeSchema
using Dates
using CUDA, Adapt
using Printf
using Serialization
using JSON
using Random

plt = pyimport("matplotlib.pyplot")
blo = pyimport("bloqade")
SHOW_PROGRESS_BAR = false

abstract type AbstractLayer end

#Define the struct
Base.@kwdef struct DetuningLayer <: AbstractLayer
    dim::Int64 # problem dimension
    t_start::Float64 # evolution starting time
    t_end::Float64 # evolution ending time
    step::Float64  # readout time step
    readouts = readouts(dim) # readout observables
    Ω::Float64 = 2π  # Rabi frequency
    scale::Float64 = 10.0 # distance between atoms
    atoms = generate_sites(ChainLattice(), dim; scale = scale) # atom positions
    reg::AbstractArrayReg = zero_state(dim) # quantum state storage
    Δ_max::Float64 = 6.0
    Δ::Float64 = 0.0
end;

#Define the struct
Base.@kwdef struct PositionLayer <: AbstractLayer
    dim::Int64 # problem dimension
    Δ::Float64 # Detuning frequency
    λ::Float64 # Scale factor
    t_start::Float64 # evolution starting time
    t_end::Float64 # evolution ending time
    step::Float64  # readout time step
    readouts = readouts(dim+1) # readout observables
    Ω::Float64 = 2π # Rabi frequency
    scale::Float64 = 10.0 # distance between atoms
    atoms = generate_sites(ChainLattice(), dim+1; scale = scale) # atom positions
    reg::AbstractArrayReg = zero_state(dim+1) # quantum state storage
end;

"""
    directaccess_config()

Endpoint and request key for QuEra direct-access submission, returned as a
`(url, key)` named tuple.

Read from the environment (`QRC_DIRECT_ACCESS_URL`, `QRC_DIRECT_ACCESS_KEY`)
or, failing that, from an untracked config file — `direct_access.conf` in the
working directory by default, overridable with `QRC_DIRECT_ACCESS_CONFIG`.
See `direct_access.conf.example` for the format.

These are credentials: keep them out of the repository. Only hardware
submission calls this; emulation runs never do.
"""
function directaccess_config()
    url = get(ENV, "QRC_DIRECT_ACCESS_URL", "")
    key = get(ENV, "QRC_DIRECT_ACCESS_KEY", "")
    path = get(ENV, "QRC_DIRECT_ACCESS_CONFIG", joinpath(pwd(), "direct_access.conf"))
    if (isempty(url) || isempty(key)) && isfile(path)
        for line in eachline(path)
            line = strip(line)
            (isempty(line) || startswith(line, "#")) && continue
            parts = split(line, '=', limit = 2)
            length(parts) == 2 || continue
            k = strip(parts[1])
            v = String(strip(parts[2], ['"', '\'', ' ']))
            k == "url" && isempty(url) && (url = v)
            k == "key" && isempty(key) && (key = v)
        end
    end
    if isempty(url) || isempty(key)
        error("""
              Direct-access submission needs an endpoint URL and a request key, and neither \
              the environment nor a config file provided them.
              Set QRC_DIRECT_ACCESS_URL and QRC_DIRECT_ACCESS_KEY, or copy \
              direct_access.conf.example to $(path) and fill it in.
              Emulation runs do not need this.
              """)
    end
    return (url = url, key = key)
end

"""
    git_state(dir = pwd())

Commit SHA of the checkout containing `dir`, and whether tracked files have
uncommitted changes. Both are `nothing` outside a git checkout.
"""
function git_state(dir::AbstractString = pwd())
    commit = try readchomp(Cmd(`git rev-parse HEAD`; dir = dir)) catch; nothing end
    dirty = try
        !isempty(readchomp(Cmd(`git status --porcelain --untracked-files=no`; dir = dir)))
    catch
        nothing
    end
    return (commit = commit, dirty = dirty)
end

"""
    write_manifest(path, config; extra...)

Write a JSON provenance record for a run next to its outputs: the resolved
configuration, the script and its arguments, the git commit and whether the
tree was dirty, and the Julia version. Keyword arguments are added as extra
top-level fields. A clean commit pins the code and, through the tracked
Manifest.toml, every Julia dependency.
"""
function write_manifest(path::AbstractString, config::AbstractDict; extra...)
    g = git_state()
    g.dirty === true && @warn "Working tree has uncommitted changes; the manifest's git_commit does not fully describe this run."
    record = Dict{String, Any}(
        "created" => Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS"),
        "script" => PROGRAM_FILE,
        "args" => ARGS,
        "git_commit" => g.commit,
        "git_dirty" => g.dirty,
        "julia_version" => string(VERSION),
        "threads" => Threads.nthreads(),
        "config" => config,
    )
    for (k, v) in extra
        record[string(k)] = v
    end
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        JSON.print(io, record, 2)
    end
    return path
end

function resubmit_single_timestep(experiment::Experiment, task_id::Int, time_step::Int64, n_shots::Int64)
    blo = pyimport("bloqade")
    cfg = directaccess_config()
    waveform = deserialize(joinpath(experiment.experiment_dir, "waveforms", "task_$(task_id).jls"))
    request_options = pydict(params=pydict(key=cfg.key, note= experiment.name*"_"*string(task_id)))
    task = waveform.assign(run_time=experiment.timesteps[time_step]-0.12).quera.custom().submit(n_shots, cfg.url, "{task_ir}", request_options = request_options, use_experimental=true)
end

function readouts(nsites::Int64)
    readouts = AbstractBlock[put(nsites, i => Z) for i in 1:nsites]
    for i in 1:nsites
        for j in i+1:nsites
            push!(readouts, chain(put(nsites, i => Z), put(nsites, j => Z)))
        end
    end
    return readouts
end

function replicate!(atoms_enc::Vector{Tuple{Float64, Float64}}, atoms_dist::Float64, height::Float64)
    temp = copy(atoms_enc)
    y_max = temp[end][2]
    scale_dist = 1.5
    copies = floor(Int, (height+scale_dist*atoms_dist)/(y_max+scale_dist*atoms_dist))
    for i in range(1, copies-1)
        append!(atoms_enc, [(atom[1], atom[2]+i*(y_max+scale_dist*atoms_dist)) for atom in temp])
    end
    return copies
end

function register_embedding!(atoms_enc::Vector{Tuple{Float64, Float64}}, distance::Float64, width::Float64, height::Float64; filling::Bool=true, print::String = "")
#function register_embedding!(atoms_enc::Vector{Tuple{Float64}}, distance::Float64; filling::Bool=true)
    atoms_dist = [atoms_enc[index][1] - atoms_enc[index-1][1] for index in 2:length(atoms_enc)]
    #atoms_enc = [(atom[1], 0.0) for atom in atoms_enc]
    area = [width, height]
    reversed = 0
    dim = 1
    for index in 1:length(atoms_enc)-1
        dis = 0.0
        temp = [atoms_enc[index][1],atoms_enc[index][2]]
        if dim == 1
            dis = (-1)^(reversed) * atoms_dist[index]
            if temp[dim] + dis > area[dim] || temp[dim] + dis < 0.0
                dim = 2
                reversed = abs(reversed-1)
            else
                temp[dim] += dis
            end
        end
        if dim == 2
            temp[dim] += atoms_dist[index]
            if temp[dim] > area[dim]
                error("ABORTING: features not embeddable! Try to reduce atoms distance...")
            elseif dis == 0.0
                dim = 1
            end
        end
        atoms_enc[index+1] = (temp[1], temp[2])
    end
    if print != ""
        Bloqade.plot(AtomList(atoms_enc); filename=print)
    end
    copies = 1
    if filling
        copies = replicate!(atoms_enc, distance, area[2])
    end
    if print != ""
        Bloqade.plot(AtomList(atoms_enc); filename="replicated_$print")
    end
    offset = (0.0, 0.0)
    max = (area[1] - maximum([x[1] for x in atoms_enc]), area[2] - maximum([x[2] for x in atoms_enc]))
    min = (minimum([x[1] for x in atoms_enc]), minimum([x[2] for x in atoms_enc]))
    if area[1] - max[1] > 0.2
        offset = ((max[1] - min[1])/2, offset[2])
    end
    if area[2] - max[2] > 0.2
        offset = (offset[1], (max[2] - min[2])/2)
    end
    atoms_enc = [(atom[1]+offset[1], atom[2]+offset[2]) for atom in atoms_enc]
    if print != ""
        Bloqade.plot(AtomList(atoms_enc); filename="centered_$print")
    end
    return copies
end

function detuning_waveform(Δ::Float64, t_start::Float64, t_end::Float64)
    return piecewise_linear(clocks=[t_start, 0.06, t_end], values= [0.0, Δ, Δ]);
end

function detuning_waveform(Δ::Vector{<:Real}, t_start::Float64, t_end::Float64)
    Δ_t = map(1:length(Δ)) do idx
        return detuning_waveform(Δ[idx], t_start, t_end)
    end
    return Δ_t
end

function build_waveform(atoms_enc::Vector{Tuple{Float64, Float64}}, layer::AbstractLayer)
    blo = pyimport("bloqade")
    return blo.start.add_position_list_tuples(atoms_enc).rydberg.rabi.amplitude.uniform.piecewise_linear(
        durations = pylist([0.06, "run_time", 0.06]), values = pylist([0.0, layer.Ω, layer.Ω, 0.0])).rydberg.detuning.uniform.piecewise_linear(
        durations = pylist([0.06, "run_time", 0.06]), values = pylist([layer.Δ, layer.Δ, layer.Δ, layer.Δ]))
end

function build_waveform(atoms_enc::Vector{Tuple{Float64, Float64}}, layer::DetuningLayer, Δ_loc::Vector{<:Real})
    return build_waveform(atoms_enc, layer).rydberg.detuning.
        location(pylist([i-1 for i in range(1,length(Δ_loc))]), pylist(1 .- Δ_loc)).piecewise_linear(
            durations = pylist([0.06, "run_time", 0.06]), values = pylist([0.0, layer.Δ_max, layer.Δ_max, 0.0]))
end

function build_waveform(atoms_enc::Vector{Tuple{Float64, Float64}}, layer::PositionLayer, Δ_loc::Vector{<:Real})
    return build_waveform(atoms_enc, layer)
end

function shot_exec!(experiment::Experiment, atoms_enc::Vector{Tuple{Float64, Float64}}, layer::AbstractLayer, data_id::Int; copies::Int = 1, Δ_loc::Vector{<:Real} = Vector{Real}(), direct_access::Bool = true)
    println("time: ", Dates.now())
    waveform = build_waveform(atoms_enc, layer, Δ_loc)
    n_shots::Int = floor(Int, experiment.nshots/copies)
    if !direct_access
        task = waveform.batch_assign(run_time=pylist([time_step-0.12 for time_step in layer.t_start+layer.step:layer.step:layer.t_end])).braket.aquila().run_async(shots=n_shots, name = experiment.name * "_task_$data_id", use_experimental=true)
        arn = joinpath(experiment.experiment_dir, "tasks", "task_$(data_id).json")
        blo = pyimport("bloqade")
        blo.save(task, arn,)
    else
        cfg = directaccess_config()
        request_options = pydict(params=pydict(key=cfg.key, note= experiment.name*"_"*string(data_id)))
        task = waveform.batch_assign(run_time=pylist([time_step-0.12 for time_step in layer.t_start+layer.step:layer.step:layer.t_end])).quera.custom().submit(n_shots, cfg.url, "{task_ir}", request_options = request_options, use_experimental=true)
        task = Dict(Dates.now() => pylist([string(element) for element in task]))
        arn = joinpath(experiment.experiment_dir, "tasks", "task_$(data_id).jls")
        serialize(arn, task)
    end
    path = joinpath(experiment.experiment_dir, "waveforms", "task_$(data_id).jls")
    serialize(path, waveform)
    gt = ExpDataTask(arn, data_id, n_shots, path, copies);
    create_task!(experiment, gt)
end

function shot_exec!(experiment::Experiment, atoms_enc, layer, Δ, data_id::Int; copies::Int = 1, GPU::Bool = true)
    println("time: ", Dates.now())
    h_t::Array{RydbergHamiltonian, 1} = [];
    for time_step in layer.t_start:layer.step:layer.t_end
        time_step == layer.t_start && continue
        Ω_t = piecewise_linear(clocks=[layer.t_start, 0.06, time_step-0.06, time_step], values= [0.0, layer.Ω, layer.Ω, 0.0]);
        Δ_t = detuning_waveform(Δ, layer.t_start, time_step)
        ϕ = constant(duration=time_step, value=0.0);
        push!(h_t, rydberg_h(AtomList(atoms_enc); Δ = Δ_t, Ω = Ω_t, ϕ = ϕ))
    end
    issubspace = false
    if issubspace
        Rsubspace = 0.5 * distance
        space = blockade_subspace(AtomList(atoms_enc), Rsubspace)
    else
        space = length(atoms_enc)
    end
    #reg = zero_state(length(atoms_enc))
    reg = zero_state(space)
    countedmeasures_ = Vector{Vector{DitStr{2, length(atoms_enc), Int64}}}()
    clocks = [layer.t_start, layer.t_start+layer.step]
    temp = [clocks[1], 0.05, clocks[2]]
    if GPU && !CUDA.functional()
        @info "No functional GPU, running shot-based emulation on the CPU" maxlog=1
        GPU = false
    end
    for h in h_t
        #println("starting GPU emu")
        prob = KrylovEvolution(reg, temp, h)
        #prob =  SchrodingerProblem(reg, layer.t_end, h, adaptive = true)
        #println("prob defined")
        if GPU
            gpu_prob = adapt(CuArray, prob)
            #println("CPU mem copy completed")
            emulate!(gpu_prob)
            #println("gpu emu completed")
            prob = adapt(Array, gpu_prob)
        else
            emulate!(prob)
        end
        #println("GPU mem copy completed")
        reg = prob.reg
        measurements = Bloqade.measure(prob.reg; nshots=floor(Int, experiment.nshots/copies))
        push!(countedmeasures_, measurements)
        clocks .+= layer.step
        temp = clocks
    end
    create_task!(experiment, h_t, countedmeasures_, data_id; copies = copies)
    return 0
end

# `exact_times = false` (the default): each readout is taken at the
# adaptive integrator's step endpoint at or just after the requested time (docs/architecture.md).
# `exact_times = true` steps the integrator exactly onto every readout time.
function statevector_sim(atoms_enc, layer::AbstractLayer, Δ; exact_times::Bool = false)
    Ω_waveform = piecewise_linear(clocks=[layer.t_start, 0.06, layer.t_end - 0.06, layer.t_end], values= [0.0, layer.Ω, layer.Ω, 0.0]);
    Δ_waveform = detuning_waveform(Δ, layer.t_start, layer.t_end);
    # Bloqade.plot(AtomList(atoms_enc); filename="atoms_emb_Aquila_after.svg")
    
    # fig, (ax1, ax2) = plt.subplots(figsize=(12, 4), ncols=2)
    # Bloqade.plot!(ax1, Ω_waveform)
    # fig.savefig("waveforms.svg")
    h = rydberg_h(atoms_enc; Δ = Δ_waveform, Ω = Ω_waveform)
    
    # At the start of the simulation, all atoms of the system are in ground states
    reg = layer.reg
    set_zero_state!(reg) 
    
    t_start = layer.t_start
    t_end = layer.t_end
    t_step = layer.step
    
    # initialize output vector
    steps = floor(Int, (t_end - t_start) / t_step)
    out = zeros(steps * length(layer.readouts))
    
    # Numerically simulate the quantum evolution with Krylov methods and store the readouts
    i = 1
    #prob = KrylovEvolution(reg, layer.t_start:layer.step:layer.t_end, h)
    prob =  SchrodingerProblem(reg, layer.t_end, h, adaptive = true)
    integrator = init(prob, Vern8())

    if exact_times
        for T in readout_times(layer)
            T > integrator.t && step!(integrator, T - integrator.t, true)   # land exactly on T
            for op in layer.readouts
                out[i] = real(expect(op, prob.reg))   # `expect` does not modify the register
                i += 1
            end
        end
        return out
    end
    for _ in TimeChoiceIterator(integrator, collect((t_start+t_step):t_step:t_end))
        prob.reg # state at selected time
        for op in layer.readouts
            out[i] = real(expect(op, prob.reg)) # store the expectation of each operator; `expect` does not modify the register
            i+=1
        end
    end

    #for (step, reg, _) in prob # step through the state at each time step 
    #    step == 1 && continue # ignore first time step, this is just the initial state
    #    for op in layer.readouts
    #        out[i] = real(expect(op, prob.reg)) # store the expectation of each operator for the given state in the output vector 
    #        i+=1
    #    end
    #end
    return out
end

# ---------------------------------------------------------------------------
# Backends added for the article (docs/architecture.md, "What each backend evolves"):
#   ideal_features             continuous pulse, readouts at exact times
#   hardware_protocol_states   the Aquila programs, one fresh program per readout time
#   shot_features              ⟨Zᵢ⟩, ⟨ZᵢZⱼ⟩ estimated from sampled bitstrings
# ---------------------------------------------------------------------------

"Readout times of a layer: `t_start + step, …, t_end` (`t_start` itself is not read out)."
readout_times(layer::AbstractLayer) = collect((layer.t_start + layer.step):layer.step:layer.t_end)

"Ramp duration of Ω and of the local detuning in every hardware program, µs."
const HW_RAMP = 0.06

"The single, unreplicated chain encoding one datapoint, as 1-D positions."
encoded_atoms(layer::DetuningLayer, x::AbstractVector{<:Real}) = convert(Vector{Tuple{Float64}}, layer.atoms)
function encoded_atoms(layer::PositionLayer, x::AbstractVector{<:Real})
    atoms = convert(Vector{Tuple{Float64}}, layer.atoms)
    for index in 2:length(atoms)
        atoms[index:end] = [(atom[1] - layer.scale/6*layer.λ*x[index-1],) for atom in atoms[index:end]]
    end
    return atoms
end

"Expectation values of the layer's readout operators, in readout order."
readout_features(layer::AbstractLayer, reg) = [real(expect(op, reg)) for op in layer.readouts]

"""
    ideal_features(layer, x; exact_times = true)

Statevector features of one datapoint under the statevector backend's continuous pulse (Ω held
through every readout). With `exact_times = false` the readouts are at the integrator's step endpoints.
"""
ideal_features(layer::DetuningLayer, x::AbstractVector{<:Real}; exact_times::Bool = true) =
    statevector_sim(layer.atoms, layer, x .* -layer.Δ_max .+ (layer.Δ + layer.Δ_max); exact_times = exact_times)
ideal_features(layer::PositionLayer, x::AbstractVector{<:Real}; exact_times::Bool = true) =
    statevector_sim(encoded_atoms(layer, x), layer, layer.Δ; exact_times = exact_times)

"""
    hardware_hamiltonian(layer, x, T; ramp_down = true)

Single-chain Hamiltonian of the Aquila program with readout time `T`, with the waveforms of the
submitted programs (`build_waveform`, verified against stored tasks): Ω ramps from 0 over `HW_RAMP`,
holds, and returns to 0 over the last `HW_RAMP`; the global detuning is constant from t = 0; the
detuning encoding adds a local term (1 − xᵢ)·Δ_max ramped like Ω. `ramp_down = false` gives the hold
Hamiltonian that every program follows up to `T − HW_RAMP`.
"""
function hardware_hamiltonian(layer::AbstractLayer, x::AbstractVector{<:Real}, T::Real; ramp_down::Bool = true)
    clocks, profile = ramp_down ? ([0.0, HW_RAMP, T - HW_RAMP, T], [0.0, 1.0, 1.0, 0.0]) :
                                  ([0.0, HW_RAMP, T], [0.0, 1.0, 1.0])
    Ω = piecewise_linear(clocks = clocks, values = layer.Ω .* profile)
    return rydberg_h(encoded_atoms(layer, x); Ω = Ω, Δ = hardware_detuning(layer, x, clocks, profile))
end
hardware_detuning(layer::PositionLayer, x, clocks, profile) =
    piecewise_linear(clocks = clocks, values = fill(layer.Δ, length(clocks)))
hardware_detuning(layer::DetuningLayer, x, clocks, profile) =
    [piecewise_linear(clocks = clocks, values = layer.Δ .+ (1 - xi) * layer.Δ_max .* profile) for xi in x]

"Evolve |0…0⟩ under `h`, returning copies of the register at exactly each of `times` (ascending)."
function evolve_to_times(h, nsites::Integer, times::AbstractVector{<:Real})
    reg = zero_state(nsites)
    prob = SchrodingerProblem(reg, times[end], h)
    integrator = init(prob, Vern8())
    states = typeof(reg)[]
    for T in times
        T > integrator.t && step!(integrator, T - integrator.t, true)
        push!(states, copy(prob.reg))
    end
    return states
end

"""
    hardware_protocol_states(layer, x)

Noiseless single-chain registers at each readout time T, each prepared as on the hardware: a fresh
program from |0…0⟩ that ends at T. All programs coincide up to T − `HW_RAMP`, so one evolution under
the hold Hamiltonian is branched into a `HW_RAMP` ramp-down per T — identical to running every program
separately, at little more than the cost of one.
"""
function hardware_protocol_states(layer::AbstractLayer, x::AbstractVector{<:Real})
    layer.t_start == 0 || error("hardware programs start at t = 0; got t_start = $(layer.t_start)")
    times = readout_times(layer)
    minimum(times) >= 2 * HW_RAMP || error("readout times must be at least $(2 * HW_RAMP) µs")
    n = length(encoded_atoms(layer, x))
    pre = evolve_to_times(hardware_hamiltonian(layer, x, layer.t_end; ramp_down = false), n, times .- HW_RAMP)
    return map(zip(times, pre)) do (T, reg)
        prob = SchrodingerProblem(reg, (T - HW_RAMP, T), hardware_hamiltonian(layer, x, T))
        emulate!(prob)
        prob.reg
    end
end

"""
    sample_bitstrings(states, n_shots, rng)

`n_shots × length(states)` matrix of computational-basis samples, one column per state, each sample
an integer whose bit k (least significant = 1) is atom k.
"""
function sample_bitstrings(states, n_shots::Integer, rng)
    bits = zeros(UInt32, n_shots, length(states))
    for (t, reg) in enumerate(states)
        bits[:, t] .= UInt32.(getfield.(Bloqade.measure(reg; nshots = n_shots, rng = rng), :buf))
    end
    return bits
end

"""
    shot_features(bits, nsites, n)

⟨Zᵢ⟩ then ⟨ZᵢZⱼ⟩ (i < j, row-major) per readout time, estimated from the first `n` samples of each
column of `bits`, concatenated — the layout of the readout features. Prefixes of one sample set give
nested subsamples, so features at increasing `n` form a consistent shot-count sweep.
"""
function shot_features(bits::AbstractMatrix{<:Integer}, nsites::Integer, n::Integer)
    1 <= n <= size(bits, 1) || error("requested $n shots, only $(size(bits, 1)) sampled")
    out = Float64[]
    for t in axes(bits, 2)
        z = [1.0 - 2.0 * ((bits[s, t] >> (i - 1)) & 1) for s in 1:n, i in 1:nsites]
        append!(out, vec(sum(z; dims = 1)) ./ n)
        for i in 1:nsites, j in i+1:nsites
            push!(out, sum(view(z, :, i) .* view(z, :, j)) / n)
        end
    end
    return out
end

function apply_layer(layer::DetuningLayer, x::Vector{<:Real}, emu::Bool, experiment, data_id::Int; exp_feat::Bool = true)
    if !isnothing(experiment)
        atoms_enc = convert(Vector{Tuple{Float64}}, layer.atoms)
        atoms_enc = [(atom[1], 0.0) for atom in atoms_enc]
        device_capabilities = BloqadeSchema.get_device_capabilities();
        height = device_capabilities.lattice.area.height
        if exp_feat
            height = 128.0
        end
        if atoms_enc[end][1] < height
            copies = register_embedding!(atoms_enc, layer.scale, height, device_capabilities.lattice.area.width, filling = !emu)
            atoms_enc = [(atom[2], atom[1]) for atom in atoms_enc]
        else
            copies = register_embedding!(atoms_enc, layer.scale, device_capabilities.lattice.area.width, height, filling = !emu)
        end
        #Bloqade.plot(AtomList(atoms_enc); filename="atoms_emb_Aquila_det_final.svg")
        temp = copy(x)
        for i in 1:copies - 1
            append!(x, copy(temp))
        end
        if emu
            x = x .* -layer.Δ_max .+ (layer.Δ + layer.Δ_max)
            shot_exec!(experiment, atoms_enc, layer, x, data_id; copies = copies)
        else
            shot_exec!(experiment, atoms_enc, layer, data_id; copies = copies, Δ_loc = x)
        end
        out = []
    else
        x = x .* -layer.Δ_max .+ (layer.Δ + layer.Δ_max)
        out = statevector_sim(layer.atoms, layer, x)
    end
    return out
end

function apply_layer(layer::PositionLayer, x::Vector{<:Real}, emu::Bool, experiment, data_id::Int; exp_feat::Bool = true)
    atoms_enc = convert(Vector{Tuple{Float64}}, layer.atoms)
    for index in 2:length(atoms_enc)
        atoms_enc[index:end] = [(atom[1] - layer.scale/6*layer.λ*x[index-1],) for atom in atoms_enc[index:end]]
    end
    if !isnothing(experiment)
        atoms_enc = [(atom[1], 0.0) for atom in atoms_enc]
        device_capabilities = BloqadeSchema.get_device_capabilities();
        height = device_capabilities.lattice.area.height
        if exp_feat
            height = 128.0
        end
        if atoms_enc[end][1] < height
            copies = register_embedding!(atoms_enc, layer.scale, height, device_capabilities.lattice.area.width, filling = !emu)
            atoms_enc = [(atom[2], atom[1]) for atom in atoms_enc]
        else
            copies = register_embedding!(atoms_enc, layer.scale, device_capabilities.lattice.area.width, height, filling = !emu)
        end
        if emu
            shot_exec!(experiment, atoms_enc, layer, layer.Δ, data_id; copies = copies)
        else
            shot_exec!(experiment, atoms_enc, layer, data_id; copies = copies)
        end
        out = []
    else
        out = statevector_sim(atoms_enc, layer, layer.Δ)
    end
    return out
end

"""
    experiment_name(layer, n_features, emu)

Name prefix of the experiment directory for a shot-based run, e.g. `QRC_12_emu_true`.
"""
experiment_name(layer::AbstractLayer, n_features::Integer, emu::Bool) = "QRC_$(n_features)_emu_$(emu)"

"""
    layer_timesteps(layer)

Timestep grid stored on an experiment: `t_start:step:t_end`, start included.
Readout `i` of a datapoint is labelled with entry `i` of this grid (see docs/architecture.md).
"""
layer_timesteps(layer::AbstractLayer) = [t for t in layer.t_start:layer.step:layer.t_end]

# implement functions that apply a `DetuningLayer` to a matrix containing scaled detunings for each image
#
# With `Aquila = true` the run is recorded as an experiment. Pass `experiment` to
# append to an existing one (e.g. to checkpoint a long run chunk by chunk, or to
# choose where it is stored); otherwise a new one is created under the default root.
function apply_layer(layer::AbstractLayer, x::Matrix{<:Real}; Aquila::Bool=false, emu::Bool=true, n_shots::Int = 1000, IDs::Vector{Int} = [i for i in 1:size(x, 2)], experiment = nothing)
    if size(x, 2) != length(IDs)
        @error "Number of IDs and datapoints are different."
    end
    if Aquila && isnothing(experiment)
        experiment = create_experiment(experiment_name(layer, size(x, 1), emu); n_shots = n_shots, timesteps = layer_timesteps(layer), create_folder = true)
    elseif !Aquila
        experiment = nothing
    end
    iter = SHOW_PROGRESS_BAR ? ProgressBar(1:size(x, 2)) : 1:size(x, 2)
    outs = [apply_layer(layer, x[:, i][:], emu, experiment, IDs[i]) for i in iter]
    if Aquila
        save_experiment(experiment)
    end
    return hcat(outs...)
end

end