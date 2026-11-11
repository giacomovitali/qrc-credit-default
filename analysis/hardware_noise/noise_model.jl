# Noise-model emulation of the Aquila programs.
#
#   julia --project=. -t 12 analysis/hardware_noise/noise_model.jl [n_samples=100] [realisations=8]
#
# Re-runs the hardware-protocol emulation (one fresh program per readout time, the
# submitted waveforms, single chain) on a fixed random subset of CARDS_1000 with one
# perturbation at a time, so each can be compared with the Aquila records:
#
#   coherent (deterministic): Rabi amplitude scale, global detuning offset, local-detuning
#     amplitude scale (detuning encoding only);
#   quasi-static noise (averaged over `realisations` draws per datapoint, i.e. the
#     shot-averaged expectation values): atom-position jitter, per-atom detuning error,
#     per-atom Rabi amplitude error, local-detuning pattern error, and the device
#     specifications combined.
#
# Magnitudes from the Aquila whitepaper (Wurtz et al., arXiv:2306.11727, Table 1.2):
# thermal position error 0.2 µm RMS per axis; detuning inhomogeneity 0.37 and shot-to-shot
# 0.18 rad/µs RMS; global detuning systematic error 0.63 rad/µs; Rabi inhomogeneity 2 % and
# shot-to-shot 0.8 % RMS. No accuracy specification is published for the local-detuning
# field, so its errors are scanned. Measurement errors (1 % g→r, 8 % r→g) act on the
# expectation values analytically and are applied in the Python analysis.
#
# Output: results/hardware_noise/noise_model/<encoding>_<setting>.csv, one column per datapoint
# (named by ID), in the layout of the reservoir output; the nominal setting must reproduce
# results/emulation/*_hw.csv.

using qrc_module
using Bloqade
using CSV, DataFrames, Random, Printf

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUT = joinpath(ROOT, "results", "hardware_noise", "noise_model")
const SPEC = (pos = 0.2, det = sqrt(0.37^2 + 0.18^2), omega = sqrt(0.02^2 + 0.008^2))

function arg(name, default)
    i = findfirst(a -> startswith(a, name * "="), ARGS)
    return isnothing(i) ? default : parse(Int, split(ARGS[i], "=")[2])
end

# --- inputs, scaled exactly as run_reservoir.jl does --------------------------------------
function load_scaled(set, reference)
    rd(name) = DataFrame(CSV.File(joinpath(ROOT, "dataset_aquila", name * ".csv")))
    ref = permutedims(Matrix{Float64}(DataFrames.select(rd(reference), Not(:ID))))
    df = rd(set)
    X = permutedims(Matrix{Float64}(DataFrames.select(df, Not(:ID))))
    return Vector{Int}(df.ID), (X .- minimum(X)) ./ maximum(abs, ref .- minimum(ref))
end

layer_for(enc) = enc == "detuning" ?
    Qrcutils.DetuningLayer(; dim = 12, Δ_max = -9.0, Δ = 4.5 + π, t_start = 0.0, t_end = 2.5, step = 0.5, Ω = 2π, scale = 10.0) :
    Qrcutils.PositionLayer(; dim = 12, Δ = 2.0, λ = 1.0, t_start = 0.0, t_end = 2.5, step = 0.5, Ω = 2π, scale = 10.0)

# --- perturbed hardware program ---------------------------------------------------------
struct Perturbation
    omega::Vector{Float64}     # relative Rabi amplitude per atom
    delta::Vector{Float64}     # detuning offset per atom, rad/µs
    loc::Vector{Float64}       # local-detuning amplitude factor per atom
    shift::Vector{NTuple{2, Float64}}   # position displacement per atom, µm
end
nominal(n) = Perturbation(ones(n), zeros(n), ones(n), fill((0.0, 0.0), n))

function hamiltonian(layer, x, T, p::Perturbation; ramp_down = true)
    clocks, profile = ramp_down ? ([0.0, Qrcutils.HW_RAMP, T - Qrcutils.HW_RAMP, T], [0.0, 1.0, 1.0, 0.0]) :
                                  ([0.0, Qrcutils.HW_RAMP, T], [0.0, 1.0, 1.0])
    base = Qrcutils.encoded_atoms(layer, x)
    atoms = [(a[1] + s[1], s[2]) for (a, s) in zip(base, p.shift)]
    Ω = all(==(p.omega[1]), p.omega) ? piecewise_linear(clocks = clocks, values = layer.Ω * p.omega[1] .* profile) :
        [piecewise_linear(clocks = clocks, values = layer.Ω * w .* profile) for w in p.omega]
    Δ = layer isa Qrcutils.DetuningLayer ?
        [piecewise_linear(clocks = clocks, values = (layer.Δ + d) .+ (1 - xi) * layer.Δ_max * l .* profile)
         for (xi, d, l) in zip(x, p.delta, p.loc)] :
        (all(==(p.delta[1]), p.delta) ? piecewise_linear(clocks = clocks, values = fill(layer.Δ + p.delta[1], length(clocks))) :
         [piecewise_linear(clocks = clocks, values = fill(layer.Δ + d, length(clocks))) for d in p.delta])
    return rydberg_h(atoms; Ω = Ω, Δ = Δ)
end

"Features at every readout time, prepared as in `Qrcutils.hardware_protocol_states`."
function features(layer, x, p)
    times = Qrcutils.readout_times(layer)
    n = length(p.omega)
    pre = Qrcutils.evolve_to_times(hamiltonian(layer, x, layer.t_end, p; ramp_down = false), n, times .- Qrcutils.HW_RAMP)
    out = Float64[]
    for (T, reg) in zip(times, pre)
        prob = SchrodingerProblem(reg, (T - Qrcutils.HW_RAMP, T), hamiltonian(layer, x, T, p))
        emulate!(prob)
        append!(out, Qrcutils.readout_features(layer, prob.reg))
    end
    return out
end

# --- settings ---------------------------------------------------------------------------
# (name, stochastic?, draw(rng, n) -> Perturbation)
function settings(enc)
    S = Any[("nominal", false, (r, n) -> nominal(n))]
    for f in (0.9, 0.95, 1.05, 1.1)
        push!(S, (@sprintf("omega_x%.2f", f), false, (r, n) -> (p = nominal(n); p.omega .= f; p)))
    end
    for d in (-1.0, -0.63, -0.3, 0.3, 0.63, 1.0)
        push!(S, (@sprintf("delta_%+.2f", d), false, (r, n) -> (p = nominal(n); p.delta .= d; p)))
    end
    if enc == "detuning"
        for l in (0.5, 0.7, 0.85, 1.15)
            push!(S, (@sprintf("local_x%.2f", l), false, (r, n) -> (p = nominal(n); p.loc .= l; p)))
        end
    end
    push!(S, ("pos_jitter_0.2um", true, (r, n) -> (p = nominal(n); p.shift .= [(SPEC.pos * randn(r), SPEC.pos * randn(r)) for _ in 1:n]; p)))
    push!(S, ("det_noise_spec", true, (r, n) -> (p = nominal(n); p.delta .= SPEC.det .* randn(r, n); p)))
    push!(S, ("omega_noise_spec", true, (r, n) -> (p = nominal(n); p.omega .= 1 .+ SPEC.omega .* randn(r, n); p)))
    if enc == "detuning"
        for s in (0.1, 0.2)
            push!(S, (@sprintf("local_pattern_%.1f", s), true, (r, n) -> (p = nominal(n); p.loc .= 1 .+ s .* randn(r, n); p)))
        end
    end
    push!(S, ("spec_combined", true, (r, n) -> Perturbation(1 .+ SPEC.omega .* randn(r, n), SPEC.det .* randn(r, n), ones(n),
                                                            [(SPEC.pos * randn(r), SPEC.pos * randn(r)) for _ in 1:n])))
    return S
end

function main()
    nsamp, K = arg("n_samples", 100), arg("realisations", 8)
    mkpath(OUT)
    pos_ids, _ = load_scaled("qpu_samples_pos", "qpu_samples")
    sel = sort(shuffle(MersenneTwister(20261021), pos_ids)[1:nsamp])
    for enc in ("detuning", "position")
        ids, X = enc == "detuning" ? load_scaled("qpu_samples", "qpu_samples") : load_scaled("qpu_samples_pos", "qpu_samples")
        cols = [findfirst(==(i), ids) for i in sel]
        layer = layer_for(enc)
        n = length(Qrcutils.encoded_atoms(layer, X[:, 1]))
        for (si, (name, stochastic, draw)) in enumerate(settings(enc))
            path = joinpath(OUT, "$(enc)_$(name).csv")
            isfile(path) && (println("exists: $path"); continue)
            t0 = time()
            reps = stochastic ? K : 1
            R = zeros(5 * (n + n * (n - 1) ÷ 2), nsamp, reps)
            Threads.@threads :dynamic for job in 1:(nsamp * reps)
                c, k = divrem(job - 1, reps) .+ 1
                rng = MersenneTwister(UInt32[20261021, si, sel[c], k])   # seeded per setting, datapoint, draw
                R[:, c, k] .= features(layer_for(enc), X[:, cols[c]], draw(rng, n))
            end
            F = dropdims(sum(R; dims = 3); dims = 3) ./ reps
            CSV.write(path, DataFrame(F, string.(sel)))
            @printf("%s %-20s %d × %d evaluations in %.0f s\n", enc, name, nsamp, reps, time() - t0)
            flush(stdout)
        end
    end
end

main()
