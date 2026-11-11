"""
This module contains all the function used to run and manage the experiments.
"""
module ExperimentUtils
include("utils.jl")

using .HardwareUtils: submittobraket, gettaskresults, gettaskstatus, hardwarevalidate, VerbosityLevel, LOW, getschemaparams, getschema, bitstrings_filtered
using .qcutils: compute_zi, compute_zij
using AWS, Braket
using JLD2
using Bloqade, BloqadeSchema
using Dates
using CSV
using DataFrames
using StatsBase: sample
using Random: Random, MersenneTwister
using PythonCall

export DataTask, ExpDataTask, Experiment, create_experiment, create_task!, save_experiment, save_task, resume_experiment, compute_exp_values, compute_exp_value, experiment_features

EXPERIMENT_ROOT_DIR = "data/experiments/"
DATASET_ROOT_DIR = "data/"

"""
    struct DataTask
    - arn: task id handle to submitted tasks (1 per time step)
    - data_id: id of the data point
    - n_shots: number of shots
    - timesteps: timesteps of the routine
"""

abstract type AbstractTask end
abstract type AbstractExperiment end

struct DataTask <: AbstractTask
    arn::Array{String, 1}
    data_id::Int
    n_shots::Int
    h_t::Array{RydbergHamiltonian, 1}
    copies::Int
end

struct ExpDataTask <: AbstractTask
    arn::String
    data_id::Int
    n_shots::Int
    h_t::String
    #h_t
    copies::Int
end

mutable struct Experiment <: AbstractExperiment
    name::String
    tasks::Array{AbstractTask, 1}
    tasks_id::Dict{Int, AbstractTask}
    nshots::Int
    timesteps::Array{Float64, 1}
    experiment_dir::String
end

function create_experiment(name; n_shots = 1000, timesteps = [0.087, 0.032, 0.084, 0.054], create_folder = true, root_dir = EXPERIMENT_ROOT_DIR)
    tasks = []
    today = Dates.format(Dates.now(), "yyyy-mm-dd-HH.MM.SS")
    @info "Creating experiment $name - $today"
    @info "Loaded $(length(tasks)) tasks"
    @info "Selected timesteps: $timesteps"
    @info "Selected n_shots: $n_shots"
    experiment_dir = joinpath(root_dir, "$(name)_$(today)")
    if create_folder
        mkpath(experiment_dir)
        mkdir(joinpath(experiment_dir, "results"))
        mkdir(joinpath(experiment_dir, "waveforms"))
        mkdir(joinpath(experiment_dir, "tasks"))
    end
    return Experiment("$(name)_$(today)", tasks, Dict(), n_shots, timesteps, experiment_dir)
end

function create_task!(experiment::Experiment, data::AbstractTask)
    experiment.tasks_id[data.data_id] = data;
    push!(experiment.tasks, data);
end

function create_task!(experiment::Experiment, h_t::Array{RydbergHamiltonian, 1}; copies::Int = 1)
    # retrieve the data and construct the hamiltonian given the experiment parameters
    data_id = length(experiment.tasks) + 1
    
    @info "Validating data $data_id"
    arns::Array{String, 1} = []
    n_shots::Int = floor(Int, experiment.nshots/copies)
    for h in h_t
        hardwarevalidate(h; verbosity = LOW)
        @info "Submitting data $data_id"    
        schema_params = getschemaparams(n_shots, BloqadeSchema.get_device_capabilities())
        schema = getschema(h, schema_params)
        task = submittobraket(schema)
        push!(arns, task.arn)
    end
    # save the task object
    gt = DataTask(arns, data_id, n_shots, h_t, copies);
    experiment.tasks_id[data_id] = gt;
    push!(experiment.tasks, gt);
end

function create_task!(experiment::Experiment, h_t::Array{RydbergHamiltonian, 1}, countedmeasures_, data_id::Int; copies::Int = 1)
    # retrieve the data and construct the hamiltonian given the experiment parameters
    #data_id = length(experiment.tasks) + 1
    
    @info "Saving data $data_id"
    arns::Array{String, 1} = []
    n_shots::Int = floor(Int, experiment.nshots/copies)
    mkdir(joinpath(experiment.experiment_dir, "results", "task_$(data_id)"))
    for i in range(1, length(countedmeasures_))
        #schema_params = getschemaparams(n_shots, BloqadeSchema.get_device_capabilities())
        #schema = getschema(h, schema_params)
        #task = submittobraket(schema)
        save_object(joinpath(experiment.experiment_dir, "results", "task_$(data_id)", "task_$(data_id)_step_$(experiment.timesteps[i]).jld2"), countedmeasures_[i])
        push!(arns, joinpath(experiment.experiment_dir, "results", "task_$(data_id)", "task_$(data_id)_step_$(experiment.timesteps[i]).jld2"))
    end
    # save the task object
    gt = DataTask(arns, data_id, n_shots, h_t, copies);
    experiment.tasks_id[data_id] = gt;
    push!(experiment.tasks, gt);
end

function save_experiment(experiment::Experiment)
    save_object(joinpath(experiment.experiment_dir, "experiment.jld2"), experiment)
end

function load_experiment(path)
    return load_object(path)
end
    
function save_task(experiment::Experiment, data_id::Int; direct_access::Bool = false)
    gt = experiment.tasks_id[data_id]
    folder_str = joinpath(experiment.experiment_dir, "results", "task_$(data_id)")
    blo = pyimport("bloqade")
    if !isfile(joinpath(folder_str, "raw_data", "results_$(data_id).json")) && !direct_access
        @info "Retrieving tasks for Datapoint $data_id"
        tasks = blo.load(gt.arn).fetch()
        if length(tasks.get_completed_tasks().tasks) != length(experiment.timesteps) - 1
            @info "Tasks status for Datapoint $data_id:"
            println(tasks)
        end
        if length(tasks.get_completed_tasks().tasks) + length(tasks.get_failed_tasks().tasks) == length(experiment.timesteps) - 1
            @info "Datapoint $data_id: $(length(tasks.get_failed_tasks().tasks)) tasks completed. $(length(tasks.get_failed_tasks().tasks)) tasks failed. Saving data."
            mkdir(folder_str)
            mkdir(joinpath(folder_str, "raw_data"))
            blo.save(
                tasks,
                joinpath(folder_str, "raw_data", "results_$(data_id).json"),
            )
        else
            @info "Pending tasks: skipping datapoint"
            return
        end
    elseif !isfile(joinpath(folder_str, "raw_data", "results_$(data_id).json")) && direct_access
        @info "Missing direct access tasks for Datapoint $data_id. Skipping."
        return
    else
        @info "Tasks for Datapoint $data_id already completed"
    end
    
    tasks = blo.load(joinpath(folder_str, "raw_data", "results_$(data_id).json"))
    if !direct_access
        tasks = [tasks.get_completed_tasks().tasks[i].task_result_ir.shot_outputs for i in tasks.get_completed_tasks().tasks]
        tasks = [pylist([pydict(pre_sequence = j.pre_sequence, post_sequence = j.post_sequence) for j in task]) for task in tasks]
    else
        tasks = [task["shot_outputs"] for task in tasks["tasks"]]
    end

    #for i in tasks.get_completed_tasks().tasks
    #    save_object(joinpath(folder_str, "task_$(data_id)_step_$(experiment.timesteps[pyconvert(Int64,i)+1]).jld2"), bitstrings_filtered(
    #        [pyconvert(Vector{Int64}, j.pre_sequence) for j in tasks.get_completed_tasks().tasks[i].task_result_ir.shot_outputs], 
    #        [pyconvert(Vector{Int64}, j.post_sequence) for j in tasks.get_completed_tasks().tasks[i].task_result_ir.shot_outputs], 
    #        gt.copies
    #        ))
    #end
    
    for i in range(1, length(tasks))
        save_object(joinpath(folder_str, "task_$(data_id)_step_$(experiment.timesteps[i]).jld2"), bitstrings_filtered(
            [pyconvert(Vector{Int64}, j["pre_sequence"]) for j in tasks[i]], 
            [pyconvert(Vector{Int64}, j["post_sequence"]) for j in tasks[i]], 
            gt.copies
            ))
    end
    #for i in range(1, length(gt.arn))
    #    file_str = joinpath(experiment.experiment_dir, "results", "task_$(data_id)", "task_$(data_id)_step_$(experiment.timesteps[i]).jld2")
    #    if !isfile(file_str)
    #        @warn "Results of task $(gt.arn[i]) is already present"
    #    elseif gettaskstatus(gt.arn[i]) == "COMPLETED"
    #        results = gettaskresults(gt.arn[i], is_filter_invalid = true)
    #        save_object(file_str, results)
    #    else
    #        @warn "Task $(gt.arn[i]) is not completed"
    #    end
    ## save_object(joinpath(experiment.experiment_dir, "results", "task_$(data_id).jld2"), gt)
    #end 
end

function resume_experiment(experiment_folder)
    return load_experiment(joinpath(experiment_folder, "experiment.jld2"))
end

function check_experiment_status(experiment::Experiment)
    completed = 0
    tot = 0
    for gt in experiment.tasks
        @info "Task $(gt.arn)"
        blo = pyimport("bloqade")
        task = blo.load(gt.arn)
        
        #for arn in gt.arn
        #    tot += 1
        #    @info "Status: $(gettaskstatus(arn))"
        #    if gettaskstatus(arn) == "COMPLETED"
        #        completed += 1
        #    end
        #end
    end
    @info "Completed $completed tasks out of $tot"
end

"""
    subsample_rng(seed, n_shots, data_id)

RNG for drawing the `n_shots` subsample of one datapoint's measurement records.
Seeded from all three values, so a datapoint's subsample depends neither on
iteration order nor on which other datapoints are processed in the same pass.
`seed = nothing` returns the global RNG (unseeded).
"""
subsample_rng(seed::Nothing, n_shots, data_id) = Random.default_rng()
subsample_rng(seed::Integer, n_shots, data_id) = MersenneTwister(UInt32[seed, max(n_shots, 0), data_id])

function compute_exp_values(experiment::Experiment; n_shots::Int64 = -1, seed = nothing, dataset_dir::String = "datasets")
    out_dir = joinpath(experiment.experiment_dir, dataset_dir)
    mkpath(out_dir)
    for task in sort(collect(experiment.tasks_id), by=x->x[1])
        if !isfile(joinpath(out_dir, "task_$(task[1]).csv"))
            try
                rng = subsample_rng(seed, n_shots, task[1])
                CSV.write(joinpath(out_dir, "task_$(task[1]).csv"), compute_exp_value(experiment, task[2], n_shots; rng = rng))
            catch
                @info "Results for datapoint $(task[1]) not available"
            end
        end
    end
end

"""
    experiment_features(experiment; n_shots = -1, seed = nothing, ids = nothing)

Reduce the stored measurement records of an experiment to one feature matrix,
one column per datapoint, without writing intermediate per-datapoint files.
Returns `(ids, features, missing)`: the datapoint IDs in column order, the
`n_features × n_datapoints` matrix, and the IDs skipped for lack of results.

`n_shots > 0` subsamples each datapoint's records, seeded per datapoint as in
`subsample_rng`. `ids` restricts and orders the datapoints; by default all of
them are used in ascending ID order.
"""
function experiment_features(experiment::Experiment; n_shots::Int64 = -1, seed = nothing, ids = nothing)
    wanted = isnothing(ids) ? sort(collect(keys(experiment.tasks_id))) : collect(ids)
    columns = Vector{Vector{Float64}}()
    kept = Int[]
    skipped = Int[]
    for id in wanted
        task = get(experiment.tasks_id, id, nothing)
        if isnothing(task)
            push!(skipped, id)
            continue
        end
        df = try
            compute_exp_value(experiment, task, n_shots; rng = subsample_rng(seed, n_shots, id))
        catch err
            @warn "Datapoint $id: results unavailable" exception = err
            nothing
        end
        if isnothing(df)
            push!(skipped, id)
            continue
        end
        push!(columns, Float64.(df[:, 1]))
        push!(kept, id)
    end
    features = isempty(columns) ? zeros(0, 0) : reduce(hcat, columns)
    return kept, features, skipped
end

function compute_exp_value(experiment::Experiment, task::ExpDataTask, n_shots::Int64; rng = Random.default_rng())
    embeddings_set = []
    folder_str = joinpath(experiment.experiment_dir, "results", "task_$(task.data_id)")
    if !isdir(folder_str)
        @error "Results for datapoint $(task.data_id) not present"
        return
    end
    arns = [joinpath(folder_str, "task_$(task.data_id)_step_$(experiment.timesteps[i]).jld2") for i in range(1, length(experiment.timesteps)-1)]
    for arn in arns
        measurements = load_object(arn)
        if n_shots > 0
            measurements = sample(rng, measurements, n_shots; replace=false)
        end
        uniquemeasures = unique(measurements)
        countedmeasures = map(m -> (m, count(x -> x == m, measurements)), uniquemeasures)
        append!(embeddings_set, compute_zi(countedmeasures))
        zij = compute_zij(countedmeasures)
        dim = size(zij, 1)
        for i in 1:dim
            append!(embeddings_set, zij[i, i+1:dim])
        end
    end
    return DataFrame(hcat(embeddings_set), :auto)
end

function compute_exp_value(experiment::Experiment, task::DataTask, n_shots::Int64; rng = Random.default_rng())
    embeddings_set = []
    for arn in task.arn
        measurements = load_object(arn)
        if n_shots > 0
            measurements = sample(rng, measurements, n_shots; replace=false)
        end
        uniquemeasures = unique(measurements)
        countedmeasures = map(m -> (m, count(x -> x == m, measurements)), uniquemeasures)
        append!(embeddings_set, compute_zi(countedmeasures))
        zij = compute_zij(countedmeasures)
        dim = size(zij, 1)
        for i in 1:dim
            append!(embeddings_set, zij[i, i+1:dim])
        end
    end
    return DataFrame(hcat(embeddings_set), :auto)
end

end