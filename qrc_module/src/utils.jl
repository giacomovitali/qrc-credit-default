module qcutils
export rebin, compute_distribution, compute_occupation, resize_distribution
using BitBasis
using JSON
C6 = 0.862690 * 1e6 * 2 * π

function compute_occupation(measurements, N)
    res = [0 for i in 1:(N+1)]
    #res = [0 for i in 1:(maximum(measurements) + 1)]
    for measure in measurements
        index = sum([j for j in measure])
        res[index + 1] += 1
    end
    return res
end

function compute_distribution(measurements, N)
    res = zeros(2^N)
    for measure in measurements
        res += real(onehot(measure))
    end
    return res
end

function compute_zi(measurements)
    zi = [0 for i in 1:length(measurements[1][1])]
    tot_measurements = 0
    for measurement in measurements
        zi += (-1).^[measurement[1]...] .*= measurement[2]
        tot_measurements += measurement[2]
    end
    return zi ./ tot_measurements
end

function compute_zij(measurements)
    dim = length(measurements[1][1])
    zij = zeros(dim, dim)
    tot_measurements = 0
    for measurement in measurements
        M = zeros(dim, dim)
        for i in 1:dim
            for j in i+1:dim
                M[i,j] = ((-1)^(measurement[1][i] + measurement[1][j])) * measurement[2]
            end
        end
        tot_measurements += measurement[2]
        zij += M
    end
    return zij ./ tot_measurements
end

end

module HardwareUtils

using Bloqade, BloqadeSchema
using Braket, AWS

@enum VerbosityLevel begin
    NONE = 0
    LOW = 1
    HIGH = 2
end

function show_validation_info(validate_info; verbosity = NONE)
    if verbosity == LOW
        @info "Lattice violations: $(length(validate_info.lattice_violations))"
        @info "Ω violations: $(length(validate_info.Ω_violations))"
        @info "Δ violations: $(length(validate_info.Δ_violations))"
        @info "ϕ violations: $(length(validate_info.ϕ_violations))"
    end
    if verbosity == HIGH
        @info "Lattice violations: $(validate_info.lattice_violations)"
        @info "Ω violations: $(validate_info.Ω_violations)"
        @info "Δ violations: $(validate_info.Δ_violations)"
        @info "ϕ violations: $(validate_info.ϕ_violations)"
    end
end


function hardwarevalidate(h; verbosity = HIGH)
    validate_info = validate(h)
    show_validation_info(validate_info; verbosity=verbosity)
end

function submittobraket(schema::BloqadeSchema.QuEraTaskSpecification)
    credentials = AWS.AWSCredentials() # use default credentials provided by aws cli
    task = submit_to_braket(schema; credentials = credentials)
    return task
end

function gettaskstatus(task::String)
    awstask = AwsQuantumTask(task)
    return gettaskstatus(awstask)
end

function gettaskstatus(awstask::AwsQuantumTask)
    return Braket.state(awstask)
end

function bitstrings_filtered(pre_sequences::Vector{Vector{Int64}}, post_sequences::Vector{Vector{Int64}}, copies::Int)
    num_bits::Int = length(pre_sequences[1])
    dim::Int = num_bits/copies
    bitstrings_filtered::Vector{DitStr{2, dim, Int64}} = []
    for i in range(1, length(pre_sequences))#measurement in post_sequences
        unfiltered_bitstrings = [Yao.neg(Yao.bit_literal(post_sequences[i][j-(dim-1):j]...)) for j in dim:dim:num_bits]
        pre_sequence_bitstrings = [Yao.bit_literal(pre_sequences[i][j-(dim-1):j]...) for j in dim:dim:num_bits]
        for j in range(1, copies)
            if Yao.bmask(pre_sequence_bitstrings[j]) == dim
                push!(bitstrings_filtered, unfiltered_bitstrings[j])
            end
        end
    end
    return bitstrings_filtered
end

function gettaskresults(task::String)
    awstask = AwsQuantumTask(task)
    task_status = gettaskstatus(awstask)
    if (task_status != "COMPLETED")
        @error "The task is not completed"
        throw(ErrorException("The task is not completed"))
    end
    @info "Retrieving task results"
    return task_results = Braket.result(awstask)
end


function gettaskresults(task::String, is_filter_invalid::Bool)
    awstask = AwsQuantumTask(task)
    gettaskresults(awstask; is_filter_invalid=is_filter_invalid)
end

function gettaskresults(awstask::AwsQuantumTask; is_filter_invalid::Bool=false)
    task_status = gettaskstatus(awstask)
    if (task_status != "COMPLETED")
        @error "The task is not completed"
        throw(ErrorException("The task is not completed"))
    end
    @info "The task is completed"
    task_results = Braket.result(awstask)
    task_measurements = task_results.measurements
    num_bits = length(task_measurements[1].pre_sequence)
    measurements = [Yao.neg(Yao.bit_literal(x.post_sequence...)) for x in task_measurements if !is_filter_invalid || Yao.bmask(Yao.bit_literal(x.pre_sequence...)) == num_bits]
    unique_measurements = unique(measurements)
    counted_measurements = map(m -> (m, count(x -> x == m, measurements)), unique_measurements)
    sorted_solutions = sort(counted_measurements, rev=true, by=x -> x[2])
    return measurements
end

function getschemaparams(nshots::Integer, devicecapabilities::BloqadeSchema.DeviceCapabilities)
    return BloqadeSchema.SchemaTranslationParams(nshots, devicecapabilities)
end

function getschema(transformedhamiltonian::RydbergHamiltonian, schemaparams::BloqadeSchema.SchemaTranslationParams)
    return to_schema(transformedhamiltonian, schemaparams)
end

end
