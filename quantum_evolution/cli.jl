# Shared command-line handling for the quantum_evolution drivers.
#
#   julia --project=. quantum_evolution/<driver>.jl [CONFIG.toml] [key=value ...]
#
# Settings start from the driver's defaults, are overlaid with the optional TOML
# config file, and finally with `key=value` overrides, in that order whatever
# order the arguments are given in. Override values are parsed as TOML, so
# 7, 2.5, true and [100, 240] keep their types; anything that does not parse
# (e.g. position) is taken as a string. Dotted keys address tables:
# layer.delta=2.0.
#
# The defaults double as the schema: a key the driver does not define is an
# error, so a typo cannot silently fall back to a default.

using TOML

function parse_override_value(raw::AbstractString)
    try
        return TOML.parse("v = " * raw)["v"]
    catch
        return String(raw)
    end
end

function set_dotted!(d::AbstractDict, key::AbstractString, value)
    parts = split(key, '.')
    node = d
    for p in parts[1:end-1]
        node = get!(node, String(p), Dict{String, Any}())
        node isa AbstractDict || error("Setting `$key`: `$p` is not a table")
    end
    node[String(parts[end])] = value
    return d
end

function merge_nested!(dst::AbstractDict, src::AbstractDict)
    for (k, v) in src
        if v isa AbstractDict && get(dst, k, nothing) isa AbstractDict
            merge_nested!(dst[k], v)
        else
            dst[k] = v
        end
    end
    return dst
end

function check_known_keys(settings::AbstractDict, schema::AbstractDict, prefix = "")
    for (k, v) in settings
        haskey(schema, k) || error("Unknown setting `$(prefix)$(k)`. Known settings: $(join(sort(collect(keys(schema))), ", "))")
        if v isa AbstractDict && schema[k] isa AbstractDict
            check_known_keys(v, schema[k], "$(prefix)$(k).")
        end
    end
end

"""
    load_settings(defaults; args = ARGS) -> (settings, config_file)

Resolve the driver settings from `defaults`, an optional `.toml` file and
`key=value` overrides in `args`.
"""
function load_settings(defaults::AbstractDict; args = ARGS)
    config_files = filter(a -> !occursin('=', a), args)
    overrides = filter(a -> occursin('=', a), args)
    length(config_files) <= 1 || error("At most one config file may be given, got: $(join(config_files, ", "))")
    for f in config_files
        endswith(f, ".toml") || error("Unrecognised argument `$f`: expected a .toml config file or key=value")
    end

    settings = deepcopy(defaults)
    config_file = isempty(config_files) ? nothing : only(config_files)
    if !isnothing(config_file)
        from_file = TOML.parsefile(config_file)
        check_known_keys(from_file, defaults)
        merge_nested!(settings, from_file)
    end
    for o in overrides
        k, v = split(o, '=', limit = 2)
        k = strip(k)
        set_dotted!(settings, k, parse_override_value(strip(v)))
    end
    check_known_keys(settings, defaults)
    return settings, config_file
end

# Comma-separated strings are accepted wherever a list is expected, so that
# lists can be passed on the command line without TOML quoting.
as_list(x::AbstractVector) = collect(x)
as_list(x::AbstractString) = [parse_override_value(strip(s)) for s in split(x, ',') if !isempty(strip(s))]
as_list(x) = [x]

using SHA: sha256
file_sha256(path::AbstractString) = bytes2hex(open(sha256, path))
