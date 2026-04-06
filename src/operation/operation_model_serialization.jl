const _SERIALIZED_MODEL_FILENAME = "model.bin"
const _SERIALIZED_HDF5_MODEL_FILENAME = "model.h5"

struct OptimizerAttributes
    name::String
    version::String
    attributes::Any
end

function OptimizerAttributes(model::OperationModel, optimizer::MOI.OptimizerWithAttributes)
    jump_model = get_jump_model(model)
    name = JuMP.solver_name(jump_model)
    # Note that this uses private field access to MOI.OptimizerWithAttributes because there
    # is no public method available.
    # This could break if MOI changes their implementation.
    try
        version = MOI.get(JuMP.backend(jump_model), MOI.SolverVersion())
        return OptimizerAttributes(name, version, optimizer.params)
    catch
        @debug "Solver Version not supported by the solver"
        version = "MOI.SolverVersion not supported"
        return OptimizerAttributes(name, version, optimizer.params)
    end
end

function _get_optimizer_attributes(model::OperationModel)
    return get_optimizer(get_settings(model)).params
end

struct ProblemSerializationWrapper
    template::ProblemTemplate
    sys::Union{Nothing, String}
    settings::Settings
    model_type::DataType
    name::String
    optimizer::OptimizerAttributes
end

function serialize_problem(model::OperationModel; optimizer = nothing)
    _serialize_problem_to_file(model, get_store(model); optimizer = optimizer)
end

function _serialize_problem_to_file(
    model::OperationModel,
    store::AbstractModelStore;
    optimizer = nothing,
)
    # A PowerSystem cannot be serialized in this format because of how it stores
    # time series data. Use its specialized serialization method instead.
    sys_to_file = get_system_to_file(get_settings(model))
    if sys_to_file
        sys = get_system(model)
        sys_filename = joinpath(get_output_dir(model), make_system_filename(sys))
        # Skip serialization if the system is already in the folder
        !ispath(sys_filename) && PSY.to_json(sys, sys_filename)
    else
        sys_filename = nothing
    end
    container = get_optimization_container(model)

    if optimizer === nothing
        optimizer = get_optimizer(get_settings(model))
        @assert optimizer !== nothing "optimizer must be passed if it wasn't saved in Settings"
    end

    obj = ProblemSerializationWrapper(
        model.template,
        sys_filename,
        container.settings_copy,
        typeof(model),
        string(get_name(model)),
        OptimizerAttributes(model, optimizer),
    )
    bin_file_name = joinpath(get_output_dir(model), _SERIALIZED_MODEL_FILENAME)
    Serialization.serialize(bin_file_name, obj)
    @info "Serialized OperationModel to" bin_file_name
end

# Convert Settings fields to HDF5-compatible scalar values via multiple dispatch.
_settings_field_to_hdf5(val::Base.RefValue) = _settings_field_to_hdf5(val[])
_settings_field_to_hdf5(val::Dates.Millisecond) = val.value
_settings_field_to_hdf5(val::Dates.DateTime) = string(val)
_settings_field_to_hdf5(val::Bool) = val
_settings_field_to_hdf5(val::Number) = val
_settings_field_to_hdf5(val::AbstractString) = val
_settings_field_to_hdf5(::Nothing) = "nothing"
_settings_field_to_hdf5(val::Dict) = string(val)
_settings_field_to_hdf5(val) = string(val)

function _serialize_problem_to_file(
    model::OperationModel,
    store::EmulationModelStore{HDF5Dataset};
    optimizer = nothing,
)
    output_dir = get_output_dir(model)
    h5_file_path = joinpath(output_dir, _SERIALIZED_HDF5_MODEL_FILENAME)

    if optimizer === nothing
        optimizer = get_optimizer(get_settings(model))
        @assert optimizer !== nothing "optimizer must be passed if it wasn't saved in Settings"
    end

    container = get_optimization_container(model)
    opt_attrs = OptimizerAttributes(model, optimizer)

    HDF5.h5open(h5_file_path, "w") do h5_file
        # Embed system JSON directly in HDF5 (always, regardless of system_to_file setting)
        sys = get_system(model)
        sys_json_tmp = joinpath(mktempdir(), make_system_filename(sys))
        PSY.to_json(sys, sys_json_tmp)
        h5_file["system_json"] = read(sys_json_tmp, String)

        # Model metadata
        h5_file["name"] = string(get_name(model))
        h5_file["model_type"] = string(typeof(model))

        # Settings — each field as its own dataset
        settings = container.settings_copy
        sg = HDF5.create_group(h5_file, "settings")
        for name in fieldnames(Settings)
            sg[string(name)] = _settings_field_to_hdf5(getfield(settings, name))
        end

        # Optimizer attributes
        og = HDF5.create_group(h5_file, "optimizer")
        og["name"] = opt_attrs.name
        og["version"] = opt_attrs.version
        og["attributes"] = string(opt_attrs.attributes)

        # Template — structured groups with formulation type names
        tg = HDF5.create_group(h5_file, "template")
        template = model.template
        tg["network_model_type"] = string(get_network_formulation(template))

        dg = HDF5.create_group(tg, "devices")
        for (sym, dm) in template.devices
            dg[string(sym)] = string(get_formulation(dm))
        end

        bg = HDF5.create_group(tg, "branches")
        for (sym, bm) in template.branches
            bg[string(sym)] = string(get_formulation(bm))
        end

        svg = HDF5.create_group(tg, "services")
        for ((sname, sym), sm) in template.services
            svg["$(sname)__$(sym)"] = string(get_formulation(sm))
        end
    end
    @info "Serialized OperationModel to HDF5" h5_file_path
end

function deserialize_problem(
    ::Type{T},
    directory::AbstractString;
    kwargs...,
) where {T <: OperationModel}
    filename = joinpath(directory, _SERIALIZED_MODEL_FILENAME)
    if !isfile(filename)
        error("$directory does not contain a serialized model")
    end
    obj = Serialization.deserialize(filename)
    if !(obj isa ProblemSerializationWrapper)
        throw(IS.DataFormatError("deserialized object has incorrect type $(typeof(obj))"))
    end
    sys = get(kwargs, :system, nothing)

    if sys === nothing
        if obj.sys === nothing && !settings[:sys_to_file]
            throw(
                IS.DataFormatError(
                    "Operations Problem System was not serialized and a System has not been specified.",
                ),
            )
        elseif !ispath(obj.sys)
            throw(IS.DataFormatError("PowerSystems.System file $(obj.sys) does not exist"))
        end
        sys = PSY.System(obj.sys)
    end
    settings =
        Settings(sys; restore_from_copy(obj.settings; optimizer = kwargs[:optimizer])...)
    model =
        obj.model_type(obj.template, sys, settings, kwargs[:jump_model]; name = obj.name)
    jump_model = get_jump_model(model)
    if obj.optimizer.name == JuMP.solver_name(jump_model)
        orig_attrs = obj.optimizer.attributes
        new_attrs = _get_optimizer_attributes(model)
        if length(orig_attrs) != length(new_attrs)
            @warn "Different optimizer attributes are set. Original: $orig_attrs New: $new_attrs"
        else
            for attrs in (orig_attrs, new_attrs)
                sort!(attrs; by = x -> x.first.name)
            end
            for i in 1:length(orig_attrs)
                name = orig_attrs[i].first.name
                orig = orig_attrs[i].second
                new = new_attrs[i].second
                if new != orig
                    @warn "Original solver used $name = $orig. New solver uses $new."
                end
            end
        end
    else
        @warn "Original solver was $(obj.optimizer.name), new solver is $(JuMP.solver_name(jump_model))"
    end

    return model
end
