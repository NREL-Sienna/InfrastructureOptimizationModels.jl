
const DevicesModelContainer = Dict{Symbol, DeviceModel}
const ServicesModelContainer = Dict{Tuple{String, Symbol}, ServiceModel}

abstract type AbstractProblemTemplate end

# Interface stubs — concrete implementations provided by downstream packages
function get_device_models end
function get_branch_models end
function get_service_models end
function get_network_model end
function get_network_formulation end
function get_hvdc_network_model end
function get_component_types end
function get_model end
function set_network_model! end
function set_hvdc_network_model! end
function set_device_model! end
function set_service_model! end
function finalize_template! end

"""
Return the set of device types whose formulation is FixedOutput (incompatible with
service provision).
"""
function get_incompatible_devices(devices_template::Dict)
    incompatible_device_types = Set{DataType}()
    for model in values(devices_template)
        formulation = get_formulation(model)
        if formulation == FixedOutput
            if !isempty(get_services(model))
                @info "$(formulation) for $(get_component_type(model)) is not compatible with the provision of reserve services"
            end
            push!(incompatible_device_types, get_component_type(model))
        end
    end
    return incompatible_device_types
end
