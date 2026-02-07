# Test types defined in test_utils/test_types.jl
import InfrastructureOptimizationModels:
    VariableKey,
    OptimizationContainerMetadata,
    encode_key_as_string

@testset "Testset Optimization Container Metadata" begin
    metadata = OptimizationContainerMetadata()
    var_key = VariableKey(MockVariable, IS.TestComponent)
    IOM.add_container_key!(metadata, encode_key_as_string(var_key), var_key)
    @test IOM.has_container_key(metadata, encode_key_as_string(var_key))
    @test IOM.get_container_key(metadata, encode_key_as_string(var_key)) ==
          var_key
    file_dir = mktempdir()
    model_name = :MockModel
    IOM.serialize_metadata(file_dir, metadata, model_name)
    file_path = IOM._make_metadata_filename(model_name, file_dir)
    deserialized_metadata = IOM.deserialize_metadata(
        OptimizationContainerMetadata,
        file_dir,
        model_name,
    )
    @test IOM.has_container_key(
        deserialized_metadata,
        encode_key_as_string(var_key),
    )
    key = IOM.deserialize_key(
        metadata,
        "MockVariable__TestComponent",
    )
    @test key == var_key
end
