import InfrastructureOptimizationModels: ModelInternal
@testset "Test Model Internal" begin
    internal = ModelInternal(
        MockContainer(),
    )
    @test IOM.get_status(internal) == IOM.ModelBuildStatus.EMPTY
    IOM.set_initial_conditions_model_container!(
        internal,
        MockContainer(),
    )
    @test isa(
        IOM.get_initial_conditions_model_container(internal),
        MockContainer,
    )
    IOM.add_recorder!(internal, :MockRecorder)
    @test IOM.get_recorders(internal)[1] == :MockRecorder
    IOM.set_status!(internal, IOM.ModelBuildStatus.BUILT)
    @test IOM.get_status(internal) == IOM.ModelBuildStatus.BUILT
    IOM.set_output_dir!(internal, mktempdir())
    log_config = IOM.configure_logging(internal, "test_log.log", "a")
    @test !isempty(log_config.loggers)
end
