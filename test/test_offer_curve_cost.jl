#=
Tests for MarketBidCost / MarketBidTimeSeriesCost / ImportExportCost /
ImportExportTimeSeriesCost code paths in value_curve_cost.jl and start_up_shut_down.jl.

Uses c_sys5_uc as a base system and attaches MBC/IEC costs with real time series.

Exercises: accessor wrappers, detection predicates, _has_parameter_time_series,
validation, _shutdown_cost_value, _get_parameter_field, get_initial_input,
validate_occ_breakpoints_slopes, validate_occ_component.
=#

import PowerSystemCaseBuilder: PSITestSystems
using DataStructures: OrderedDict

# ─── system builders ──────────────────────────────────────────────────────────

"""Build a deterministic time series matching the system's forecast parameters."""
function _make_ts(sys::PSY.System, name::String, value::Float64)
    init_time = first(PSY.get_forecast_initial_times(sys))
    horizon = PSY.get_forecast_horizon(sys)
    interval = PSY.get_forecast_interval(sys)
    resolution = first(PSY.get_time_series_resolutions(sys))
    count = PSY.get_forecast_window_count(sys)
    horizon_count = IS.get_horizon_count(horizon, resolution)
    data = OrderedDict{Dates.DateTime, Vector{Float64}}()
    for i in 0:(count - 1)
        data[init_time + i * interval] = fill(value, horizon_count)
    end
    return PSY.Deterministic(; name = name, data = data, resolution = resolution)
end

"""Build a deterministic time series of NTuple{3, Float64} matching the system's forecast params."""
function _make_tuple_ts(sys::PSY.System, name::String, value::NTuple{3, Float64})
    init_time = first(PSY.get_forecast_initial_times(sys))
    horizon = PSY.get_forecast_horizon(sys)
    interval = PSY.get_forecast_interval(sys)
    resolution = first(PSY.get_time_series_resolutions(sys))
    count = PSY.get_forecast_window_count(sys)
    horizon_count = IS.get_horizon_count(horizon, resolution)
    data = OrderedDict{Dates.DateTime, Vector{NTuple{3, Float64}}}()
    for i in 0:(count - 1)
        data[init_time + i * interval] = fill(value, horizon_count)
    end
    return PSY.Deterministic(; name = name, data = data, resolution = resolution)
end

"""Build a deterministic time series of PiecewiseStepData matching the system's forecast params."""
function _make_pwl_ts(
    sys::PSY.System,
    name::String,
    breakpoints::Vector{Float64},
    slopes::Vector{Float64},
)
    init_time = first(PSY.get_forecast_initial_times(sys))
    horizon = PSY.get_forecast_horizon(sys)
    interval = PSY.get_forecast_interval(sys)
    resolution = first(PSY.get_time_series_resolutions(sys))
    count = PSY.get_forecast_window_count(sys)
    horizon_count = IS.get_horizon_count(horizon, resolution)
    psd = PSY.PiecewiseStepData(breakpoints, slopes)
    data = OrderedDict{Dates.DateTime, Vector{PSY.PiecewiseStepData}}()
    for i in 0:(count - 1)
        data[init_time + i * interval] = fill(psd, horizon_count)
    end
    return PSY.Deterministic(; name = name, data = data, resolution = resolution)
end

"""
Load c_sys5_uc, pick a ThermalStandard, give it a static MarketBidCost,
and return (sys, component).
"""
function _make_static_mbc_system(;
    slopes = [25.0, 30.0],
    breakpoints = [0.0, 50.0, 100.0],
    initial_input = 10.0,
    no_load = 5.0,
    start_up = (hot = 100.0, warm = 200.0, cold = 300.0),
    shut_down = 50.0,
)
    sys = Logging.with_logger(Logging.NullLogger()) do
        build_system(PSITestSystems, "c_sys5_uc")
    end
    comp = first(PSY.get_components(PSY.ThermalStandard, sys))
    incr_vc = PSY.PiecewiseIncrementalCurve(
        PSY.PiecewiseStepData(breakpoints, slopes), initial_input, nothing)
    decr_vc = PSY.PiecewiseIncrementalCurve(
        PSY.PiecewiseStepData(breakpoints, reverse(slopes)), initial_input, nothing)
    mbc = PSY.MarketBidCost(;
        no_load_cost = PSY.LinearCurve(no_load),
        start_up = start_up,
        shut_down = PSY.LinearCurve(shut_down),
        incremental_offer_curves = PSY.CostCurve(incr_vc),
        decremental_offer_curves = PSY.CostCurve(decr_vc),
    )
    PSY.set_operation_cost!(comp, mbc)
    return sys, comp
end

"""
Load c_sys5_uc, pick a ThermalStandard, give it a MarketBidTimeSeriesCost
with real time series attached, and return (sys, component).
"""
function _make_ts_mbc_system(;
    slopes = [25.0, 30.0],
    breakpoints = [0.0, 50.0, 100.0],
    initial_input = 10.0,
    no_load = 5.0,
    start_up_val = (100.0, 100.0, 100.0),
    shut_down = 50.0,
)
    sys = Logging.with_logger(Logging.NullLogger()) do
        build_system(PSITestSystems, "c_sys5_uc")
    end
    comp = first(PSY.get_components(PSY.ThermalStandard, sys))

    # Create and attach time series
    incr_ts = _make_pwl_ts(sys, "variable_cost incremental", breakpoints, slopes)
    incr_key = PSY.add_time_series!(sys, comp, incr_ts)
    incr_init_ts = _make_ts(sys, "initial_input incremental", initial_input)
    incr_init_key = PSY.add_time_series!(sys, comp, incr_init_ts)

    decr_ts = _make_pwl_ts(sys, "variable_cost decremental", breakpoints, reverse(slopes))
    decr_key = PSY.add_time_series!(sys, comp, decr_ts)
    decr_init_ts = _make_ts(sys, "initial_input decremental", initial_input)
    decr_init_key = PSY.add_time_series!(sys, comp, decr_init_ts)

    no_load_ts = _make_ts(sys, "no_load_cost", no_load)
    no_load_key = PSY.add_time_series!(sys, comp, no_load_ts)

    shut_down_ts = _make_ts(sys, "shut_down", shut_down)
    shut_down_key = PSY.add_time_series!(sys, comp, shut_down_ts)

    startup_ts = _make_tuple_ts(sys, "start_up", start_up_val)
    startup_key = PSY.add_time_series!(sys, comp, startup_ts)

    ts_mbc = PSY.MarketBidTimeSeriesCost(;
        no_load_cost = PSY.TimeSeriesLinearCurve(no_load_key),
        start_up = IS.TupleTimeSeries{PSY.StartUpStages}(startup_key),
        shut_down = PSY.TimeSeriesLinearCurve(shut_down_key),
        incremental_offer_curves = PSY.make_market_bid_ts_curve(incr_key, incr_init_key),
        decremental_offer_curves = PSY.make_market_bid_ts_curve(decr_key, decr_init_key),
    )
    PSY.set_operation_cost!(comp, ts_mbc)
    return sys, comp
end

"""
Load c_sys5_uc, add a Source with static ImportExportCost, return (sys, source).
"""
function _make_static_iec_system()
    sys = Logging.with_logger(Logging.NullLogger()) do
        build_system(PSITestSystems, "c_sys5_uc")
    end
    bus = first(PSY.get_components(PSY.ACBus, sys))
    source = PSY.Source(;
        name = "test_source",
        available = true,
        bus = bus,
        active_power = 0.0,
        reactive_power = 0.0,
        active_power_limits = (min = -2.0, max = 2.0),
        reactive_power_limits = (min = -2.0, max = 2.0),
        R_th = 0.01,
        X_th = 0.02,
        internal_voltage = 1.0,
        internal_angle = 0.0,
        base_power = 100.0,
    )
    import_curve = PSY.make_import_curve(
        [0.0, 100.0, 105.0, 120.0, 200.0], [5.0, 10.0, 20.0, 40.0])
    export_curve = PSY.make_export_curve(
        [0.0, 100.0, 105.0, 120.0, 200.0], [12.0, 8.0, 4.0, 1.0])
    iec = PSY.ImportExportCost(;
        import_offer_curves = import_curve,
        export_offer_curves = export_curve,
    )
    PSY.set_operation_cost!(source, iec)
    PSY.add_component!(sys, source)
    return sys, source
end

"""
Load c_sys5_uc, add a Source with ImportExportTimeSeriesCost backed by real TS,
return (sys, source).
"""
function _make_ts_iec_system()
    sys = Logging.with_logger(Logging.NullLogger()) do
        build_system(PSITestSystems, "c_sys5_uc")
    end
    bus = first(PSY.get_components(PSY.ACBus, sys))
    source = PSY.Source(;
        name = "test_source",
        available = true,
        bus = bus,
        active_power = 0.0,
        reactive_power = 0.0,
        active_power_limits = (min = -2.0, max = 2.0),
        reactive_power_limits = (min = -2.0, max = 2.0),
        R_th = 0.01,
        X_th = 0.02,
        internal_voltage = 1.0,
        internal_angle = 0.0,
        base_power = 100.0,
    )
    PSY.add_component!(sys, source)

    im_ts = _make_pwl_ts(
        sys, "variable_cost_import",
        [0.0, 100.0, 105.0, 120.0, 200.0], [5.0, 10.0, 20.0, 40.0])
    im_key = PSY.add_time_series!(sys, source, im_ts)
    ex_ts = _make_pwl_ts(
        sys, "variable_cost_export",
        [0.0, 100.0, 105.0, 120.0, 200.0], [12.0, 8.0, 4.0, 1.0])
    ex_key = PSY.add_time_series!(sys, source, ex_ts)

    ts_iec = PSY.ImportExportTimeSeriesCost(;
        import_offer_curves = PSY.make_import_export_ts_curve(im_key),
        export_offer_curves = PSY.make_import_export_ts_curve(ex_key),
    )
    PSY.set_operation_cost!(source, ts_iec)
    return sys, source
end

# ─── tests ────────────────────────────────────────────────────────────────────

@testset "Offer Curve Cost: is_time_variant with new types" begin
    _, comp_mbc = _make_static_mbc_system()
    mbc = PSY.get_operation_cost(comp_mbc)
    _, comp_ts = _make_ts_mbc_system()
    ts_mbc = PSY.get_operation_cost(comp_ts)
    _, source_iec = _make_static_iec_system()
    iec = PSY.get_operation_cost(source_iec)
    _, source_ts = _make_ts_iec_system()
    ts_iec = PSY.get_operation_cost(source_ts)

    # Static → not time variant
    @test !IOM.is_time_variant(PSY.get_incremental_offer_curves(mbc))
    @test !IOM.is_time_variant(PSY.get_import_offer_curves(iec))
    @test !IOM.is_time_variant(PSY.get_shut_down(mbc))
    @test !IOM.is_time_variant(PSY.get_start_up(mbc))

    # TS → time variant
    @test IOM.is_time_variant(PSY.get_incremental_offer_curves(ts_mbc))
    @test IOM.is_time_variant(PSY.get_import_offer_curves(ts_iec))
    @test IOM.is_time_variant(PSY.get_shut_down(ts_mbc))
    @test IOM.is_time_variant(PSY.get_start_up(ts_mbc))
end

@testset "Offer Curve Cost: _shutdown_cost_value" begin
    @test IOM._shutdown_cost_value(42.0) == 42.0
    @test IOM._shutdown_cost_value(PSY.LinearCurve(99.0)) ≈ 99.0
    @test IOM._shutdown_cost_value(PSY.LinearCurve(0.0)) ≈ 0.0
end

@testset "Offer Curve Cost: Detection predicates on devices" begin
    _, comp_mbc = _make_static_mbc_system()
    @test IOM._has_market_bid_cost(comp_mbc)
    @test !IOM._has_import_export_cost(comp_mbc)
    @test IOM._has_offer_curve_cost(comp_mbc)

    _, comp_ts = _make_ts_mbc_system()
    @test IOM._has_market_bid_cost(comp_ts)
    @test !IOM._has_import_export_cost(comp_ts)
    @test IOM._has_offer_curve_cost(comp_ts)

    _, source_iec = _make_static_iec_system()
    @test !IOM._has_market_bid_cost(source_iec)
    @test IOM._has_import_export_cost(source_iec)
    @test IOM._has_offer_curve_cost(source_iec)

    _, source_ts = _make_ts_iec_system()
    @test !IOM._has_market_bid_cost(source_ts)
    @test IOM._has_import_export_cost(source_ts)
    @test IOM._has_offer_curve_cost(source_ts)
end

@testset "Offer Curve Cost: _has_parameter_time_series on devices" begin
    _, comp_static = _make_static_mbc_system()
    _, comp_ts = _make_ts_mbc_system()
    _, source_static = _make_static_iec_system()
    _, source_ts = _make_ts_iec_system()

    param = IOM.IncrementalPiecewiseLinearSlopeParameter
    @test !IOM._has_parameter_time_series(param, comp_static)
    @test IOM._has_parameter_time_series(param, comp_ts)
    @test !IOM._has_parameter_time_series(param, source_static)
    @test IOM._has_parameter_time_series(param, source_ts)

    startup_param = IOM.StartupCostParameter
    @test !IOM._has_parameter_time_series(startup_param, comp_static)
    @test IOM._has_parameter_time_series(startup_param, comp_ts)
end

@testset "Offer Curve Cost: get_initial_input on devices" begin
    _, comp = _make_static_mbc_system(; initial_input = 7.5)
    @test IOM.get_initial_input(IOM.IncrementalOffer(), comp) ≈ 7.5
    @test IOM.get_initial_input(IOM.DecrementalOffer(), comp) ≈ 7.5

    _, comp_ts = _make_ts_mbc_system(; initial_input = 12.0)
    # TS path: initial_input is a TimeSeriesKey, not a Float64
    @test IOM.get_initial_input(IOM.IncrementalOffer(), comp_ts) isa IS.TimeSeriesKey
    @test IOM.get_initial_input(IOM.DecrementalOffer(), comp_ts) isa IS.TimeSeriesKey
end

@testset "Offer Curve Cost: validate_occ_breakpoints_slopes on devices" begin
    # Static MBC: convex incremental curve should validate without error
    _, comp = _make_static_mbc_system(; slopes = [25.0, 30.0, 35.0],
        breakpoints = [0.0, 50.0, 80.0, 100.0])
    IOM.validate_occ_breakpoints_slopes(comp, IOM.IncrementalOffer())

    # Static MBC: concave decremental curve should validate without error
    IOM.validate_occ_breakpoints_slopes(comp, IOM.DecrementalOffer())

    # TS MBC: should return immediately (no validation for TS)
    _, comp_ts = _make_ts_mbc_system()
    IOM.validate_occ_breakpoints_slopes(comp_ts, IOM.IncrementalOffer())
    IOM.validate_occ_breakpoints_slopes(comp_ts, IOM.DecrementalOffer())
end

@testset "Offer Curve Cost: validate_occ_component on devices" begin
    _, comp = _make_static_mbc_system()
    # Startup: static StartUpStages, should warn about multistart but not error
    @test_logs (:warn, r"Multi-start") IOM.validate_occ_component(
        IOM.StartupCostParameter, comp)
    # Shutdown: LinearCurve is valid
    IOM.validate_occ_component(IOM.ShutdownCostParameter, comp)

    # CostAtMin: should not error (simplified to no-op)
    IOM.validate_occ_component(IOM.IncrementalCostAtMinParameter, comp)
    IOM.validate_occ_component(IOM.DecrementalCostAtMinParameter, comp)

    # Breakpoints: validates the static curve
    IOM.validate_occ_component(
        IOM.IncrementalPiecewiseLinearBreakpointParameter, comp)
    IOM.validate_occ_component(
        IOM.DecrementalPiecewiseLinearBreakpointParameter, comp)

    # TS MBC: startup validation should return immediately (skip TS)
    _, comp_ts = _make_ts_mbc_system()
    IOM.validate_occ_component(IOM.StartupCostParameter, comp_ts)
    IOM.validate_occ_component(IOM.ShutdownCostParameter, comp_ts)
end

@testset "Offer Curve Cost: Validation errors (static IEC)" begin
    _, source = _make_static_iec_system()
    iec = PSY.get_operation_cost(source)

    # Valid IEC should not error
    IOM._validate_occ_subtype(iec, IOM.IncrementalOffer(),
        PSY.get_import_offer_curves(iec), "test")

    # IEC with non-zero VOM: error
    bad_import = PSY.CostCurve(
        PSY.PiecewiseIncrementalCurve(
            PSY.PiecewiseStepData([0.0, 100.0], [10.0]), 0.0, nothing),
        PSY.UnitSystem.NATURAL_UNITS,
        PSY.LinearCurve(5.0),
    )
    bad_iec = PSY.ImportExportCost(; import_offer_curves = bad_import,
        export_offer_curves = PSY.make_export_curve([0.0, 100.0], [10.0]))
    @test_throws ArgumentError IOM._validate_occ_subtype(
        bad_iec, IOM.IncrementalOffer(), bad_import, "test")

    # IEC with non-zero first breakpoint: error
    bad_import2 = PSY.CostCurve(
        PSY.PiecewiseIncrementalCurve(
            PSY.PiecewiseStepData([10.0, 100.0], [10.0]), 0.0, nothing))
    bad_iec2 = PSY.ImportExportCost(; import_offer_curves = bad_import2,
        export_offer_curves = PSY.make_export_curve([0.0, 100.0], [10.0]))
    @test_throws ArgumentError IOM._validate_occ_subtype(
        bad_iec2, IOM.IncrementalOffer(), bad_import2, "test")
end

@testset "Offer Curve Cost: TS curve properties (MBC)" begin
    _, comp = _make_ts_mbc_system()
    ts_mbc = PSY.get_operation_cost(comp)
    incr = PSY.get_incremental_offer_curves(ts_mbc)
    decr = PSY.get_decremental_offer_curves(ts_mbc)

    @test IS.is_time_series_backed(incr)
    @test IS.is_time_series_backed(decr)
    @test IOM.is_time_variant(incr)
    @test IOM.is_time_variant(decr)

    vc = PSY.get_value_curve(incr)
    @test IS.get_time_series_key(vc) isa IS.TimeSeriesKey
    @test IS.get_initial_input(vc) isa IS.TimeSeriesKey
end

@testset "Offer Curve Cost: TS curve properties (IEC)" begin
    _, source = _make_ts_iec_system()
    ts_iec = PSY.get_operation_cost(source)
    im = PSY.get_import_offer_curves(ts_iec)
    ex = PSY.get_export_offer_curves(ts_iec)

    @test IS.is_time_series_backed(im)
    @test IS.is_time_series_backed(ex)

    # IEC curves have no initial_input
    @test IS.get_initial_input(PSY.get_value_curve(im)) === nothing
    @test IS.get_initial_input(PSY.get_value_curve(ex)) === nothing
end
