from __future__ import annotations

import collections.abc
import logging
from typing import Any, List, Optional, Sequence, Set, Tuple, Union

import torch
from torch.export import ExportedProgram
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import (  # TODO: Should probabably be the TRT EngineCapability Enum
    EngineCapability,
)
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo import CompilationSettings, partitioning
from torch_tensorrt.dynamo._defaults import (
    DEBUG,
    DEVICE,
    ENABLE_EXPERIMENTAL_DECOMPOSITIONS,
    MAX_AUX_STREAMS,
    MIN_BLOCK_SIZE,
    OPTIMIZATION_LEVEL,
    PASS_THROUGH_BUILD_FAILURES,
    PRECISION,
    REQUIRE_FULL_COMPILATION,
    TRUNCATE_LONG_AND_DOUBLE,
    USE_FAST_PARTITIONER,
    USE_PYTHON_RUNTIME,
    VERSION_COMPATIBLE,
    WORKSPACE_SIZE,
)
from torch_tensorrt.dynamo.conversion import (
    convert_module,
    repair_long_or_double_inputs,
)
from torch_tensorrt.dynamo.lowering import apply_lowering_passes
from torch_tensorrt.dynamo.utils import (
    get_torch_inputs,
    prepare_inputs,
    set_log_level,
    to_torch_device,
    to_torch_tensorrt_device,
)

import torch_tensorrt

logger = logging.getLogger(__name__)


def compile(
    exported_program: ExportedProgram,
    inputs: Any,
    *,
    device: Optional[Union[Device, torch.device, str]] = DEVICE,
    disable_tf32: bool = False,
    sparse_weights: bool = False,
    enabled_precisions: Set[torch.dtype] | Tuple[torch.dtype] = (torch.float32,),
    refit: bool = False,
    debug: bool = DEBUG,
    capability: EngineCapability = EngineCapability.default,
    num_avg_timing_iters: int = 1,
    workspace_size: int = WORKSPACE_SIZE,
    dla_sram_size: int = 1048576,
    dla_local_dram_size: int = 1073741824,
    dla_global_dram_size: int = 536870912,
    calibrator: object = None,
    truncate_long_and_double: bool = TRUNCATE_LONG_AND_DOUBLE,
    require_full_compilation: bool = REQUIRE_FULL_COMPILATION,
    min_block_size: int = MIN_BLOCK_SIZE,
    torch_executed_ops: Optional[List[str]] = None,
    torch_executed_modules: Optional[List[str]] = None,
    pass_through_build_failures: bool = PASS_THROUGH_BUILD_FAILURES,
    max_aux_streams: Optional[int] = MAX_AUX_STREAMS,
    version_compatible: bool = VERSION_COMPATIBLE,
    optimization_level: Optional[int] = OPTIMIZATION_LEVEL,
    use_python_runtime: bool = USE_PYTHON_RUNTIME,
    use_fast_partitioner: bool = USE_FAST_PARTITIONER,
    enable_experimental_decompositions: bool = ENABLE_EXPERIMENTAL_DECOMPOSITIONS,
    **kwargs: Any,
) -> torch.fx.GraphModule:
    if debug:
        set_log_level(logger.parent, logging.DEBUG)

    if not isinstance(inputs, collections.abc.Sequence):
        inputs = [inputs]

    # Prepare torch_trt inputs
    inputs = prepare_inputs(inputs)
    device = to_torch_tensorrt_device(device)

    gm = exported_program.module()
    logger.debug("Input graph: " + str(gm.graph))

    # Apply lowering on the graph module
    torch_inputs = get_torch_inputs(inputs, device)
    gm = apply_lowering_passes(gm, torch_inputs)
    logger.debug("Lowered Input graph: " + str(gm.graph))

    enabled_precisions = set(enabled_precisions)

    if (
        torch.float16 in enabled_precisions
        or torch_tensorrt.dtype.half in enabled_precisions
    ):
        precision = torch.float16
    elif (
        torch.float32 in enabled_precisions
        or torch_tensorrt.dtype.float in enabled_precisions
    ):
        precision = torch.float32
    elif len(enabled_precisions) == 0:
        logger.info(f"No precision specified, defaulting to {PRECISION}")
        precision = PRECISION
    else:
        raise ValueError(
            f"Precision {enabled_precisions} not supported in the Dynamo Path"
        )

    compilation_options = {
        "precision": precision,
        "debug": debug,
        "device": device,
        "workspace_size": workspace_size,
        "min_block_size": min_block_size,
        "torch_executed_ops": torch_executed_ops
        if torch_executed_ops is not None
        else [],
        "pass_through_build_failures": pass_through_build_failures,
        "max_aux_streams": max_aux_streams,
        "version_compatible": version_compatible,
        "optimization_level": optimization_level,
        "use_python_runtime": use_python_runtime,
        "truncate_long_and_double": truncate_long_and_double,
        "use_fast_partitioner": use_fast_partitioner,
        "enable_experimental_decompositions": enable_experimental_decompositions,
        "require_full_compilation": require_full_compilation,
    }

    settings = CompilationSettings(**compilation_options)
    logger.info("Compilation Settings: %s\n", settings)
    return compile_module(gm, inputs, settings)


def compile_module(
    gm: torch.fx.GraphModule,
    sample_inputs: Sequence[Input],
    settings: CompilationSettings = CompilationSettings(),
) -> torch.fx.GraphModule:
    """Compile a traced FX module

    Includes: Partitioning + Conversion Phases

    Args:
        module: FX GraphModule to convert
        inputs: Inputs to the module
        settings: Compilation settings
    Returns:
        Compiled FX GraphModule
    """

    # Check the number of supported operations in the graph
    num_supported_ops, total_ops = partitioning.get_graph_converter_support(
        gm, settings.debug, settings.torch_executed_ops
    )

    # If the number of supported operations is 0 or less than the block size, skip the subgraph
    # TODO: Add condition to second expression below when require_full_compilation is added
    if num_supported_ops == 0 or (num_supported_ops < settings.min_block_size):
        logger.warning(
            f"{num_supported_ops} supported operations detected in subgraph containing {total_ops} computational nodes. "
            f"Skipping this subgraph, since min_block_size was detected to be {settings.min_block_size}"
        )
        return gm
    else:
        logger.debug(
            f"Detected support for {num_supported_ops} operators out of {total_ops} in subgraph."
        )

    # Partition module into components that can be TRT-accelerated
    fast_partitioner_failed = False

    logger.info("Beginning TensorRT operator Partitioning Phase")
    # If specified, try using the fast partitioner and fall back to the global one on failure
    if settings.use_fast_partitioner:
        try:
            logger.info("Partitioning the graph via the fast partitioner")
            partitioned_module = partitioning.fast_partition(
                gm,
                verbose=settings.debug,
                min_block_size=settings.min_block_size,
                torch_executed_ops=settings.torch_executed_ops,
            )
        except torch.fx.passes.splitter_base.FxNetSplitterInternalError:
            logger.error(
                "Partitioning failed on the subgraph with fast partition. See trace above. "
                "Retrying with global partition.",
                exc_info=True,
            )

            fast_partitioner_failed = True
            settings.use_fast_partitioner = False

    if not settings.use_fast_partitioner:
        logger.info("Partitioning the graph via the global partitioner")
        partitioned_module = partitioning.global_partition(
            gm,
            verbose=settings.debug,
            min_block_size=settings.min_block_size,
            torch_executed_ops=settings.torch_executed_ops,
        )

    logger.info(
        "Successfully completed graph partitioning phase. "
        "Beginning the conversion phase."
    )

    # Store TRT replicas of Torch subgraphs
    trt_modules = {}
    # Iterate over all components that can be accelerated
    # Generate the corresponding TRT Module for those
    for name, _ in partitioned_module.named_children():
        submodule = getattr(partitioned_module, name)
        # Criteria for a module to be convertible to TRT
        if settings.use_fast_partitioner and "_run_on_acc" not in name:
            continue

        # Get the submodule inputs for min, opt, max shapes of the graph inputs
        submodule_inputs = partitioning.get_submod_inputs(
            partitioned_module,
            submodule,
            sample_inputs,
            to_torch_device(settings.device),
        )

        assert submodule_inputs is not None

        logger.debug(
            "Converting submodule: %s\n Input shapes: %s\n %s",
            str(name),
            [input.shape for input in submodule_inputs],
            str(submodule.graph),
        )

        # Handle long/double inputs if requested by the user
        if settings.truncate_long_and_double:
            submodule_inputs = repair_long_or_double_inputs(
                partitioned_module,
                submodule,
                submodule_inputs,
                to_torch_device(settings.device),
                name,
            )

        # Create TRT engines from submodule
        trt_module = convert_module(
            submodule,
            submodule_inputs,
            settings=settings,
            name=name,
        )

        trt_modules[name] = trt_module

    # Replace all FX Modules with TRT Modules
    for name, trt_module in trt_modules.items():
        setattr(partitioned_module, name, trt_module)

    # Reset settings object to user specification after fallback to global partitioning mode
    if fast_partitioner_failed:
        settings.use_fast_partitioner = True

    return partitioned_module
