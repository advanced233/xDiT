"""Unit tests for narrowing declared INT8 targets with --int8_gemm_modules.

Wan 2.2 runs two experts, and quantizing only one of them is how its per-expert INT8
sensitivity is measured, so the selection has to reach every INT8 consumer through the
declared target list.

Run with:
    python -m pytest tests/unit/model_executor/loading/test_int8_plan.py -v
"""

import copy
from types import SimpleNamespace

import pytest

from xfuser.model_executor.models.runner_models.loading.quantization_plan import (
    QuantizationPlan,
    apply_int8_module_selection,
)

WAN22_EXPERTS = ["transformer.blocks", "transformer_2.blocks"]


def _settings(targets):
    return SimpleNamespace(int8_gemm_module_list=list(targets))


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (None, WAN22_EXPERTS),
        ("transformer.blocks", ["transformer.blocks"]),
        ("transformer_2.blocks", ["transformer_2.blocks"]),
        # Declared order wins over request order, and repeats collapse.
        (" transformer_2.blocks , transformer.blocks,transformer_2.blocks", WAN22_EXPERTS),
    ],
)
def test_selection_keeps_the_named_declared_targets(requested, expected):
    settings = _settings(WAN22_EXPERTS)

    apply_int8_module_selection(SimpleNamespace(int8_gemm_modules=requested), settings)

    assert settings.int8_gemm_module_list == expected


def test_unknown_target_names_the_declared_ones():
    settings = _settings(WAN22_EXPERTS)

    with pytest.raises(ValueError, match=r"declared INT8 targets.*unknown: \['transformer_3.blocks'\]"):
        apply_int8_module_selection(
            SimpleNamespace(int8_gemm_modules="transformer.blocks,transformer_3.blocks"), settings
        )
    assert settings.int8_gemm_module_list == WAN22_EXPERTS


def test_a_list_of_only_separators_is_rejected():
    with pytest.raises(ValueError, match="at least one module"):
        apply_int8_module_selection(SimpleNamespace(int8_gemm_modules=" , "), _settings(WAN22_EXPERTS))


@pytest.fixture(scope="module")
def wan_runners():
    pytest.importorskip("torch", reason="PyTorch is required to import the Wan runners")
    pytest.importorskip("diffusers", reason="diffusers is required to import the Wan runners")
    from xfuser.config.args import xFuserArgs
    from xfuser.model_executor.models.runner_models import wan

    return SimpleNamespace(args_cls=xFuserArgs, wan=wan)


def _runner(cls, config):
    """A runner with its settings customized, without loading weights or a device."""
    model = object.__new__(cls)
    model.settings = copy.deepcopy(cls.settings)
    model._customize_settings(config)
    model.config = config
    return model


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (None, WAN22_EXPERTS),
        ("transformer.blocks", ["transformer.blocks"]),
        ("transformer_2.blocks", ["transformer_2.blocks"]),
    ],
)
def test_wan22_t2v_int8_targets_follow_the_selection(wan_runners, requested, expected):
    config = wan_runners.args_cls(model="Wan2.2-T2V", gemm_quantization="int8", int8_gemm_modules=requested)
    model = _runner(wan_runners.wan.xFuserWan22T2VModel, config)

    model._validate_config(config)
    model._update_model_settings(config)

    assert QuantizationPlan(model).module_list("int8") == expected


def test_wan21_t2v_rejects_the_second_expert_it_does_not_have(wan_runners):
    config = wan_runners.args_cls(
        model="Wan2.1-T2V-1.3B", gemm_quantization="int8", int8_gemm_modules="transformer_2.blocks"
    )
    model = _runner(wan_runners.wan.xFuserWan21T2V1_3BModel, config)

    model._validate_config(config)
    with pytest.raises(ValueError, match="unknown: \\['transformer_2.blocks'\\]"):
        model._update_model_settings(config)
