from __future__ import annotations

import pytest

import arti
from arti import mechanisms


@pytest.mark.parametrize(
    "atom_type",
    [
        mechanisms.NeuralPlasticityAtom,
        mechanisms.NeuralPlasticityBlendAtom,
        mechanisms.NeuralPlasticityOuterAtom,
        mechanisms.NeuralPlasticityOuterAtomV2,
        mechanisms.NeuralPlasticityTransportAtom,
        mechanisms.NeuralPlasticityPolynomialAtom,
        mechanisms.NeuralPlasticityProximalAtom,
    ],
)
def test_effect_metadata_describes_predecessor_binding(atom_type) -> None:
    provenance = arti.component_provenance(atom_type())
    config = provenance["components"][0]["config"]
    assert config["target_binding"] == "runtime-predecessor-bank"
    assert config["state_access"] == "effect-operands-only"
    assert arti.validate_component_provenance(provenance) == provenance
