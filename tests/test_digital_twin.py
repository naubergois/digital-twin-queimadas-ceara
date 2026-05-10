"""Testes do autômato celular clássico (`FireDigitalTwin`)."""

import numpy as np
import pandas as pd
import pytest

from src.digital_twin import FireDigitalTwin


@pytest.fixture
def coarse_twin() -> FireDigitalTwin:
    # Grade pequena para testes rápidos (~3×3 células no CE)
    return FireDigitalTwin(resolution=2.0)


def test_initialize_from_history_builds_risk(coarse_twin: FireDigitalTwin):
    df = pd.DataFrame(
        {
            "lat": [-5.2, -5.3, -5.25],
            "lon": [-39.1, -39.2, -39.15],
        }
    )
    coarse_twin.initialize_from_history(df)
    assert coarse_twin.history_grid is not None
    assert coarse_twin.risk_grid is not None
    assert coarse_twin.fire_grid.shape == coarse_twin.history_grid.shape


def test_add_active_fires_marks_grid(coarse_twin: FireDigitalTwin):
    coarse_twin.initialize_from_history(pd.DataFrame({"lat": [-5.0], "lon": [-39.0]}))
    coarse_twin.add_active_fires(pd.DataFrame({"lat": [-5.1], "lon": [-39.1]}))
    assert coarse_twin.fire_grid is not None
    assert int(coarse_twin.fire_grid.sum()) >= 1


def test_step_requires_initialization(coarse_twin: FireDigitalTwin):
    assert coarse_twin.step().get("error")


def test_simulate_changes_state_deterministic_with_seed(coarse_twin: FireDigitalTwin):
    coarse_twin.initialize_from_history(pd.DataFrame({"lat": [-5.0], "lon": [-39.0]}))
    coarse_twin.add_active_fires(pd.DataFrame({"lat": [-5.0], "lon": [-39.0]}))
    np.random.seed(123)
    h1 = coarse_twin.simulate(steps=3)
    coarse_twin.initialize_from_history(pd.DataFrame({"lat": [-5.0], "lon": [-39.0]}))
    coarse_twin.add_active_fires(pd.DataFrame({"lat": [-5.0], "lon": [-39.0]}))
    np.random.seed(123)
    h2 = coarse_twin.simulate(steps=3)
    assert [x["total_affected"] for x in h1] == [x["total_affected"] for x in h2]
