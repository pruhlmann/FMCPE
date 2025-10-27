from simulator.base import (
    Simulator,
    rescale,
    generate_simulation_dataset,
    generate_calibration_dataset,
)
from simulator.pendulum import Pendulum
from simulator.pure_gaussian import PureGaussian
from simulator.light_tunnel import LightTunnel
from simulator.wind_tunnel import WindTunnel


def get_simulator(name: str, **kwargs) -> Simulator:
    """Return the simulator object."""
    if name == "pendulum":
        return Pendulum(**kwargs)
    elif name in ["high_dim_gaussian", "pure_gaussian"]:
        return PureGaussian(**kwargs)
    elif name == "light_tunnel":
        return LightTunnel(**kwargs)
    elif name == "wind_tunnel":
        return WindTunnel(**kwargs)
    else:
        raise ValueError(f"Unknown simulator {name}")


__all__ = [
    "Simulator",
    "get_simulator",
    "rescale",
    "generate_simulation_dataset",
    "generate_calibration_dataset",
]
