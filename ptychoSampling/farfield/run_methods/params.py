import dataclasses as dt
from ptychoSampling.farfield.run_methods.recons_methods import *
from ptychoSampling.reconstruction.recons import ReconstructionT


@dt.dataclass
class ReconsInitParams:
    n_validation: int = 0
    registration_log_frequency: int = 1
    r_factor_log: bool = True
    loss_init_extra_kwargs: dict = dt.field(default_factory=dict)
    update_delay_probe: int = None

@dt.dataclass
class ReconsRunParams:
    debug_output: bool = True
    debug_output_epoch_frequency: int = 50
    validation_epoch_frequency: int = 1

@dt.dataclass
class MetaParams:
    recons_method: ReconstructionT = None
    fname_prefix: str = ""
    n_recons: int = 3
    iterables: dict = dt.field(default_factory=lambda: {"reconstruct_probe": [True, False],
                                                        "training_batch_size":[0, 256]})
    recons_init_params: ReconsInitParams = dt.field(default_factory=ReconsInitParams)
    recons_run_params: ReconsRunParams = dt.field(default_factory=ReconsRunParams)
    max_epochs: int = 100
    update_delay_probe_epochs: int = None
    loss_n_spline_epochs: int = None




@dt.dataclass
class CBReconsRunParams(ReconsRunParams):
    debug_output: bool = True
    debug_output_epoch_frequency: int = 50
    validation_epoch_frequency: int = 1

@dt.dataclass
class CurveballMetaParams(MetaParams):
    recons_method: ReconstructionT = CurveballReconstructionT
    fname_prefix: str = "cb"
    n_recons: int = 1
    iterables: dict = dt.field(default_factory=lambda: {"reconstruct_probe": [True, False],
                                                        "training_batch_size": [0, 256]})
    recons_run_params: ReconsRunParams = dt.field(default_factory=CBReconsRunParams)

@dt.dataclass
class LMAMetaParams(MetaParams):
    recons_method: ReconstructionT = LMAReconstructionT
    fname_prefix: str = "lma"
    n_recons: int = 1
    iterables: dict = dt.field(default_factory=lambda: {"reconstruct_probe": [True, False],
                                                        "training_batch_size": [0]})
    recons_run_params: ReconsRunParams = dt.field(default_factory=ReconsRunParams)