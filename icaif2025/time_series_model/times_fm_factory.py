from abc import ABC, abstractmethod
import timesfm

# Abstract Factory Base
class TimesFMFactory(ABC):
    @abstractmethod
    def create_model(self) -> timesfm.TimesFm:
        pass


# Concrete Factory for TimesFM v1
class TimesFMV1Factory(TimesFMFactory):
    def create_model(self) -> timesfm.TimesFm:
        context_length = 256
        horizon_length = 64

        hparams = timesfm.TimesFmHparams(
            backend="torch",
            context_len=context_length,
            horizon_len=horizon_length,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
        )
        checkpoint = timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        )
        print(f"[TimesFM-V1] Initializing with context_len={hparams.context_len}, horizon_len={hparams.horizon_len}")
        return timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)


# Concrete Factory for TimesFM v2
class TimesFMV2Factory(TimesFMFactory):
    def create_model(self) -> timesfm.TimesFm:
        hparams = timesfm.TimesFmHparams(
            backend="gpu",
            per_core_batch_size=32,
            horizon_len=128,
            num_layers=50,
            use_positional_embedding=False,
            context_len=2048,
        )
        checkpoint = timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
        )
        print(f"[TimesFM-V2] Initializing with context_len={hparams.context_len}, horizon_len={hparams.horizon_len}")
        return timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)


# Client Utility Function
def get_timesfm_model(version: str = "v1") -> timesfm.TimesFm:
    factories = {
        "v1": TimesFMV1Factory(),
        "v2": TimesFMV2Factory(),
    }
    factory = factories.get(version)
    if factory is None:
        raise ValueError(f"Unsupported TimesFM version: {version}")
    return factory.create_model()

