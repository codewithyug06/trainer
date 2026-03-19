# KEP-2839: Dynamic LLM Trainer Framework

**Authors:** Yugendhar S (@codewithyug06)
**Status:** Draft
**Created:** 2026-03-19
**Updated:** 2026-03-19
**Tracking Issue:** https://github.com/kubeflow/trainer/issues/2839

---

## Table of Contents

- [Summary](#summary)
- [Motivation](#motivation)
  - [Goals](#goals)
  - [Non-Goals](#non-goals)
- [Proposal](#proposal)
  - [Current Architecture](#current-architecture)
  - [Proposed Architecture](#proposed-architecture)
  - [LLMTrainerBackend Interface](#llmtrainerbackend-interface)
  - [BackendRegistry](#backendregistry)
  - [Python SDK Changes](#python-sdk-changes)
  - [Go Controller Changes](#go-controller-changes)
- [Backend Support Matrix](#backend-support-matrix)
- [Implementation Plan](#implementation-plan)
- [Alternatives Considered](#alternatives-considered)
- [References](#references)

---

## Summary

This KEP proposes a **Dynamic LLM Trainer Framework** that decouples
Kubeflow Trainer from its current hard dependency on TorchTune as the
sole `BuiltinTrainer` backend. The framework introduces a pluggable
backend architecture that enables multiple LLM fine-tuning libraries —
including TRL, Unsloth, and LlamaFactory — to integrate as first-class
`BuiltinTrainer` options while preserving full backward compatibility
with existing TorchTune-based workflows.

---

## Motivation

[KEP-2401](../2401-llm-trainer-v2/README.md) introduced `BuiltinTrainer`
with TorchTune as the initial backend. The KEP explicitly anticipated
future backends:

> _"This approach means that in the future, we can add more frameworks,
> such as Unsloth, as additional BuiltinTrainer options."_

Since KEP-2401 shipped, the LLM fine-tuning ecosystem has evolved
significantly. Three critical gaps now exist in Kubeflow Trainer:

**1. TorchTune is no longer actively adding new features.**
TorchTune has entered maintenance mode. It does not support emerging
post-training methods (DPO, PPO, ORPO) and has limited support for the
latest model architectures (Llama 3.1, Mistral v0.3, Qwen2.5).

**2. No support for RLHF fine-tuning methods.**
Reinforcement Learning from Human Feedback (RLHF) alignment techniques
— Direct Preference Optimization (DPO), Proximal Policy Optimization
(PPO), and Odds Ratio Preference Optimization (ORPO) — are now standard
practice for production LLM development. None of these are available
through the current `BuiltinTrainer`.

**3. No external backend extensibility.**
There is no mechanism for the community or enterprise users to add new
training backends without modifying Kubeflow Trainer's core source code.
This creates ecosystem fragmentation.

### Goals

- Design a `LLMTrainerBackend` interface that all `BuiltinTrainer`
  backends must implement.
- Implement a `BackendRegistry` for dynamic discovery of both in-tree
  and external (pip-installable) backends via Python `entry_points`.
- Refactor the existing TorchTune integration as the first pluggable
  `BuiltinBackend` with zero breaking changes to existing users.
- Add TRL as a new backend, enabling SFT, DPO, PPO, and ORPO fine-tuning.
- Add Unsloth as a new backend for memory-efficient, high-speed fine-tuning.
- Add LlamaFactory as a new backend for broad model architecture coverage.
- Extend the Go controller to support a `spec.trainer.backend` field on
  `TrainJob` for backend selection at the CRD level.

### Non-Goals

- Replacing or deprecating the `CustomTrainer` path.
- Changing the `TrainingRuntime` / `ClusterTrainingRuntime` API.
- Supporting non-PyTorch backends (JAX, TensorFlow) in this KEP.
- Multi-node fine-tuning with backends that do not support it natively.

---

## Proposal

### Current Architecture

The current `BuiltinTrainer` is tightly coupled to TorchTune:
```
train() SDK call
    └── BuiltinTrainer(config=TorchTuneConfig(...))
            └── torch plugin (Go) mutates TrainJob
                    └── TorchTune container image
                            └── torchtune CLI
```

`TorchTuneConfig` is the only accepted configuration type. There is no
interface or registry — the backend is hardcoded.

### Proposed Architecture
```
train() SDK call
    └── BuiltinTrainer(config=TRLConfig(...) | TorchTuneConfig(...) | UnslothConfig(...))
            └── BackendRegistry.resolve(config)
                    └── Selected Backend (TRL | TorchTune | Unsloth | LlamaFactory | External)
                            └── torch plugin (Go) reads spec.trainer.backend
                                    └── Correct container image + CLI
```

### LLMTrainerBackend Interface

All backends implement the following Python Protocol. Using
[structural subtyping (PEP 544)](https://peps.python.org/pep-0544/)
means external backends require no inheritance from Kubeflow classes:
```python
from typing import Protocol, runtime_checkable
from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Backend-agnostic training configuration."""
    model_name_or_path: str
    dataset_name: str
    strategy: str           # "sft" | "dpo" | "ppo" | "orpo"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    learning_rate: float = 2e-4
    output_dir: str = "/workspace/output"


@runtime_checkable
class LLMTrainerBackend(Protocol):
    """Protocol that all BuiltinTrainer backends must satisfy.

    This uses structural subtyping: any class with the required
    attributes and methods is automatically a valid backend,
    regardless of inheritance.
    """

    #: Unique identifier used in BackendRegistry and TrainJob CRD.
    name: str

    #: Fine-tuning strategies supported by this backend.
    supported_strategies: list[str]

    def train(self, config: TrainConfig) -> None:
        """Execute fine-tuning with the given configuration."""
        ...

    def get_supported_models(self) -> list[str]:
        """Return a list of supported base model identifiers."""
        ...

    def validate_config(self, config: TrainConfig) -> None:
        """Raise ValueError if config is invalid for this backend."""
        ...

    def get_runtime_image(self) -> str:
        """Return the container image URI for this backend."""
        ...
```

### BackendRegistry

The `BackendRegistry` discovers backends at runtime using Python
[`importlib.metadata` entry points](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/).
This allows community-maintained backends to be installed via `pip`
without any changes to the Kubeflow Trainer codebase:
```python
# Third-party backend — zero Kubeflow source changes needed.
# In the community package's pyproject.toml:
#
# [project.entry-points."kubeflow.trainer.backends"]
# my_backend = "my_package.trainer:MyBackend"

import importlib.metadata
from typing import Optional


class BackendRegistry:
    """Registry for LLMTrainerBackend implementations.

    Backends are discovered from three sources (in priority order):
    1. In-tree backends shipped with kubeflow-sdk
    2. Externally registered backends via entry_points
    3. Explicitly registered backends via BackendRegistry.register()
    """

    _registry: dict[str, "LLMTrainerBackend"] = {}
    _ENTRY_POINT_GROUP = "kubeflow.trainer.backends"

    @classmethod
    def discover(cls) -> "BackendRegistry":
        """Discover and register all available backends."""
        # Register built-in backends first.
        from kubeflow.trainer.backends.torchtune import TorchTuneBackend
        from kubeflow.trainer.backends.trl import TRLBackend
        from kubeflow.trainer.backends.unsloth import UnslothBackend

        cls.register(TorchTuneBackend())
        cls.register(TRLBackend())
        cls.register(UnslothBackend())

        # Discover external backends via entry_points.
        for ep in importlib.metadata.entry_points(group=cls._ENTRY_POINT_GROUP):
            try:
                backend_cls = ep.load()
                cls.register(backend_cls())
            except Exception as e:
                import warnings
                warnings.warn(
                    f"Failed to load backend '{ep.name}': {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return cls()

    @classmethod
    def register(cls, backend: "LLMTrainerBackend") -> None:
        """Register a backend instance."""
        if not isinstance(backend, LLMTrainerBackend):
            raise TypeError(
                f"Backend {backend!r} does not implement LLMTrainerBackend protocol."
            )
        cls._registry[backend.name] = backend

    @classmethod
    def get(cls, name: str) -> "LLMTrainerBackend":
        """Retrieve a backend by name."""
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(
                f"Backend '{name}' not found. Available backends: {available}"
            )
        return cls._registry[name]

    @classmethod
    def resolve(cls, config: "TrainConfig") -> "LLMTrainerBackend":
        """Auto-select the best backend for a given config.

        Selection priority:
        1. If config has an explicit ``backend`` attribute, use it.
        2. Otherwise pick the first backend that supports the strategy.
        """
        if hasattr(config, "backend") and config.backend:
            return cls.get(config.backend)

        for backend in cls._registry.values():
            if config.strategy in backend.supported_strategies:
                return backend

        raise ValueError(
            f"No backend supports strategy '{config.strategy}'. "
            f"Available backends: {list(cls._registry.keys())}"
        )
```

### Python SDK Changes

The `train()` method in the Kubeflow SDK gains a `TrainerConfig` union
type that includes all supported backend configs:
```python
# kubeflow/trainer/types.py  (additions)

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TRLConfig:
    """Configuration for TRL (HuggingFace) BuiltinTrainer backend.

    Supports SFT, DPO, PPO, and ORPO fine-tuning strategies.
    """
    model_name_or_path: str
    dataset_name: str
    strategy: str = "sft"       # "sft" | "dpo" | "ppo" | "orpo"
    backend: str = "trl"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    learning_rate: float = 2e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    output_dir: str = "/workspace/output"
    resources_per_node: dict = field(default_factory=lambda: {"gpu": 1})


@dataclass
class UnslothConfig:
    """Configuration for Unsloth BuiltinTrainer backend.

    Unsloth provides 2x faster training and 70% less VRAM compared
    to standard SFT/DPO training.
    """
    model_name_or_path: str
    dataset_name: str
    strategy: str = "sft"       # "sft" | "dpo"
    backend: str = "unsloth"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    output_dir: str = "/workspace/output"
    resources_per_node: dict = field(default_factory=lambda: {"gpu": 1})
```

SDK usage remains clean and backward compatible:
```python
from kubeflow.trainer import TrainerClient
from kubeflow.trainer.types import (
    BuiltinTrainer, TorchTuneConfig, TRLConfig, UnslothConfig,
    Initializer, HuggingFaceModelInitializer, HuggingFaceDatasetInitializer,
    Runtime,
)

client = TrainerClient()

# Existing TorchTune workflow — UNCHANGED.
job_name = client.train(
    runtime=Runtime(name="torchtune-llama3.2-1b"),
    initializer=Initializer(
        dataset=HuggingFaceDatasetInitializer(storage_uri="hf://tatsu-lab/alpaca/data"),
        model=HuggingFaceModelInitializer(storage_uri="hf://meta-llama/Llama-3.2-1B-Instruct"),
    ),
    trainer=BuiltinTrainer(config=TorchTuneConfig(resources_per_node={"gpu": 1})),
)

# New TRL DPO workflow — enabled by this KEP.
job_name = client.train(
    runtime=Runtime(name="trl-llama3.1-8b"),
    initializer=Initializer(
        dataset=HuggingFaceDatasetInitializer(storage_uri="hf://trl-lib/ultrafeedback_binarized/data"),
        model=HuggingFaceModelInitializer(storage_uri="hf://meta-llama/Llama-3.1-8B-Instruct"),
    ),
    trainer=BuiltinTrainer(config=TRLConfig(strategy="dpo", lora_rank=16)),
)

# New Unsloth SFT workflow — 2x faster, 70% less VRAM.
job_name = client.train(
    runtime=Runtime(name="unsloth-qwen2.5-7b"),
    initializer=Initializer(
        dataset=HuggingFaceDatasetInitializer(storage_uri="hf://tatsu-lab/alpaca/data"),
        model=HuggingFaceModelInitializer(storage_uri="hf://unsloth/Qwen2.5-7B-Instruct"),
    ),
    trainer=BuiltinTrainer(config=UnslothConfig(load_in_4bit=True)),
)
```

### Go Controller Changes

The `torch` plugin in `pkg/runtime/framework/plugins/torch/` is extended
to read a new optional `spec.trainer.backend` field on `TrainJob`:
```go
// api/trainer/v1alpha1/types.go (additions)

// LLMTrainerBackend identifies which BuiltinTrainer backend to use.
// +kubebuilder:validation:Enum=torchtune;trl;unsloth;llamafactory
type LLMTrainerBackend string

const (
    LLMTrainerBackendTorchTune    LLMTrainerBackend = "torchtune"
    LLMTrainerBackendTRL          LLMTrainerBackend = "trl"
    LLMTrainerBackendUnsloth      LLMTrainerBackend = "unsloth"
    LLMTrainerBackendLlamaFactory LLMTrainerBackend = "llamafactory"
)

// TrainerSpec is extended with an optional Backend field.
type TrainerSpec struct {
    // ... existing fields ...

    // Backend identifies which BuiltinTrainer backend to use.
    // Defaults to "torchtune" for backward compatibility.
    // +optional
    // +kubebuilder:default=torchtune
    Backend *LLMTrainerBackend `json:"backend,omitempty"`
}
```

The `torch` plugin reads `spec.trainer.backend` and sets the correct
container image and entrypoint for the selected backend. If `backend`
is not set, it defaults to `torchtune`, preserving full backward
compatibility.

---

## Backend Support Matrix

| Backend | Strategies | Key Advantage | Container Image |
|---|---|---|---|
| `torchtune` | SFT | Existing — backward compat | `kubeflow/torchtune-trainer:latest` |
| `trl` | SFT, DPO, PPO, ORPO | Industry standard RLHF | `kubeflow/trl-trainer:latest` |
| `unsloth` | SFT, DPO | 2× faster, 70% less VRAM | `kubeflow/unsloth-trainer:latest` |
| `llamafactory` | SFT, DPO, PPO, KTO | 100+ model architectures | `kubeflow/llamafactory-trainer:latest` |
| `<external>` | User-defined | Community extensible via pip | User-provided |

---

## Implementation Plan

### Phase 1 — Core Framework (Community Bonding + Weeks 1–6)

- [ ] Define `LLMTrainerBackend` Protocol in `sdk/`
- [ ] Implement `BackendRegistry` with `entry_points` discovery
- [ ] Refactor TorchTune as first `BuiltinBackend` (no behavior change)
- [ ] Add `spec.trainer.backend` field to `TrainJob` CRD (Go)
- [ ] Unit tests > 85% coverage
- [ ] Design doc review with maintainers

### Phase 2 — New Backends (Weeks 7–10)

- [ ] Implement TRL backend (SFT + DPO + PPO + ORPO)
- [ ] Implement Unsloth backend (SFT + DPO)
- [ ] `ClusterTrainingRuntime` manifests for TRL and Unsloth
- [ ] Integration tests on Kind cluster

### Phase 3 — Community Extensibility (Weeks 11–13)

- [ ] Implement LlamaFactory backend
- [ ] External plugin registration via `entry_points`
- [ ] End-to-end tests (all backends × all strategies)
- [ ] User documentation + Jupyter notebook examples
- [ ] Community blog post

---

## Alternatives Considered

### A. One Config class per backend, no shared interface

**Rejected.** Without a shared `LLMTrainerBackend` interface, the SDK
`train()` method would need explicit `if/elif` chains for each backend.
Adding a new backend requires modifying `train()` directly, breaking
the open/closed principle.

### B. Inherit from an abstract base class

**Considered.** Using `abc.ABC` is a valid alternative. However,
Python's `Protocol` (structural subtyping) was preferred because it
allows community backends to be written as plain Python classes without
importing anything from `kubeflow-sdk`, reducing the dependency surface
for external contributors.

### C. One Kubeflow repo per backend

**Rejected.** Maintaining separate repositories for TRL, Unsloth, etc.
would fragment the community and make it difficult to ensure
compatibility with new Kubeflow Trainer releases.

---

## References

- [KEP-2401: LLM Trainer v2](../2401-llm-trainer-v2/README.md)
- [Tracking Issue #2839](https://github.com/kubeflow/trainer/issues/2839)
- [TRL (HuggingFace)](https://github.com/huggingface/trl)
- [Unsloth](https://github.com/unslothai/unsloth)
- [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory)
- [PEP 544 — Protocols (Structural Subtyping)](https://peps.python.org/pep-0544/)
- [Python Entry Points](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/)