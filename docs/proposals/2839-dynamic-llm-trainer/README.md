\# KEP-2839: Dynamic LLM Trainer Framework



\*\*Authors:\*\* Yugendhar S (@codewithyug06)

\*\*Status:\*\* Draft

\*\*Created:\*\* 2026-03-19

\*\*Updated:\*\* 2026-03-19

\*\*Tracking Issue:\*\* https://github.com/kubeflow/trainer/issues/2839



\---



\## Table of Contents



\- \[Summary](#summary)

\- \[Motivation](#motivation)

&#x20; - \[Goals](#goals)

&#x20; - \[Non-Goals](#non-goals)

\- \[Proposal](#proposal)

&#x20; - \[Current Architecture](#current-architecture)

&#x20; - \[Proposed Architecture](#proposed-architecture)

&#x20; - \[LLMTrainerBackend Interface](#llmtrainerbackend-interface)

&#x20; - \[BackendRegistry](#backendregistry)

&#x20; - \[Python SDK Changes](#python-sdk-changes)

&#x20; - \[Go Controller Changes](#go-controller-changes)

\- \[Backend Support Matrix](#backend-support-matrix)

\- \[Implementation Plan](#implementation-plan)

\- \[Alternatives Considered](#alternatives-considered)

\- \[References](#references)



\---



\## Summary



This KEP proposes a \*\*Dynamic LLM Trainer Framework\*\* that decouples

Kubeflow Trainer from its current hard dependency on TorchTune as the

sole `BuiltinTrainer` backend. The framework introduces a pluggable

backend architecture that enables multiple LLM fine-tuning libraries —

including TRL, Unsloth, and LlamaFactory — to integrate as first-class

`BuiltinTrainer` options while preserving full backward compatibility

with existing TorchTune-based workflows.



\---



\## Motivation



\[KEP-2401](../2401-llm-trainer-v2/README.md) introduced `BuiltinTrainer`

with TorchTune as the initial backend. The KEP explicitly anticipated

future backends:



> \_"This approach means that in the future, we can add more frameworks,

> such as Unsloth, as additional BuiltinTrainer options."\_



Since KEP-2401 shipped, the LLM fine-tuning ecosystem has evolved

significantly. Three critical gaps now exist in Kubeflow Trainer:



\*\*1. TorchTune is no longer actively adding new features.\*\*

TorchTune has entered maintenance mode. It does not support emerging

post-training methods (DPO, PPO, ORPO) and has limited support for the

latest model architectures (Llama 3.1, Mistral v0.3, Qwen2.5).



\*\*2. No support for RLHF fine-tuning methods.\*\*

Reinforcement Learning from Human Feedback (RLHF) alignment techniques

— Direct Preference Optimization (DPO), Proximal Policy Optimization

(PPO), and Odds Ratio Preference Optimization (ORPO) — are now standard

practice for production LLM development. None of these are available

through the current `BuiltinTrainer`.



\*\*3. No external backend extensibility.\*\*

There is no mechanism for the community or enterprise users to add new

training backends without modifying Kubeflow Trainer's core source code.

This creates ecosystem fragmentation.



\### Goals



\- Design a `LLMTrainerBackend` interface that all `BuiltinTrainer`

&#x20; backends must implement.

\- Implement a `BackendRegistry` for dynamic discovery of both in-tree

&#x20; and external (pip-installable) backends via Python `entry\_points`.

\- Refactor the existing TorchTune integration as the first pluggable

&#x20; `BuiltinBackend` with zero breaking changes to existing users.

\- Add TRL as a new backend, enabling SFT, DPO, PPO, and ORPO fine-tuning.

\- Add Unsloth as a new backend for memory-efficient, high-speed fine-tuning.

\- Add LlamaFactory as a new backend for broad model architecture coverage.

\- Extend the Go controller to support a `spec.trainer.backend` field on

&#x20; `TrainJob` for backend selection at the CRD level.



\### Non-Goals



\- Replacing or deprecating the `CustomTrainer` path.

\- Changing the `TrainingRuntime` / `ClusterTrainingRuntime` API.

\- Supporting non-PyTorch backends (JAX, TensorFlow) in this KEP.

\- Multi-node fine-tuning with backends that do not support it natively.



\---



\## Proposal



\### Current Architecture



The current `BuiltinTrainer` is tightly coupled to TorchTune:

```

train() SDK call

&#x20;   └── BuiltinTrainer(config=TorchTuneConfig(...))

&#x20;           └── torch plugin (Go) mutates TrainJob

&#x20;                   └── TorchTune container image

&#x20;                           └── torchtune CLI

```



`TorchTuneConfig` is the only accepted configuration type. There is no

interface or registry — the backend is hardcoded.



\### Proposed Architecture

```

train() SDK call

&#x20;   └── BuiltinTrainer(config=TRLConfig(...) | TorchTuneConfig(...) | UnslothConfig(...))

&#x20;           └── BackendRegistry.resolve(config)

&#x20;                   └── Selected Backend (TRL | TorchTune | Unsloth | LlamaFactory | External)

&#x20;                           └── torch plugin (Go) reads spec.trainer.backend

&#x20;                                   └── Correct container image + CLI

```



\### LLMTrainerBackend Interface



All backends implement the following Python Protocol. Using

\[structural subtyping (PEP 544)](https://peps.python.org/pep-0544/)

means external backends require no inheritance from Kubeflow classes:

```python

from typing import Protocol, runtime\_checkable

from dataclasses import dataclass





@dataclass

class TrainConfig:

&#x20;   """Backend-agnostic training configuration."""

&#x20;   model\_name\_or\_path: str

&#x20;   dataset\_name: str

&#x20;   strategy: str           # "sft" | "dpo" | "ppo" | "orpo"

&#x20;   num\_train\_epochs: int = 3

&#x20;   per\_device\_train\_batch\_size: int = 4

&#x20;   learning\_rate: float = 2e-4

&#x20;   output\_dir: str = "/workspace/output"





@runtime\_checkable

class LLMTrainerBackend(Protocol):

&#x20;   """Protocol that all BuiltinTrainer backends must satisfy.



&#x20;   This uses structural subtyping: any class with the required

&#x20;   attributes and methods is automatically a valid backend,

&#x20;   regardless of inheritance.

&#x20;   """



&#x20;   #: Unique identifier used in BackendRegistry and TrainJob CRD.

&#x20;   name: str



&#x20;   #: Fine-tuning strategies supported by this backend.

&#x20;   supported\_strategies: list\[str]



&#x20;   def train(self, config: TrainConfig) -> None:

&#x20;       """Execute fine-tuning with the given configuration."""

&#x20;       ...



&#x20;   def get\_supported\_models(self) -> list\[str]:

&#x20;       """Return a list of supported base model identifiers."""

&#x20;       ...



&#x20;   def validate\_config(self, config: TrainConfig) -> None:

&#x20;       """Raise ValueError if config is invalid for this backend."""

&#x20;       ...



&#x20;   def get\_runtime\_image(self) -> str:

&#x20;       """Return the container image URI for this backend."""

&#x20;       ...

```



\### BackendRegistry



The `BackendRegistry` discovers backends at runtime using Python

\[`importlib.metadata` entry points](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/).

This allows community-maintained backends to be installed via `pip`

without any changes to the Kubeflow Trainer codebase:

```python

\# Third-party backend — zero Kubeflow source changes needed.

\# In the community package's pyproject.toml:

\#

\# \[project.entry-points."kubeflow.trainer.backends"]

\# my\_backend = "my\_package.trainer:MyBackend"



import importlib.metadata

from typing import Optional





class BackendRegistry:

&#x20;   """Registry for LLMTrainerBackend implementations.



&#x20;   Backends are discovered from three sources (in priority order):

&#x20;   1. In-tree backends shipped with kubeflow-sdk

&#x20;   2. Externally registered backends via entry\_points

&#x20;   3. Explicitly registered backends via BackendRegistry.register()

&#x20;   """



&#x20;   \_registry: dict\[str, "LLMTrainerBackend"] = {}

&#x20;   \_ENTRY\_POINT\_GROUP = "kubeflow.trainer.backends"



&#x20;   @classmethod

&#x20;   def discover(cls) -> "BackendRegistry":

&#x20;       """Discover and register all available backends."""

&#x20;       # Register built-in backends first.

&#x20;       from kubeflow.trainer.backends.torchtune import TorchTuneBackend

&#x20;       from kubeflow.trainer.backends.trl import TRLBackend

&#x20;       from kubeflow.trainer.backends.unsloth import UnslothBackend



&#x20;       cls.register(TorchTuneBackend())

&#x20;       cls.register(TRLBackend())

&#x20;       cls.register(UnslothBackend())



&#x20;       # Discover external backends via entry\_points.

&#x20;       for ep in importlib.metadata.entry\_points(group=cls.\_ENTRY\_POINT\_GROUP):

&#x20;           try:

&#x20;               backend\_cls = ep.load()

&#x20;               cls.register(backend\_cls())

&#x20;           except Exception as e:

&#x20;               import warnings

&#x20;               warnings.warn(

&#x20;                   f"Failed to load backend '{ep.name}': {e}",

&#x20;                   RuntimeWarning,

&#x20;                   stacklevel=2,

&#x20;               )



&#x20;       return cls()



&#x20;   @classmethod

&#x20;   def register(cls, backend: "LLMTrainerBackend") -> None:

&#x20;       """Register a backend instance."""

&#x20;       if not isinstance(backend, LLMTrainerBackend):

&#x20;           raise TypeError(

&#x20;               f"Backend {backend!r} does not implement LLMTrainerBackend protocol."

&#x20;           )

&#x20;       cls.\_registry\[backend.name] = backend



&#x20;   @classmethod

&#x20;   def get(cls, name: str) -> "LLMTrainerBackend":

&#x20;       """Retrieve a backend by name."""

&#x20;       if name not in cls.\_registry:

&#x20;           available = list(cls.\_registry.keys())

&#x20;           raise KeyError(

&#x20;               f"Backend '{name}' not found. Available backends: {available}"

&#x20;           )

&#x20;       return cls.\_registry\[name]



&#x20;   @classmethod

&#x20;   def resolve(cls, config: "TrainConfig") -> "LLMTrainerBackend":

&#x20;       """Auto-select the best backend for a given config.



&#x20;       Selection priority:

&#x20;       1. If config has an explicit ``backend`` attribute, use it.

&#x20;       2. Otherwise pick the first backend that supports the strategy.

&#x20;       """

&#x20;       if hasattr(config, "backend") and config.backend:

&#x20;           return cls.get(config.backend)



&#x20;       for backend in cls.\_registry.values():

&#x20;           if config.strategy in backend.supported\_strategies:

&#x20;               return backend



&#x20;       raise ValueError(

&#x20;           f"No backend supports strategy '{config.strategy}'. "

&#x20;           f"Available backends: {list(cls.\_registry.keys())}"

&#x20;       )

```



\### Python SDK Changes



The `train()` method in the Kubeflow SDK gains a `TrainerConfig` union

type that includes all supported backend configs:

```python

\# kubeflow/trainer/types.py  (additions)



from dataclasses import dataclass, field

from typing import Optional





@dataclass

class TRLConfig:

&#x20;   """Configuration for TRL (HuggingFace) BuiltinTrainer backend.



&#x20;   Supports SFT, DPO, PPO, and ORPO fine-tuning strategies.

&#x20;   """

&#x20;   model\_name\_or\_path: str

&#x20;   dataset\_name: str

&#x20;   strategy: str = "sft"       # "sft" | "dpo" | "ppo" | "orpo"

&#x20;   backend: str = "trl"

&#x20;   num\_train\_epochs: int = 3

&#x20;   per\_device\_train\_batch\_size: int = 4

&#x20;   learning\_rate: float = 2e-4

&#x20;   lora\_rank: int = 16

&#x20;   lora\_alpha: int = 32

&#x20;   lora\_dropout: float = 0.05

&#x20;   output\_dir: str = "/workspace/output"

&#x20;   resources\_per\_node: dict = field(default\_factory=lambda: {"gpu": 1})





@dataclass

class UnslothConfig:

&#x20;   """Configuration for Unsloth BuiltinTrainer backend.



&#x20;   Unsloth provides 2x faster training and 70% less VRAM compared

&#x20;   to standard SFT/DPO training.

&#x20;   """

&#x20;   model\_name\_or\_path: str

&#x20;   dataset\_name: str

&#x20;   strategy: str = "sft"       # "sft" | "dpo"

&#x20;   backend: str = "unsloth"

&#x20;   num\_train\_epochs: int = 3

&#x20;   per\_device\_train\_batch\_size: int = 4

&#x20;   learning\_rate: float = 2e-4

&#x20;   max\_seq\_length: int = 2048

&#x20;   load\_in\_4bit: bool = True

&#x20;   output\_dir: str = "/workspace/output"

&#x20;   resources\_per\_node: dict = field(default\_factory=lambda: {"gpu": 1})

```



SDK usage remains clean and backward compatible:

```python

from kubeflow.trainer import TrainerClient

from kubeflow.trainer.types import (

&#x20;   BuiltinTrainer, TorchTuneConfig, TRLConfig, UnslothConfig,

&#x20;   Initializer, HuggingFaceModelInitializer, HuggingFaceDatasetInitializer,

&#x20;   Runtime,

)



client = TrainerClient()



\# Existing TorchTune workflow — UNCHANGED.

job\_name = client.train(

&#x20;   runtime=Runtime(name="torchtune-llama3.2-1b"),

&#x20;   initializer=Initializer(

&#x20;       dataset=HuggingFaceDatasetInitializer(storage\_uri="hf://tatsu-lab/alpaca/data"),

&#x20;       model=HuggingFaceModelInitializer(storage\_uri="hf://meta-llama/Llama-3.2-1B-Instruct"),

&#x20;   ),

&#x20;   trainer=BuiltinTrainer(config=TorchTuneConfig(resources\_per\_node={"gpu": 1})),

)



\# New TRL DPO workflow — enabled by this KEP.

job\_name = client.train(

&#x20;   runtime=Runtime(name="trl-llama3.1-8b"),

&#x20;   initializer=Initializer(

&#x20;       dataset=HuggingFaceDatasetInitializer(storage\_uri="hf://trl-lib/ultrafeedback\_binarized/data"),

&#x20;       model=HuggingFaceModelInitializer(storage\_uri="hf://meta-llama/Llama-3.1-8B-Instruct"),

&#x20;   ),

&#x20;   trainer=BuiltinTrainer(config=TRLConfig(strategy="dpo", lora\_rank=16)),

)



\# New Unsloth SFT workflow — 2x faster, 70% less VRAM.

job\_name = client.train(

&#x20;   runtime=Runtime(name="unsloth-qwen2.5-7b"),

&#x20;   initializer=Initializer(

&#x20;       dataset=HuggingFaceDatasetInitializer(storage\_uri="hf://tatsu-lab/alpaca/data"),

&#x20;       model=HuggingFaceModelInitializer(storage\_uri="hf://unsloth/Qwen2.5-7B-Instruct"),

&#x20;   ),

&#x20;   trainer=BuiltinTrainer(config=UnslothConfig(load\_in\_4bit=True)),

)

```



\### Go Controller Changes



The `torch` plugin in `pkg/runtime/framework/plugins/torch/` is extended

to read a new optional `spec.trainer.backend` field on `TrainJob`:

```go

// api/trainer/v1alpha1/types.go (additions)



// LLMTrainerBackend identifies which BuiltinTrainer backend to use.

// +kubebuilder:validation:Enum=torchtune;trl;unsloth;llamafactory

type LLMTrainerBackend string



const (

&#x20;   LLMTrainerBackendTorchTune    LLMTrainerBackend = "torchtune"

&#x20;   LLMTrainerBackendTRL          LLMTrainerBackend = "trl"

&#x20;   LLMTrainerBackendUnsloth      LLMTrainerBackend = "unsloth"

&#x20;   LLMTrainerBackendLlamaFactory LLMTrainerBackend = "llamafactory"

)



// TrainerSpec is extended with an optional Backend field.

type TrainerSpec struct {

&#x20;   // ... existing fields ...



&#x20;   // Backend identifies which BuiltinTrainer backend to use.

&#x20;   // Defaults to "torchtune" for backward compatibility.

&#x20;   // +optional

&#x20;   // +kubebuilder:default=torchtune

&#x20;   Backend \*LLMTrainerBackend `json:"backend,omitempty"`

}

```



The `torch` plugin reads `spec.trainer.backend` and sets the correct

container image and entrypoint for the selected backend. If `backend`

is not set, it defaults to `torchtune`, preserving full backward

compatibility.



\---



\## Backend Support Matrix



| Backend | Strategies | Key Advantage | Container Image |

|---|---|---|---|

| `torchtune` | SFT | Existing — backward compat | `kubeflow/torchtune-trainer:latest` |

| `trl` | SFT, DPO, PPO, ORPO | Industry standard RLHF | `kubeflow/trl-trainer:latest` |

| `unsloth` | SFT, DPO | 2× faster, 70% less VRAM | `kubeflow/unsloth-trainer:latest` |

| `llamafactory` | SFT, DPO, PPO, KTO | 100+ model architectures | `kubeflow/llamafactory-trainer:latest` |

| `<external>` | User-defined | Community extensible via pip | User-provided |



\---



\## Implementation Plan



\### Phase 1 — Core Framework (Community Bonding + Weeks 1–6)



\- \[ ] Define `LLMTrainerBackend` Protocol in `sdk/`

\- \[ ] Implement `BackendRegistry` with `entry\_points` discovery

\- \[ ] Refactor TorchTune as first `BuiltinBackend` (no behavior change)

\- \[ ] Add `spec.trainer.backend` field to `TrainJob` CRD (Go)

\- \[ ] Unit tests > 85% coverage

\- \[ ] Design doc review with maintainers



\### Phase 2 — New Backends (Weeks 7–10)



\- \[ ] Implement TRL backend (SFT + DPO + PPO + ORPO)

\- \[ ] Implement Unsloth backend (SFT + DPO)

\- \[ ] `ClusterTrainingRuntime` manifests for TRL and Unsloth

\- \[ ] Integration tests on Kind cluster



\### Phase 3 — Community Extensibility (Weeks 11–13)



\- \[ ] Implement LlamaFactory backend

\- \[ ] External plugin registration via `entry\_points`

\- \[ ] End-to-end tests (all backends × all strategies)

\- \[ ] User documentation + Jupyter notebook examples

\- \[ ] Community blog post



\---



\## Alternatives Considered



\### A. One Config class per backend, no shared interface



\*\*Rejected.\*\* Without a shared `LLMTrainerBackend` interface, the SDK

`train()` method would need explicit `if/elif` chains for each backend.

Adding a new backend requires modifying `train()` directly, breaking

the open/closed principle.



\### B. Inherit from an abstract base class



\*\*Considered.\*\* Using `abc.ABC` is a valid alternative. However,

Python's `Protocol` (structural subtyping) was preferred because it

allows community backends to be written as plain Python classes without

importing anything from `kubeflow-sdk`, reducing the dependency surface

for external contributors.



\### C. One Kubeflow repo per backend



\*\*Rejected.\*\* Maintaining separate repositories for TRL, Unsloth, etc.

would fragment the community and make it difficult to ensure

compatibility with new Kubeflow Trainer releases.



\---



\## References



\- \[KEP-2401: LLM Trainer v2](../2401-llm-trainer-v2/README.md)

\- \[Tracking Issue #2839](https://github.com/kubeflow/trainer/issues/2839)

\- \[TRL (HuggingFace)](https://github.com/huggingface/trl)

\- \[Unsloth](https://github.com/unslothai/unsloth)

\- \[LlamaFactory](https://github.com/hiyouga/LLaMA-Factory)

\- \[PEP 544 — Protocols (Structural Subtyping)](https://peps.python.org/pep-0544/)

\- \[Python Entry Points](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/)

