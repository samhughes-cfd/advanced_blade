# API standards (code style and structure)

This document describes **API design and code structure conventions** for an agentic orchestration codebase. These conventions define consistent patterns for orchestrators, processing classes, utility classes, input dataclasses, result containers, and visualisation entry points.

The goal is to ensure:

- consistent APIs across modules
- strong separation of concerns
- low-verbosity top-level orchestration scripts
- clear execution and result-access rules
- easy extension of repeating orchestration units

---

## Table of Contents

1. [Class Types Overview](#class-types-overview)
2. [Orchestrator/Facade Classes](#orchestratorfacade-classes)
3. [Processing Classes](#processing-classes)
4. [Utility Classes](#utility-classes)
5. [Input Classes](#input-classes)
6. [Container Classes](#container-classes)
7. [Visualisation Convention](#visualisation-convention)
8. [Method Naming Conventions](#method-naming-conventions)
9. [Type Hints and Forward References](#type-hints-and-forward-references)
10. [Error Handling Patterns](#error-handling-patterns)
11. [Orchestration Chain Pattern](#orchestration-chain-pattern)
12. [Known Inconsistencies](#known-inconsistencies)
13. [Summary](#summary)
14. [Examples](#examples)

---

## Class Types Overview

The codebase organizes classes into six distinct categories, each with a specific responsibility and API pattern.

| Class Type | Purpose | API Pattern | Location |
|------------|---------|-------------|----------|
| **Orchestrator/Facade** | High-level coordination, multi-step workflows | Constructor → `execute()` → `get_*()` | `module/*/module.py` |
| **Processing** | Single-purpose computation units | Constructor → `evaluate()` | `module/*/module.py` |
| **Utility** | Reusable computational functions | Constructor → methods (stateless/functional) | `module/utilities/*/module.py` |
| **Input** | External configuration/parameters | Dataclass + optional `load()` | `<project>_inputs/*.py` or `internal_inputs/*.py` |
| **Container** | Data structures populated by routines/modules | Dataclass + direct field access | `module/*/container.py` |
| **Visualisation** | Rendering logic for a result container | Container → `.visualise()` → paired `Vis` class | `module/*/vis.py` |

---

## Orchestrator/Facade Classes

Orchestrator classes coordinate multiple processing steps and provide a high-level public interface for a module workflow.

### Design Pattern

```python
class OrchestratorClass:
    def __init__(self, *, dependency1, dependency2, ...):
        """Store dependencies, validate inputs. No computation."""
        self._dependency1 = dependency1
        self._dependency2 = dependency2
        self._executed = False
        self._results = None

    def execute(self) -> "OrchestratorClass":
        """Perform all computation. Returns self for chaining."""
        if self._executed:
            return self  # Idempotent

        self._results = ...
        self._executed = True
        return self

    def get_results(self) -> "ResultType":
        """Retrieve results. Validates execution."""
        if not self._executed:
            raise RuntimeError("execute() must be called first")
        return self._results

    @property
    def results(self) -> "ResultType":
        """Property access (optional, for convenience)."""
        if not self._executed:
            raise RuntimeError("execute() must be called first")
        return self._results
```

The `*` in `__init__(self, *, ...)` makes all constructor arguments keyword-only.

### Key Characteristics

1. **Constructor (`__init__`)**
   - Store dependencies only
   - Validate required inputs
   - Perform no heavy computation
   - Initialize `_executed = False`
   - Initialize result containers to `None` or empty structures
   - Use keyword-only arguments for all public orchestrators

2. **Execution Method (`execute()`)**
   - Public facade method is always named `execute()`
   - Returns `self` for method chaining
   - Must be idempotent
   - Performs all heavy computation
   - Sets `_executed = True` after successful completion

3. **Result Retrieval**
   - Prefer descriptive getters such as `get_selected_items()` or `get_summary()`
   - `get_results()` is acceptable for generic modules
   - Optional `@property` access may be provided for convenience
   - All access paths must validate that `execute()` has already been called

### Usage Pattern

```python
orchestrator = OrchestratorClass(
    dependency1=dep1,
    dependency2=dep2,
)

orchestrator.execute()

results = orchestrator.get_results()   # preferred
results = orchestrator.results         # convenience
```

### Best Practices

- ✅ Use keyword-only arguments (`*`) in public constructors
- ✅ Validate dependencies in `__init__`
- ✅ Store dependencies with `_` prefix
- ✅ Use `_executed` for idempotency and access guards
- ✅ Return `self` from `execute()`
- ❌ Do not compute in `__init__`
- ❌ Do not access results before execution

---

## Processing Classes

Processing classes perform focused computations such as screening, evaluation, transformation, filtering, or scoring. They are typically used internally by orchestrators.

### Design Pattern

```python
class ProcessingClass:
    def __init__(self, config=None):
        """Store configuration. No heavy computation."""
        self.cfg = config or DefaultConfig()
        self._helper = HelperClass()
        self._computed_state = None

    def evaluate(
        self,
        *,
        input_data,
        **kwargs,
    ):
        """Perform computation. All computation happens here."""
        if self._computed_state is None:
            self._initialize_state()

        results = ...
        return results

    def _initialize_state(self) -> None:
        self._computed_state = ...
```

### Key Characteristics

1. **Constructor (`__init__`)**
   - Store configuration and lightweight helper objects
   - Avoid heavy computation
   - Allow lazy initialization

2. **Evaluation Method (`evaluate()`)**
   - Public method name is always `evaluate()`
   - Use keyword-only arguments
   - Return result values directly
   - May perform lazy initialization on first call

### Best Practices

- ✅ Use `evaluate()` consistently
- ✅ Keep the class stateless or minimal-state
- ✅ Use keyword-only inputs
- ✅ Return structured results where useful
- ❌ Do not perform heavy work in `__init__`

---

## Utility Classes

Utility classes provide reusable computation helpers. They should usually be functional, stateless, and easy to reuse across modules.

### Design Pattern

```python
from dataclasses import dataclass

@dataclass
class UtilityClass:
    param1: float = 0.0
    param2: float = 1.0

    def compute_something(self, input1, input2):
        """Functional method: input -> output."""
        return ...

    @staticmethod
    def static_helper(value):
        return ...
```

### Key Characteristics

- Often use `@dataclass`
- Hold configuration only
- Expose functional methods
- Avoid side effects except optional internal caching

### Best Practices

- ✅ Use `@dataclass` for simple configured helpers
- ✅ Keep methods functional
- ✅ Use `@staticmethod` when no instance state is needed
- ❌ Do not mix orchestration logic into utilities

---

## Input Classes

Input classes are dataclasses representing **external configuration and parameters** provided by a user, caller, config loader, or upstream orchestration layer.

### Design Pattern

```python
from dataclasses import dataclass, field
from typing import Optional, Any

@dataclass
class ModuleInput:
    """Configuration/parameter class for external inputs."""

    param_a: float
    param_b: str
    param_c: Optional[Any] = None
    metadata: dict = field(default_factory=dict)

    def load(self) -> "ModuleInput":
        """Optional load/configure step."""
        return self
```

### Key Characteristics

1. **Location**
   - Store in a dedicated inputs package, e.g. `<project>_inputs/`
   - Or use `internal_inputs/` for package-internal configuration

2. **Role**
   - These objects are passed into orchestrators
   - They are never populated as a result of computation

3. **Structure**
   - Always `@dataclass`
   - All fields must be typed
   - Use `field(default_factory=...)` for mutable defaults

### Best Practices

- ✅ Use dataclasses for all input contracts
- ✅ Type every field
- ✅ Keep inputs distinct from result containers
- ✅ Use `load()` only when file loading or derived defaults are needed
- ❌ Do not store computed results in input classes

---

## Container Classes

Container classes are dataclasses populated by routines/modules during computation. They hold intermediate outputs, final outputs, and structured result data.

### Design Pattern

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class ModuleResults:
    """Container for results populated by module routines."""

    output_a: float
    output_b: np.ndarray
    output_c: Optional[float] = None
    flags: dict = field(default_factory=dict)
```

### Key Characteristics

1. **Location**
   - All module dataclasses belong in `module/<module_name>/container.py`

2. **Role**
   - Populated by processing classes and orchestrators
   - Passed downstream into later modules
   - Used as the source object for visualisation

3. **Immutability**
   - Use `@dataclass(frozen=True)` when immutability is desired
   - Use mutable containers only when fields are intentionally populated incrementally

### Best Practices

- ✅ Put module containers in `container.py`
- ✅ Type every field
- ✅ Use `field(default_factory=...)` for mutable defaults
- ✅ Prefer frozen containers for stable final outputs
- ❌ Do not put container classes in the inputs package

---

## Visualisation Convention

To reduce verbosity in the top-level orchestration script, result containers own the public visualisation entry point.

### Core Rule

Each final result container should expose a `.visualise()` method that internally instantiates and delegates to its paired visualisation class.

This avoids the need for the top-level script to import or reference `Vis` classes directly.

### Design Pattern

```python
# In module/my_module/container.py

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from module.my_module.vis import ModuleResultsVis


@dataclass
class ModuleResults:
    output_a: float
    output_b: np.ndarray
    converged: bool = False

    def visualise(self, mode: str = "default") -> None:
        """Render this result container using its paired visualisation class."""
        from module.my_module.vis import ModuleResultsVis
        ModuleResultsVis(self).plot(mode=mode)
```

### Paired Visualisation Class Pattern

```python
# In module/my_module/vis.py

class ModuleResultsVis:
    def __init__(self, results: "ModuleResults") -> None:
        self._results = results

    def plot(self, mode: str = "default") -> None:
        """Render the results. Read-only access to the container."""
        ...
```

### Why this convention exists

Without this convention, the orchestration script becomes unnecessarily verbose:

```python
results = module.get_results()
vis = ModuleResultsVis(results)
vis.plot()
```

With container-owned visualisation, the orchestration script stays minimal:

```python
results = module.get_results()
results.visualise()
```

### Rules

1. **Public visualisation entry point**
   - Always expose `.visualise()` on final result containers intended for presentation
   - `.visualise()` is the preferred top-level entry point

2. **Local import inside `.visualise()`**
   - Import the paired `Vis` class inside the method body
   - This avoids circular imports between `container.py` and `vis.py`

3. **Single paired visualiser**
   - Each container should have one canonical paired visualisation class
   - Name it `<ContainerName>Vis` or `<ModuleName>ResultsVis`

4. **Read-only visualisation**
   - The `Vis` class must not recompute analysis
   - It must not mutate the container
   - It only reads from the container and renders outputs

5. **Optional mode argument**
   - `.visualise(mode="default")` may support alternate render modes such as:
     - `"default"`
     - `"summary"`
     - `"export"`
   - Prefer a single method with a small controlled mode set rather than multiple public visualization entry methods on the container

### Best Practices

- ✅ Put plotting/render logic in `vis.py`
- ✅ Put only the convenience entry point `.visualise()` in the container
- ✅ Use a local import inside `.visualise()`
- ✅ Keep orchestration scripts free of direct `Vis` imports
- ✅ Keep visualisation read-only
- ❌ Do not implement plotting logic directly inside the dataclass
- ❌ Do not import the `Vis` class at module top level in `container.py`
- ❌ Do not call `execute()` from a visualisation class

---

## Method Naming Conventions

### Orchestrator/Facade Classes

| Method Type | Naming Pattern | Example |
|-------------|----------------|---------|
| Execution | `execute()` | `orchestrator.execute()` |
| Result retrieval | `get_*()` | `get_results()`, `get_summary()` |
| Internal helpers | `_*()` | `_validate_inputs()`, `_run_stage()` |
| Convenience properties | plain name | `results` |

### Processing Classes

| Method Type | Naming Pattern | Example |
|-------------|----------------|---------|
| Evaluation | `evaluate()` | `processor.evaluate(input_data=data)` |
| Internal helpers | `_*()` | `_initialize_state()` |

### Utility Classes

| Method Type | Naming Pattern | Example |
|-------------|----------------|---------|
| Computation | `compute_*()` or descriptive verb | `compute_score()`, `transform_coordinates()` |
| Static helpers | plain descriptive name | `normalise()` |

### Container Classes

| Method Type | Naming Pattern | Example |
|-------------|----------------|---------|
| Visualisation entry | `visualise()` | `results.visualise()` |
| Derived properties | plain name | `summary_metric` |

### Visualisation Classes

| Method Type | Naming Pattern | Example |
|-------------|----------------|---------|
| Primary render | `plot()` | `ModuleResultsVis(results).plot()` |
| Optional exports | descriptive verb | `export_csv()`, `save_png()` |

---

## Type Hints and Forward References

When classes reference each other, use `from __future__ import annotations` and `TYPE_CHECKING`.

### Pattern

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from module.other_module.container import OtherResults


class MyClass:
    def method(self, other: OtherResults) -> None:
        pass
```

### Best Practices

- ✅ Use `from __future__ import annotations`
- ✅ Use `TYPE_CHECKING` blocks for import-only type references
- ✅ Use concrete type imports inside the `TYPE_CHECKING` block
- ✅ Use local imports inside methods to avoid circular dependencies when needed
- ❌ Do not create avoidable runtime circular imports

---

## Error Handling Patterns

### Validation in Constructors

```python
def __init__(self, *, dependency=None):
    if dependency is None:
        raise ValueError("Must provide 'dependency'")
    self._dependency = dependency
```

### Execution Validation

```python
def get_results(self):
    if not self._executed:
        raise RuntimeError("execute() must be called before accessing results")
    return self._results
```

### Input Validation in Methods

```python
def evaluate(self, *, input_data):
    if input_data is None:
        raise ValueError("input_data cannot be None")
    return ...
```

### Best Practices

- ✅ Validate early and fail fast
- ✅ Use descriptive `ValueError` messages for invalid inputs
- ✅ Use `RuntimeError` for execution-order violations
- ✅ Use specific custom exceptions where appropriate
- ❌ Do not silently ignore errors
- ❌ Do not use generic exceptions when a specific one fits

---

## Orchestration Chain Pattern

The repeating orchestration unit follows this sequence:

```python
inputs = ModuleInput(...)
module = ModuleOrchestrator(inputs=inputs)
module.execute()
results = module.get_results()
results.visualise()
```

### Expanded Pattern

```python
# ── Repeating Unit ──────────────────────────────────────────────
inputs = ModuleInput(...)                  # 1. Define inputs
module = ModuleOrchestrator(inputs=inputs) # 2. Instantiate
module.execute()                           # 3. Execute
results = module.get_results()             # 4. Extract results
results.visualise()                        # 5. Visualise
# ── End Unit ────────────────────────────────────────────────────

next_inputs = NextModuleInput(
    param_x=results.output_a,
    param_y=results.output_b,
)
```

### Chain Integrity Rules

- The only valid way data passes between modules is through dataclass containers
- The top-level orchestration script should not instantiate visualisation classes directly
- Each module instance should be treated as single-run stateful and recreated for new runs
- No module should depend on another module’s internal implementation details
- Cross-module communication should occur through public input and container types only

---

## Known Inconsistencies

The following issues commonly arise and should be avoided or refactored over time:

1. **Input vs container naming**
   - Some result containers may have names that look like inputs
   - The distinction is determined by role and file location, not only class name

2. **Overly verbose top-level visualisation**
   - Older code may instantiate `Vis` classes explicitly in the orchestration script
   - Prefer container-owned `.visualise()` instead

3. **Mixed method names**
   - Some legacy classes may use `run()`, `compute()`, or `select()`
   - Public patterns should converge on `execute()`, `evaluate()`, and `visualise()`

---

## Summary

### Quick Reference

| Class Type | Constructor | Execution | Result Retrieval | Visualisation Entry | Location |
|------------|-------------|-----------|------------------|---------------------|----------|
| **Orchestrator** | Store deps, validate | `execute()` → `self` | `get_*()` / property | — | `module/*/module.py` |
| **Processing** | Store config | `evaluate()` → result | Return value | — | `module/*/module.py` |
| **Utility** | Store config | Functional methods | Return value | — | `module/utilities/*/module.py` |
| **Input** | Dataclass fields | `load()` optional | Direct field access | — | `<project>_inputs/*.py` |
| **Container** | Dataclass fields | N/A | Direct field access | `visualise()` | `module/*/container.py` |
| **Visualisation** | Accept container | `plot()` | N/A | internal only | `module/*/vis.py` |

### Design Principles

1. **Separation of concerns**: each class type has one clear role
2. **Lazy execution**: computation happens only in `execute()` or `evaluate()`
3. **Fail-fast validation**: invalid inputs raise clear errors early
4. **Idempotency**: public orchestrator `execute()` is safe to call repeatedly
5. **Low-verbosity orchestration**: top-level scripts should remain compact
6. **Type safety**: inputs and outputs use typed dataclasses
7. **Container-owned visualisation**: results know how to visualise themselves through a thin convenience method

---

## Examples

### Complete Orchestrator Example

```python
class MyOrchestrator:
    def __init__(self, *, dependency1, dependency2):
        if dependency1 is None or dependency2 is None:
            raise ValueError("All dependencies required")
        self._dependency1 = dependency1
        self._dependency2 = dependency2
        self._executed = False
        self._results = None

    def execute(self) -> "MyOrchestrator":
        if self._executed:
            return self
        self._results = MyResults(output_a=1.0, output_b=np.array())[4][5][1]
        self._executed = True
        return self

    def get_results(self) -> "MyResults":
        if not self._executed:
            raise RuntimeError("execute() must be called first")
        return self._results
```

### Complete Container + Visualisation Example

```python
# container.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class MyResults:
    output_a: float
    output_b: np.ndarray

    def visualise(self, mode: str = "default") -> None:
        from module.my_module.vis import MyResultsVis
        MyResultsVis(self).plot(mode=mode)
```

```python
# vis.py
class MyResultsVis:
    def __init__(self, results: "MyResults") -> None:
        self._results = results

    def plot(self, mode: str = "default") -> None:
        ...
```

### Complete Top-Level Usage Example

```python
inputs = ModuleInput(param_a=1.0, param_b="example")
module = ModuleOrchestrator(inputs=inputs)

results = module.execute().get_results()
results.visualise()

next_inputs = NextModuleInput(
    param_x=results.output_a,
    param_y=results.output_b,
)
```

---

*These conventions define a reusable, subject-agnostic API pattern for agentic orchestration chains and are intended as a project-wide reference.*