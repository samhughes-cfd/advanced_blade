"""Frozen bundle threaded through ``main_precompute`` stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .component_materials import ComponentMaterialsMap
from .system_layout import SystemLayoutSpec


@dataclass(frozen=True)
class PrecomputeOrchestrationContext:
    system_type_key: str
    layout: SystemLayoutSpec
    component_materials: ComponentMaterialsMap

    def job_meta(self) -> dict[str, Any]:
        return {
            "system_type_key": self.system_type_key,
            "system_layout": self.layout.to_jsonable(),
            "component_materials": self.component_materials.to_dict(),
        }
