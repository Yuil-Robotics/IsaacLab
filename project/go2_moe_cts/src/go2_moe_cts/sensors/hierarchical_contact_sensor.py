# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX contact sensor support for nested URDF rigid-body prims."""

from __future__ import annotations

import re

from isaaclab_physx.physics import PhysxManager as SimulationManager
from isaaclab_physx.sensors.contact_sensor.contact_sensor import ContactSensor

from pxr import UsdPhysics

from isaaclab.sensors.contact_sensor import BaseContactSensor, ContactSensorCfg
from isaaclab.sim.utils.queries import get_all_matching_child_prims, resolve_matching_prims_from_source


class HierarchicalContactSensor(ContactSensor):
    """Contact sensor that builds PhysX views from complete nested body paths."""

    def __init__(self, cfg: ContactSensorCfg):
        self._sensor_body_names: list[str] = []
        super().__init__(cfg)

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies with contact sensors attached."""
        if self._sensor_body_names:
            return self._sensor_body_names
        return super().body_names

    def _initialize_impl(self) -> None:
        """Initialize contact views without assuming sibling rigid-body prims."""
        BaseContactSensor._initialize_impl(self)
        self._physics_sim_view = SimulationManager.get_physics_sim_view()

        parent_expr, leaf_pattern = self.cfg.prim_path.rsplit("/", 1)
        name_pattern = re.compile(leaf_pattern)

        def has_contact_report(prim) -> bool:
            if not bool(name_pattern.fullmatch(prim.GetName())):
                return False
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return False
            applied = prim.GetAppliedSchemas()
            if "PhysxContactReportAPI" in applied:
                return True
            api_schemas = prim.GetMetadata("apiSchemas")
            if api_schemas:
                if hasattr(api_schemas, "explicitItems") and "PhysxContactReportAPI" in api_schemas.explicitItems:
                    return True
                if hasattr(api_schemas, "prependedItems") and "PhysxContactReportAPI" in api_schemas.prependedItems:
                    return True
            return False

        matches = resolve_matching_prims_from_source(parent_expr)
        if not matches:
            raise RuntimeError(f"No prim found at '{parent_expr}'.")
        asset_prim, asset_expr = matches[0]
        source_root = asset_prim.GetPath().pathString
        prims = get_all_matching_child_prims(
            source_root,
            predicate=has_contact_report,
            traverse_instance_prims=False,
        )
        if not prims:
            raise RuntimeError(
                f"Sensor at path '{self.cfg.prim_path}' could not find any bodies with contact reporter API."
                "\nHINT: Make sure to enable 'activate_contact_sensors' in the corresponding asset spawn configuration."
            )

        self._sensor_body_names = [prim.GetName() for prim in prims]

        # Resolve environment root paths in environment-major order (env_0, env_1, ...)
        if re.search(r"/env_\d+(/|$)", source_root):
            env_roots = [re.sub(r"/env_\d+(/|$)", f"/env_{i}\\1", source_root, count=1) for i in range(self._num_envs)]
        elif hasattr(self, "_parent_prims") and len(self._parent_prims) == self._num_envs:

            def _extract_env_idx(prim_path: str) -> int:
                m = re.search(r"_(\d+)(?:/|$)", prim_path)
                return int(m.group(1)) if m else 0

            sorted_parents = sorted(self._parent_prims, key=lambda p: _extract_env_idx(p.GetPath().pathString))
            env_roots = [p.GetPath().pathString for p in sorted_parents]
        else:
            env_roots = [source_root] * self._num_envs

        # Construct concrete body paths ordered as env_0 (all bodies), env_1 (all bodies), ...
        # This guarantees (env * num_sensors + sensor) alignment for PhysX tensors and Warp kernels.
        all_body_paths = []
        for env_root in env_roots:
            for prim in prims:
                rel = prim.GetPath().pathString[len(source_root) :]
                all_body_paths.append(f"{env_root}{rel}")

        filter_prim_paths_glob = [expr.replace(".*", "*") for expr in self.cfg.filter_prim_paths_expr]
        if filter_prim_paths_glob:
            contact_filter_patterns = [filter_prim_paths_glob for _ in range(len(all_body_paths))]
        else:
            contact_filter_patterns = []

        self._body_physx_view = self._physics_sim_view.create_rigid_body_view(all_body_paths)
        self._contact_view = self._physics_sim_view.create_rigid_contact_view(
            all_body_paths,
            filter_patterns=contact_filter_patterns,
            max_contact_data_count=self.cfg.max_contact_data_count_per_prim * len(prims) * self._num_envs,
        )
        self._num_sensors = self.body_physx_view.count // self._num_envs
        if self._num_sensors != len(prims):
            raise RuntimeError(
                "Failed to initialize hierarchical contact reporter for specified bodies."
                f"\n\tInput prim path    : {self.cfg.prim_path}"
                f"\n\tResolved prim paths (first env): {all_body_paths[: len(prims)]}"
            )

        if self.cfg.track_contact_points or self.cfg.track_friction_forces:
            if not self.cfg.filter_prim_paths_expr:
                raise ValueError(
                    "The 'filter_prim_paths_expr' is empty. Specify a filter pattern when tracking contact points or "
                    "friction forces."
                )
            if self.cfg.max_contact_data_count_per_prim < 1:
                raise ValueError("The 'max_contact_data_count_per_prim' must be greater than zero.")

        self._create_buffers()

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        super()._invalidate_initialize_callback(event)
        self._sensor_body_names = []
