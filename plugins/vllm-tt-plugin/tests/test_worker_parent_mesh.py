# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import Mock

import pytest
import vllm_tt_plugin.worker as worker


class _FakeMesh:
    def __init__(self, num_devices: int):
        self.num_devices = num_devices
        self.created = []

    def create_submesh(self, shape, *, offset):
        child = _FakeMesh(2)
        self.created.append((tuple(shape), tuple(offset), child))
        return child

    def get_num_devices(self):
        return self.num_devices

    def get_submeshes(self):
        return [entry[2] for entry in self.created]


def test_parent_mesh_opens_physical_fabric_and_returns_requested_submesh(monkeypatch):
    parent = _FakeMesh(4)
    open_mesh = Mock(return_value=parent)
    set_fabric = Mock()
    monkeypatch.setattr(worker, "get_mesh_grid", lambda _rank=0: (1, 2))
    monkeypatch.setattr(worker, "device_params_from_tt_config", lambda *_: {})
    monkeypatch.setattr(worker, "get_dispatch_core_config", lambda *_: object())
    monkeypatch.setattr(worker, "set_fabric", set_fabric)
    monkeypatch.setattr(worker.ttnn, "open_mesh_device", open_mesh)

    child = worker.open_mesh_device(
        {"parent_mesh_shape": [2, 2], "submesh_offset": [0, 0]},
        "all",
    )

    assert child is parent.created[0][2]
    assert parent.created[0][:2] == ((1, 2), (0, 0))
    assert tuple(open_mesh.call_args.args[0]) == (2, 2)
    set_fabric.assert_called_once_with(
        {"parent_mesh_shape": [2, 2], "submesh_offset": [0, 0]}, 4
    )
    assert worker._PARENT_MESH_BY_SUBMESH_ID[id(child)] is parent


def test_parent_mesh_close_releases_child_then_parent_and_resets_physical_fabric(
    monkeypatch,
):
    parent = _FakeMesh(4)
    child = parent.create_submesh(
        worker.ttnn.MeshShape(1, 2), offset=worker.ttnn.MeshCoordinate(0, 0)
    )
    worker._PARENT_MESH_BY_SUBMESH_ID[id(child)] = parent
    closed = []
    reset_fabric = Mock()
    read_profiler = Mock()
    monkeypatch.setattr(worker.ttnn, "ReadDeviceProfiler", read_profiler)
    monkeypatch.setattr(worker.ttnn, "close_mesh_device", closed.append)
    monkeypatch.setattr(worker, "reset_fabric", reset_fabric)

    config = {"parent_mesh_shape": [2, 2]}
    worker.close_mesh_device(child, config)

    assert closed == [child, parent]
    read_profiler.assert_not_called()
    reset_fabric.assert_called_once_with(config, 4)
    assert id(child) not in worker._PARENT_MESH_BY_SUBMESH_ID


def test_direct_mesh_close_does_not_read_device_profiler(monkeypatch):
    mesh = _FakeMesh(1)
    closed = []
    read_profiler = Mock()
    reset_fabric = Mock()
    monkeypatch.setattr(worker.ttnn, "ReadDeviceProfiler", read_profiler)
    monkeypatch.setattr(worker.ttnn, "close_mesh_device", closed.append)
    monkeypatch.setattr(worker, "reset_fabric", reset_fabric)

    worker.close_mesh_device(mesh, {})

    assert closed == [mesh]
    read_profiler.assert_not_called()
    reset_fabric.assert_called_once_with({}, 1)


@pytest.mark.parametrize("has_model_runner", [False, True])
def test_worker_shutdown_closes_owned_meshes_once(monkeypatch, has_model_runner):
    parent = _FakeMesh(4)
    mesh = parent.create_submesh((1, 2), offset=(0, 0))
    nested = mesh.create_submesh((1, 1), offset=(0, 0))
    worker._PARENT_MESH_BY_SUBMESH_ID[id(mesh)] = parent
    closed = []
    reset_fabric = Mock()
    monkeypatch.setattr(worker.ttnn, "close_mesh_device", closed.append)
    monkeypatch.setattr(worker, "reset_fabric", reset_fabric)
    config = {"parent_mesh_shape": [2, 2]}
    monkeypatch.setattr(worker, "get_tt_config", lambda _: config)
    instance = worker.TTWorker.__new__(worker.TTWorker)
    instance.mesh_device = mesh
    instance.vllm_config = object()
    if has_model_runner:
        instance.model_runner = object()

    instance.shutdown()
    instance.shutdown()
    instance.__del__()

    assert closed == [nested, mesh, parent]
    assert instance.mesh_device is None
    assert not hasattr(instance, "model_runner")
    reset_fabric.assert_called_once_with(config, 4)
    assert id(mesh) not in worker._PARENT_MESH_BY_SUBMESH_ID


@pytest.mark.parametrize("has_mesh_attribute", [False, True])
def test_worker_shutdown_without_device(monkeypatch, has_mesh_attribute):
    close_mesh = Mock()
    monkeypatch.setattr(worker, "close_mesh_device", close_mesh)
    instance = worker.TTWorker.__new__(worker.TTWorker)
    instance.model_runner = object()
    if has_mesh_attribute:
        instance.mesh_device = None

    instance.shutdown()
    instance.__del__()

    assert not hasattr(instance, "model_runner")
    close_mesh.assert_not_called()


def test_worker_destructor_uses_shutdown_without_model_runner(monkeypatch):
    close_mesh = Mock()
    monkeypatch.setattr(worker, "close_mesh_device", close_mesh)
    monkeypatch.setattr(worker, "get_tt_config", lambda _: {})
    instance = worker.TTWorker.__new__(worker.TTWorker)
    mesh = object()
    instance.mesh_device = mesh
    instance.vllm_config = object()

    instance.__del__()
    instance.__del__()

    close_mesh.assert_called_once_with(mesh, {})
    assert instance.mesh_device is None


def test_explicit_shutdown_surfaces_close_failure(monkeypatch):
    close_mesh = Mock(side_effect=RuntimeError("mesh close failed"))
    logger = Mock()
    monkeypatch.setattr(worker, "close_mesh_device", close_mesh)
    monkeypatch.setattr(worker, "get_tt_config", lambda _: {})
    monkeypatch.setattr(worker, "logger", logger)
    instance = worker.TTWorker.__new__(worker.TTWorker)
    mesh = object()
    instance.mesh_device = mesh
    instance.vllm_config = object()

    with pytest.raises(RuntimeError, match="mesh close failed"):
        instance.shutdown()

    assert instance.mesh_device is mesh
    logger.info.assert_called_once_with("TTWorker mesh shutdown starting")
    # Destruction remains best effort, while a failed explicit call propagates.
    instance.__del__()
    instance.mesh_device = None
