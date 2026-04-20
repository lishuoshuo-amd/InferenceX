"""Synchronous SaFE Platform API client for CI workload management.

Adapted from Hyperloom OOB_mcp_server/safe_client.py (async) to a
synchronous requests-based client suitable for GitHub Actions CI.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any

import requests

log = logging.getLogger(__name__)


class SaFEClient:
    """Synchronous client for SaFE platform workload API."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        verify_ssl: bool = True,
    ):
        self.api_key = api_key
        self.base_url = (base_url or os.environ.get("SAFE_API_BASE", "")).rstrip("/")
        if not self.base_url:
            raise ValueError("SaFE API base URL not configured")
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        self._session.verify = verify_ssl

    def create_workload(
        self,
        workspace_id: str,
        name: str,
        image: str,
        command: str,
        gpu_count: int = 8,
        cpu: int = 4,
        memory: str = "128Gi",
        ephemeral_storage: str = "100Gi",
        replicas: int = 1,
        rdma: str = "1k",
        env_vars: dict[str, str] | None = None,
        force_host_network: bool = True,
    ) -> dict[str, Any]:
        """Create a PyTorchJob workload on SaFE.

        Args:
            command: Shell script content (will be base64 encoded).
        """
        url = f"{self.base_url}/api/v1/workloads"
        entry_point = base64.b64encode(command.encode()).decode()

        payload: dict[str, Any] = {
            "displayName": name,
            "description": f"verify-pr benchmark: {name}",
            "workspaceId": workspace_id,
            "groupVersionKind": {"kind": "PyTorchJob", "version": "v1"},
            "resources": [
                {
                    "cpu": str(cpu),
                    "gpu": str(gpu_count),
                    "memory": memory,
                    "ephemeralStorage": ephemeral_storage,
                    "replica": replicas,
                    "rdmaResource": rdma,
                }
            ],
            "images": [image],
            "entryPoints": [entry_point],
            "priority": 1,
            "maxRetry": 0,
            "ttlSecondsAfterFinished": 0,
            "forceHostNetwork": force_host_network,
        }
        if env_vars:
            payload["env"] = env_vars

        log.info("POST %s payload keys: %s", url, list(payload.keys()))
        log.info("workspaceId=%s, image=%s, gpu=%s", workspace_id, image, gpu_count)
        resp = self._session.post(url, json=payload, timeout=30)
        if resp.status_code >= 400:
            log.error("SaFE API error %d: %s", resp.status_code, resp.text[:2000])
        resp.raise_for_status()
        data = resp.json()
        wl_id = data.get("id") or data.get("workloadId", "")
        log.info("Workload created: %s (name=%s)", wl_id, name)
        return data

    def get_workload(self, workload_id: str) -> dict[str, Any]:
        url = f"{self.base_url}/api/v1/workloads/{workload_id}"
        resp = self._session.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def stop_workload(self, workload_id: str) -> bool:
        url = f"{self.base_url}/api/v1/workloads/{workload_id}/stop"
        resp = self._session.post(url, timeout=10)
        if resp.status_code not in (200, 204):
            log.error("Failed to stop %s: %d %s", workload_id, resp.status_code, resp.text)
            return False
        return True

    def get_workload_status(self, workload_id: str) -> tuple[str, str]:
        """Return (status, phase) for a workload."""
        data = self.get_workload(workload_id)
        status = (data.get("status") or "").lower()
        phase = (data.get("phase") or "").lower()
        return status, phase

    def poll_until_done(
        self,
        workload_id: str,
        timeout: int = 7200,
        interval: int = 30,
        heartbeat_interval: int = 300,
    ) -> str:
        """Poll workload status until terminal state.

        Returns: 'completed', 'failed', or 'timeout'.
        """
        start = time.time()
        last_heartbeat = start

        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                log.warning("Workload %s timed out after %ds", workload_id, timeout)
                self.stop_workload(workload_id)
                return "timeout"

            try:
                status, phase = self.get_workload_status(workload_id)
            except Exception as e:
                log.warning("Poll error for %s: %s", workload_id, e)
                time.sleep(interval)
                continue

            if phase in ("succeeded", "completed") or status in ("succeeded", "completed"):
                log.info("Workload %s completed (%.0fs)", workload_id, elapsed)
                return "completed"

            if phase == "failed" or status == "failed":
                log.error("Workload %s failed (%.0fs)", workload_id, elapsed)
                return "failed"

            if time.time() - last_heartbeat > heartbeat_interval:
                log.info("Workload %s still running... (%.0f min, status=%s, phase=%s)",
                         workload_id, elapsed / 60, status, phase)
                last_heartbeat = time.time()

            time.sleep(interval)
