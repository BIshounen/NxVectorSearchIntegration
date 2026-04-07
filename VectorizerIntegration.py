"""
DeviceAgent + VectorizerIntegration — async rewrite.

DeviceAgent owns a WebRTCClient, receives metadata callbacks,
throttles per-track, and offloads vectorization to a thread via
asyncio.to_thread (keeping the event loop free).

VectorizerIntegration manages the lifecycle of DeviceAgents within
the NX Analytics API integration framework.
"""

from __future__ import annotations
import asyncio
import logging
import time
from typing import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from av import VideoFrame

import config
import rest_utils
from AnalyticsAPIIntegration import AnalyticsAPIIntegration
from vectorizer import Vectorizer
from WebRTCTool import WebRTCClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("device_agent")


def _log(msg):
    print(f"[AGENT] {msg}", flush=True)


# ======================================================================
# DeviceAgent
# ======================================================================

class DeviceAgent:
    """
    Async device agent that consumes a WebRTC stream, parses detection
    metadata, and vectorizes cropped objects via OpenCLIP + Qdrant.

    Lifecycle:
        agent = DeviceAgent(...)
        await agent.start()   # spawns WebRTC task
        ...
        await agent.stop()    # tears down cleanly
    """

    def __init__(
        self,
        engine_id: str,
        agent_id: str,
        json_rpc_client,
        credentials: dict,
        server_url: str,
        vectorizer: Vectorizer | None = None,
        on_vectors: Callable | None = None,
        per_track_interval: float = 2.0,
    ):
        self.engine_id = engine_id
        self.agent_id = agent_id
        self.json_rpc_client = json_rpc_client
        self.credentials = credentials
        self.vectorizer = vectorizer
        self.on_vectors = on_vectors
        self.settings: dict = {}

        self._per_track_interval = per_track_interval
        self._track_last_vectorized: dict[str, float] = {}
        self._vectorize_busy = False

        self._metadata_total = 0
        self._metadata_with_detections = 0
        self._pending_detections: list[dict] = []

        self._rtc_task: asyncio.Task | None = None
        self._is_shutting_down = False

        # Dedicated executor for CPU-bound vectorization
        # (survives longer than default executor during shutdown)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"vec-{agent_id}")

        # Build WebRTC client
        webrtc_url = rest_utils._concat_url(
            server_url=server_url,
            scheme="wss",
            path=rest_utils.WEBRTC_PATH.format(device_id=agent_id),
        )
        token = rest_utils.authorize(
            server_url=server_url, credentials=credentials
        )
        _log(f"URL: {webrtc_url}")
        _log(f"Token: {'obtained' if token else 'MISSING!'}")

        self._webrtc = WebRTCClient(url=webrtc_url, token=token)
        self._webrtc.on_metadata = self._on_metadata

        print(f"DeviceAgent.__init__ complete for {agent_id}", flush=True)

    # ── lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        """Spawn the WebRTC connection as a background task."""
        print(f"DeviceAgent.start() called for {self.agent_id}", flush=True)
        self._rtc_task = asyncio.create_task(
            self._webrtc.run(), name=f"webrtc-{self.agent_id}"
        )
        _log(f"Started WebRTC task for {self.agent_id}")

    async def stop(self) -> None:
        """Close WebRTC and cancel the background task."""
        print(f"DeviceAgent.stop() called for {self.agent_id}", flush=True)
        self._is_shutting_down = True
        await self._webrtc.close()
        if self._rtc_task and not self._rtc_task.done():
            self._rtc_task.cancel()
            try:
                await self._rtc_task
            except asyncio.CancelledError:
                pass
        # Shutdown the dedicated executor
        self._executor.shutdown(wait=True, cancel_futures=False)
        _log(f"Stopped agent {self.agent_id}")

    def set_settings(self, values: dict) -> None:
        self.settings = values

    # ── metadata handler (called from WebRTCClient) ───────────────────

    async def _on_metadata(self, data: dict, frame: VideoFrame | None):
        """
        Async callback fired by WebRTCClient on each datachannel message.
        Parses detections, throttles per-track, offloads vectorization.
        """
        logger.info('hit on_metadata')
        logger.info(str(data))
        self._metadata_total += 1

        if self._metadata_total <= 3 or self._metadata_total % 500 == 0:
            _log(f"[diag] metadata #{self._metadata_total}: {str(data)[:300]}")

        detections = _extract_detections(data)
        if not detections:
            return

        self._metadata_with_detections += 1

        # Convert av.VideoFrame → numpy if we have one
        np_frame = frame.to_ndarray(format="bgr24") if frame is not None else None

        if np_frame is None:
            # Buffer until first frame arrives
            self._pending_detections = detections
            return

        # Drain anything buffered before the first frame
        if self._pending_detections:
            detections = self._pending_detections + detections
            self._pending_detections = []

        if self._vectorize_busy:
            return

        # Per-track throttle
        now = time.monotonic()
        throttled = []
        for det in detections:
            tid = det.get("track_id")
            if tid and tid in self._track_last_vectorized:
                if now - self._track_last_vectorized[tid] < self._per_track_interval:
                    continue
            throttled.append(det)
            if tid:
                self._track_last_vectorized[tid] = now

        if not throttled:
            return

        # Don't schedule work if shutting down
        if self._is_shutting_down:
            _log("Skipping vectorization: agent is shutting down")
            return

        _log(f"Vectorizing {len(throttled)} detection(s) from {len(detections)} total")

        # Stale track cleanup
        cutoff = now - 60.0
        stale = [k for k, v in self._track_last_vectorized.items() if v < cutoff]
        for k in stale:
            del self._track_last_vectorized[k]

        # Offload CPU-bound vectorization to a thread
        self._vectorize_busy = True
        asyncio.create_task(self._vectorize(np_frame, throttled))

    async def _vectorize(self, frame: np.ndarray, detections: list[dict]):
        """Run vectorization in a thread pool, then handle results."""
        try:
            # Check shutdown flag before starting
            if self._is_shutting_down:
                return

            if self.vectorizer is not None:
                # Use dedicated executor to survive interpreter shutdown
                loop = asyncio.get_running_loop()
                results = await loop.run_in_executor(
                    self._executor,
                    self.vectorizer.process_frame,
                    frame,
                    detections,
                    {
                        "device_id": self.agent_id,
                        "engine_id": self.engine_id,
                    },
                )
                if results:
                    _log(f"Vectorized {len(results)} object(s)")
                    if self.on_vectors:
                        self.on_vectors(results)
            else:
                _log(
                    f"Metadata: {len(detections)} detection(s) "
                    "(no vectorizer configured)"
                )
        except RuntimeError as e:
            if "cannot schedule new futures" in str(e) or "interpreter shutdown" in str(e):
                # Gracefully handle shutdown race condition
                _log("Vectorization skipped: interpreter shutting down")
            else:
                raise
        except Exception as e:
            import traceback

            _log(f"ERROR: Vectorization failed: {e}")
            _log(traceback.format_exc())
        finally:
            self._vectorize_busy = False


# ======================================================================
# Detection parsing
# ======================================================================

def _extract_detections(data) -> list[dict]:
    objects_raw = []
    envelope_timestamp_us = 0

    if isinstance(data, dict) and "metadata" in data:
        inner = data["metadata"]
        if isinstance(inner, dict):
            envelope_timestamp_us = (
                inner.get("timestampUs")
                or inner.get("timestamp_us")
                or 0
            )
            if "objectMetadataList" in inner:
                objects_raw.extend(inner["objectMetadataList"])
            if not objects_raw and "bestShot" in inner:
                bs = inner["bestShot"]
                if isinstance(bs, dict) and "boundingBox" in bs:
                    objects_raw.append(bs)
            data = inner

    if not objects_raw:
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if "objects" in item:
                        objects_raw.extend(item["objects"])
                    elif "objectMetadataList" in item:
                        objects_raw.extend(item["objectMetadataList"])
                    elif _looks_like_object(item):
                        objects_raw.append(item)
        elif isinstance(data, dict):
            if "objects" in data:
                objects_raw.extend(data["objects"])
            elif "objectMetadataList" in data:
                objects_raw.extend(data["objectMetadataList"])
            elif _looks_like_object(data):
                objects_raw.append(data)
            else:
                for key in ("items", "detections", "results"):
                    if key in data and isinstance(data[key], list):
                        objects_raw.extend(data[key])
                        break

    return [
        det
        for obj in objects_raw
        if (det := _parse_object(obj, envelope_timestamp_us)) is not None
    ]


def _looks_like_object(d: dict) -> bool:
    return any(
        k in d
        for k in ("objectRegion", "boundingBox", "bbox", "region", "rect")
    )


def _parse_object(obj: dict, envelope_timestamp_us: int = 0) -> dict | None:
    bbox = None
    for key in ("objectRegion", "boundingBox", "bbox", "region", "rect"):
        if key in obj:
            bbox = obj[key]
            break
    if bbox is None:
        return None

    return {
        "bbox": bbox,
        "track_id": str(
            obj.get("objectId")
            or obj.get("trackId")
            or obj.get("track_id")
            or obj.get("id")
            or "unknown"
        ),
        "type": str(
            obj.get("objectTypeId")
            or obj.get("type")
            or obj.get("typeId")
            or obj.get("label")
            or "unknown"
        ),
        "timestamp_us": (
            obj.get("timestampUs")
            or obj.get("timestamp")
            or envelope_timestamp_us
        ),
        "attributes": obj.get("attributes", []),
    }


# ======================================================================
# VectorizerIntegration
# ======================================================================

class VectorizerIntegration(AnalyticsAPIIntegration):

    def __init__(
        self,
        server_url: str,
        integration_manifest: dict,
        engine_manifest: dict,
        credentials_path: str,
        device_agent_manifest: dict,
    ):
        super().__init__(
            server_url=server_url,
            integration_manifest=integration_manifest,
            engine_manifest=engine_manifest,
            credentials_path=credentials_path,
        )

        self.device_agents: dict[str, DeviceAgent] = {}
        self.device_agent_manifest = device_agent_manifest

        self.vectorizer = Vectorizer(
            qdrant_url=config.qdrant_url,
            qdrant_api_key=config.qdrant_api_key,
            qdrant_collection=rest_utils.get_site_id(
                server_url=server_url, credentials=self.credentials
            ),
            bbox_increase_rate=0.1,
            remove_bg=False
        )

        # Track if we're in an async context
        self._keep_running = True
        self._loop: asyncio.AbstractEventLoop | None = None

    def get_device_agent_manifest(self, device_parameters: dict) -> dict:
        return self.device_agent_manifest

    def on_device_agent_created(self, device_parameters):
        device_agent_id = device_parameters["parameters"]["id"].strip("{}")
        engine_id = device_parameters["target"]["engineId"].strip("{}")

        if device_agent_id in self.device_agents:
            logger.warning("Agent %s already exists, skipping", device_agent_id)
            return  # Don't create again!

        agent = DeviceAgent(
            agent_id=device_agent_id,
            json_rpc_client=self.JSONRPC,
            engine_id=engine_id,
            credentials=self.credentials,
            server_url=self.server_url,
            vectorizer=self.vectorizer,
        )
        self.device_agents[device_agent_id] = agent

        # Schedule agent.start() in the event loop
        if self._loop:
            asyncio.run_coroutine_threadsafe(agent.start(), self._loop)
            _log(f"Scheduled agent {device_agent_id} to start")
        else:
            logger.error("Event loop not available, cannot start agent")

    def on_device_agent_deletion(self, device_id):
        logger.warning("on_device_agent_deleted called: %s", device_id)
        device_agent_id = device_id.strip("{}")
        agent = self.device_agents.pop(device_agent_id, None)
        if agent and self._loop:
            asyncio.run_coroutine_threadsafe(agent.stop(), self._loop)
            _log(f"Scheduled agent {device_agent_id} to stop")

    def on_agent_active_settings_change(self, parameters, device_id):
        self.device_agents[device_id].set_settings(parameters["settingsValues"])
        return {
            "settingsValues": parameters["settingsValues"],
            "settingsModel": self.engine_manifest["deviceAgentSettingsModel"],
        }

    def on_agent_settings_update(self, parameters, device_id):
        if device_id in self.device_agents:
            self.device_agents[device_id].set_settings(
                parameters["settingsValues"]
            )
            return {
                "settingsValues": self.device_agents[device_id].settings,
                "settingsModel": self.engine_manifest[
                    "deviceAgentSettingsModel"
                ],
            }
        return {
            "settingsValues": parameters["settingsValues"],
            "settingsModel": self.engine_manifest["deviceAgentSettingsModel"],
        }

    def on_engine_settings_update(self, parameters):
        return {
            "settingsValues": parameters["settingsValues"],
            "settingsModel": self.integration_manifest["engineSettingsModel"],
        }

    def on_engine_active_settings_change(self, parameters):
        value_to_save = parameters.get("params", {}).get("parameter", None)
        if value_to_save is not None:
            return {
                "settingsValues": {
                    "settingsValues": {"saved_param": value_to_save}
                },
                "settingsModel": self.integration_manifest[
                    "engineSettingsModel"
                ],
            }
        return {
            "settingsValues": parameters["settingsValues"],
            "settingsModel": self.integration_manifest["engineSettingsModel"],
        }

    def get_integration_engine_side_settings(self, parameters):
        return {"value_to_save": "123"}

    async def main(self):
        """Override parent's main() to keep the event loop alive."""
        # Store the event loop reference for scheduling tasks
        self._loop = asyncio.get_running_loop()

        # Call parent's setup (JSON-RPC, subscriptions, etc.)
        await super().main()

        _log("Integration setup complete, keeping event loop alive...")

        # Keep the loop running until interrupted
        try:
            while self._keep_running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            _log("Main loop cancelled")
        finally:
            await self._async_cleanup()

    def stop(self):
        """Signal the integration to stop."""
        _log("Stop requested")
        self._keep_running = False

    async def _async_cleanup(self):
        """Async cleanup of all device agents."""
        _log("Starting async cleanup...")

        # Stop all agents
        for device_id, agent in list(self.device_agents.items()):
            _log(f"Stopping agent {device_id}")
            try:
                await agent.stop()
            except Exception as e:
                _log(f"Error stopping agent {device_id}: {e}")

        _log("Async cleanup complete")

    def run(self):
        """Start the integration and block until interrupted."""
        _log("Starting VectorizerIntegration...")

        try:
            # Call parent's run() method - this starts the event loop
            super().run()
        except KeyboardInterrupt:
            _log("Keyboard interrupt received")
        finally:
            _log("VectorizerIntegration stopped")