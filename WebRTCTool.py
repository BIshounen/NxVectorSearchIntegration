"""
WebRTCClient — reusable WebRTC transport for NX VMS camera streams.

Connects via WebSocket signaling, receives video frames and analytics
metadata over a DataChannel, and invokes callbacks for each.

Intended to be owned by a DeviceAgent, not used standalone.

Requires: aiortc, aiohttp, av
"""

from __future__ import annotations
import ssl
import asyncio
import json
import logging
from typing import Awaitable, Callable, Union

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
logger = logging.getLogger("webrtc_client")
logging.basicConfig(level=logging.INFO)

# ──────────────────────────────────────────────────────────────────────
# STUN ERROR debugging patch - MUST BE BEFORE any aioice/aiortc imports
# ──────────────────────────────────────────────────────────────────────
import aioice.stun

_orig_parse = aioice.stun.parse_message

def _debug_parse_message(data: bytes):
    # Check for error response BEFORE parsing
    if len(data) >= 2:
        msg_type = (data[0] << 8) | data[1]
        # Error response = 0x0111 for Binding Error Response
        if msg_type == 0x0111:
            print(f"\n{'='*60}", flush=True)
            print(f"[RAW STUN ERROR] Hex dump:", flush=True)
            print(f"  {data.hex()}", flush=True)
            # Parse attributes manually
            pos = 20  # Skip STUN header
            while pos + 4 <= len(data):
                attr_type = (data[pos] << 8) | data[pos+1]
                attr_len = (data[pos+2] << 8) | data[pos+3]
                print(f"  Raw attr at {pos}: type=0x{attr_type:04x}, len={attr_len}", flush=True)
                if attr_type == 0x0009:  # ERROR-CODE
                    err_data = data[pos+4:pos+4+attr_len]
                    if len(err_data) >= 4:
                        err_class = err_data[2]
                        err_number = err_data[3]
                        err_code = err_class * 100 + err_number
                        reason = err_data[4:].decode('utf-8', errors='ignore')
                        print(f"  >>> FOUND ERROR-CODE: {err_code} '{reason}'", flush=True)
                pos += 4 + attr_len
                # Pad to 4-byte boundary
                if attr_len % 4:
                    pos += 4 - (attr_len % 4)
            print(f"{'='*60}\n", flush=True)

    msg = _orig_parse(data)
    return msg

aioice.stun.parse_message = _debug_parse_message
print("[PATCH] STUN parse_message patched for RAW error debugging", flush=True)

# ──────────────────────────────────────────────────────────────────────
# Patch to log OUTGOING STUN requests
# ──────────────────────────────────────────────────────────────────────
import aioice.ice

_orig_build_request = aioice.ice.Connection.build_request

def _patched_build_request(self, pair, nominate=False):
    try:
        request = _orig_build_request(self, pair, nominate)

        # REORDER ATTRIBUTES to match browser order:
        # Browser: USERNAME -> NETWORK-COST -> ICE-CONTROLLING -> PRIORITY
        # aioice:  USERNAME -> PRIORITY -> ICE-CONTROLLING
        # We swap PRIORITY and ICE-CONTROLLING (can't easily add NETWORK-COST)

        if 'PRIORITY' in request.attributes and 'ICE-CONTROLLING' in request.attributes:
            priority_val = request.attributes['PRIORITY']
            ice_ctrl_val = request.attributes['ICE-CONTROLLING']

            new_attrs = {}
            for key, val in request.attributes.items():
                if key == 'PRIORITY':
                    continue  # Skip, add after ICE-CONTROLLING
                elif key == 'ICE-CONTROLLING':
                    new_attrs['ICE-CONTROLLING'] = ice_ctrl_val
                    new_attrs['PRIORITY'] = priority_val
                else:
                    new_attrs[key] = val
            request.attributes = new_attrs
            print(f"[STUN REORDER] Attributes: {list(request.attributes.keys())}", flush=True)

        print(f"\n[STUN REQUEST BUILT]", flush=True)
        print(f"  Remote ufrag: {getattr(self, '_remote_username', 'N/A')}", flush=True)
        rp = getattr(self, '_remote_password', None)
        print(f"  Remote password: {rp[:10]}..." if rp else "  Remote password: None", flush=True)
        print(f"  Attributes: {list(request.attributes.keys())}", flush=True)
        return request
    except Exception as e:
        print(f"[STUN REQUEST ERROR] {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

aioice.ice.Connection.build_request = _patched_build_request
print("[PATCH] aioice.ice.Connection.build_request patched with ATTRIBUTE REORDERING", flush=True)

# ──────────────────────────────────────────────────────────────────────
# Patch to fix missing remote ICE credentials
# The issue: when BUNDLE is used, aiortc closes the first ICE transport
# and creates a new one, but doesn't always propagate the remote creds.
# We need to inject them BEFORE the connection starts checking.
# ──────────────────────────────────────────────────────────────────────
from aiortc.rtcicetransport import RTCIceTransport

_orig_ice_start = RTCIceTransport.start

async def _patched_ice_start(self, remoteParameters):
    print(f"\n[ICE START] Remote params: ufrag={remoteParameters.usernameFragment}, pwd={remoteParameters.password[:10] if remoteParameters.password else None}...", flush=True)

    # Ensure the connection has the remote credentials BEFORE calling original start
    if hasattr(self, '_connection') and self._connection:
        if remoteParameters.usernameFragment:
            self._connection._remote_username = remoteParameters.usernameFragment
            print(f"[ICE START] Set remote_username = {remoteParameters.usernameFragment}", flush=True)
        if remoteParameters.password:
            self._connection._remote_password = remoteParameters.password
            print(f"[ICE START] Set remote_password = {remoteParameters.password[:10]}...", flush=True)

    return await _orig_ice_start(self, remoteParameters)

RTCIceTransport.start = _patched_ice_start
print("[PATCH] RTCIceTransport.start patched to ensure remote credentials are set", flush=True)

# Patch Message serialization to see if password is being passed
# First, let's find out what methods Message has
msg_attrs = [m for m in dir(aioice.stun.Message) if not m.startswith('_')]
print(f"[DEBUG] aioice.stun.Message methods/attrs: {msg_attrs}", flush=True)

# Check aioice.stun module for add_message_integrity
stun_attrs = [m for m in dir(aioice.stun) if 'integrity' in m.lower() or 'message' in m.lower()]
print(f"[DEBUG] aioice.stun integrity/message related: {stun_attrs}", flush=True)

# Check aioice.ice module
ice_attrs = [m for m in dir(aioice.ice) if 'integrity' in m.lower() or 'send' in m.lower()]
print(f"[DEBUG] aioice.ice integrity/send related: {ice_attrs}", flush=True)

# Patch HMAC computation to see what password is being used
import hmac as hmac_module
_orig_hmac_new = hmac_module.new

def _patched_hmac_new(key, msg=None, digestmod=None):
    result = _orig_hmac_new(key, msg, digestmod)
    digestmod_str = str(digestmod) if digestmod else ''
    if 'sha1' in digestmod_str.lower() or digestmod_str == "<built-in function openssl_sha1>":
        key_preview = key[:30] if len(key) > 30 else key
        print(f"[HMAC-SHA1] key ({len(key)} bytes): {key_preview}", flush=True)
        if msg:
            msg_preview = msg[:60].hex() if len(msg) > 60 else msg.hex()
            print(f"[HMAC-SHA1] msg ({len(msg)} bytes): {msg_preview}...", flush=True)
    return result

hmac_module.new = _patched_hmac_new
print("[PATCH] hmac.new patched for debugging", flush=True)

# Patch the StunProtocol.send_stun to see what's happening
_orig_protocol_send = aioice.ice.StunProtocol.send_stun

def _patched_protocol_send(self, message, addr):
    # Try to get password from the receiver (Connection object)
    password = None
    if hasattr(self, 'receiver') and self.receiver:
        password = getattr(self, 'receiver')._remote_password if hasattr(self.receiver, '_remote_password') else None
        if password:
            print(f"[STUN SEND] Found password in receiver: {password!r}", flush=True)

    print(f"[STUN SEND] Sending to {addr}", flush=True)
    print(f"[STUN SEND] Message attributes BEFORE send: {list(message.attributes.keys())}", flush=True)

    result = _orig_protocol_send(self, message, addr)

    print(f"[STUN SEND] Message attributes AFTER send: {list(message.attributes.keys())}", flush=True)
    if 'MESSAGE-INTEGRITY' in message.attributes:
        mi = message.attributes['MESSAGE-INTEGRITY']
        print(f"[STUN SEND] MESSAGE-INTEGRITY value: {mi.hex() if isinstance(mi, bytes) else mi}", flush=True)
    return result

aioice.ice.StunProtocol.send_stun = _patched_protocol_send
print("[PATCH] StunProtocol.send_stun patched for detailed debugging", flush=True)

# Try to patch Connection.send_request which might be where MI is added
if hasattr(aioice.ice.Connection, 'send_request'):
    _orig_send_request = aioice.ice.Connection.send_request
    async def _patched_send_request(self, request, addr, password=None):
        pwd_str = repr(password) if password else 'None'
        print(f"[CONN SEND_REQUEST] password param: {pwd_str}", flush=True)
        remote_pwd = getattr(self, '_remote_password', 'N/A')
        print(f"[CONN SEND_REQUEST] self._remote_password: {repr(remote_pwd)}", flush=True)
        return await _orig_send_request(self, request, addr, password)
    aioice.ice.Connection.send_request = _patched_send_request
    print("[PATCH] Connection.send_request patched", flush=True)


# RSA PATCH

import aiortc_rsa_patch

# ──────────────────────────────────────────────────────────────────────
# Now import aiortc (after patches)
# ──────────────────────────────────────────────────────────────────────
import aiohttp
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.rtcdtlstransport import RTCCertificate, RTCDtlsFingerprint
from av import VideoFrame
import socket
import struct as _struct

# ──────────────────────────────────────────────────────────────────────
# SHA-256 monkey-patch for aiortc certificate fingerprint
# ──────────────────────────────────────────────────────────────────────
from cryptography.hazmat.primitives.serialization import Encoding
import hashlib

def _patched_getFingerprints(self):
    """Always return a single sha-256 fingerprint."""
    cert_der = self._cert.public_bytes(Encoding.DER)
    digest = hashlib.sha256(cert_der).hexdigest()
    fingerprint = ":".join(digest[i:i + 2].upper() for i in range(0, len(digest), 2))
    return [RTCDtlsFingerprint(algorithm="sha-256", value=fingerprint)]

RTCCertificate.getFingerprints = _patched_getFingerprints
print("[PATCH] RTCCertificate.getFingerprints patched for sha-256", flush=True)

# ──────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────
MetadataCallback = Union[
    Callable[[dict, VideoFrame | None], None],
    Callable[[dict, VideoFrame | None], Awaitable[None]],
]
FrameCallback = Union[
    Callable[[VideoFrame], None],
    Callable[[VideoFrame], Awaitable[None]],
]


# ──────────────────────────────────────────────────────────────────────
# WebRTCClient
# ──────────────────────────────────────────────────────────────────────
class WebRTCClient:
    """
    WebRTC transport that connects to an NX VMS stream endpoint.

    Provides the latest video frame and fires a callback on each
    metadata message from the DataChannel.

    Callbacks (on_metadata, on_frame) can be sync or async — the
    client detects and handles both transparently.

    Usage from an async DeviceAgent:

        client = WebRTCClient(url="wss://...", token="...")
        client.on_metadata = self._handle_metadata   # sync or async
        await client.run()   # blocks until connection closes
        await client.close()
    """

    def __init__(self, url: str, token: str) -> None:
        self.url = url
        self.token = token

        # Callbacks — set before calling run()
        self.on_metadata: MetadataCallback | None = None
        self.on_frame: FrameCallback | None = None

        # Detect local connections — skip STUN/srflx/trickle when local
        from urllib.parse import urlparse
        parsed = urlparse(url)
        host = parsed.hostname or ""
        self._is_local = host in ("127.0.0.1", "localhost", "::1") or host.startswith("172.") or host.startswith("10.") or host.startswith("192.168.")
        if self._is_local:
            logger.info("Local connection detected (%s) — skipping STUN/srflx/trickle", host)

        # Internal state
        self._pc: RTCPeerConnection | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = None
        self._latest_frame: VideoFrame | None = None
        self._closed = asyncio.Event()

    # ── public API ────────────────────────────────────────────────────

    @property
    def latest_frame(self) -> VideoFrame | None:
        """Most recently decoded video frame, or None."""
        return self._latest_frame

    @property
    def is_connected(self) -> bool:
        return (
            self._pc is not None
            and self._pc.connectionState in ("connected", "connecting")
        )

    async def run(self) -> None:
        """Connect and consume until closed or the server hangs up."""
        logger.info("=== run() START ===")
        logger.info("_closed event is_set=%s", self._closed.is_set())

        try:
            await self._open()
            logger.info("_open() completed, _closed.is_set=%s", self._closed.is_set())

            signaling_task = asyncio.create_task(
                self._signaling_loop(), name="webrtc-signaling"
            )
            closed_task = asyncio.create_task(
                self._closed.wait(), name="webrtc-closed-wait"
            )

            logger.info("Tasks created, waiting...")
            done, pending = await asyncio.wait(
                [signaling_task, closed_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            logger.info("Wait returned. Done: %s", [t.get_name() for t in done])

            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            for task in done:
                exc = task.exception() if not task.cancelled() else None
                if exc and task is signaling_task:
                    logger.exception("Signaling task failed: %s", exc)
                    raise exc

        except asyncio.CancelledError:
            import traceback
            logger.warning("run() was cancelled. Stack trace:")
            traceback.print_stack()
            raise
        except Exception:
            logger.exception("WebRTCClient run loop failed")
            raise
        finally:
            await self.close()

    async def close(self) -> None:
        """Tear down peer connection, websocket, and HTTP session."""
        if self._pc:
            await self._pc.close()
            self._pc = None
        if self._ws and not self._ws.closed:
            await self._ws.close()
            self._ws = None
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        self._closed.set()
        logger.info("WebRTCClient closed.")

    # ── signaling ─────────────────────────────────────────────────────

    async def _open(self) -> None:
        """Open the WebSocket and PeerConnection, then subscribe."""
        logger.info("=== _open() START ===")

        if self._is_local:
            # Local: no STUN servers needed, faster ICE gathering
            self._pc = RTCPeerConnection(configuration=RTCConfiguration(
                iceServers=[]
            ))
        else:
            self._pc = RTCPeerConnection(configuration=RTCConfiguration(
                iceServers=[
                    RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
                    RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
                ]
            ))
        logger.info("PC created, connectionState=%s, iceConnectionState=%s",
                    self._pc.connectionState, self._pc.iceConnectionState)

        self._setup_pc_handlers()
        logger.info("Handlers registered")

        # Create SSL context that doesn't verify certificates
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        self._session = aiohttp.ClientSession()
        logger.info("aiohttp session created")

        logger.info("Connecting WebSocket to %s", self.url)
        try:
            self._ws = await self._session.ws_connect(
                self.url,
                headers={"Authorization": f"Bearer {self.token}"},
                ssl=ssl_context,
            )
            logger.info("WebSocket connected! closed=%s", self._ws.closed)
        except aiohttp.ClientError as e:
            logger.exception("WebSocket connect FAILED (ClientError): %s", e)
            raise
        except Exception as e:
            logger.exception("WebSocket connect FAILED (unexpected): %s", e)
            raise

        # Send subscribe
        subscribe_msg = {"type": "open"}
        await self._ws.send_json(subscribe_msg)
        logger.info("Subscribe sent: %s", subscribe_msg)
        logger.info("=== _open() END ===")

    async def _signaling_loop(self) -> None:
        logger.info("Entering signaling loop, ws closed=%s", self._ws.closed)

        try:
            async for msg in self._ws:
                logger.info(
                    "WS RAW: type=%s, data=%r",
                    msg.type, msg.data[:500] if msg.data else None,
                )

                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self._handle_signaling_message(data)
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                ):
                    logger.warning("WebSocket closed/error: %s", msg.type)
                    self._closed.set()
                    break
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    logger.debug("Binary WS message (%d bytes)", len(msg.data))

        except Exception as e:
            logger.exception("Signaling loop exception: %s", e)

        logger.info("Exited signaling loop, ws closed=%s", self._ws.closed)

    async def _handle_signaling_message(self, data: dict) -> None:
        """Route a single signaling message."""

        # Handle SDP offer
        if "sdp" in data and isinstance(data["sdp"], dict):
            sdp_data = data["sdp"]
            if sdp_data.get("type") == "offer":
                logger.info("Received SDP offer from server")
                self._offer_sdp = sdp_data["sdp"]
                offer = RTCSessionDescription(sdp=self._offer_sdp, type="offer")
                await self._pc.setRemoteDescription(offer)

                answer = await self._pc.createAnswer()
                await self._pc.setLocalDescription(answer)

                while self._pc.iceGatheringState != "complete":
                    await asyncio.sleep(0.05)
                logger.info("ICE gathering complete — state=%s", self._pc.iceGatheringState)

                # Get the full local description WITH candidates
                local_sdp = self._pc.localDescription.sdp

                if self._is_local:
                    # ── LOCAL MODE: send vanilla SDP, no filtering ──
                    logger.info("Local mode — sending SDP answer as-is (no filtering/STUN/trickle)")
                    final_sdp = self._force_sha256_in_sdp(local_sdp)
                    # Fix SCTP collision: let VMS be the SCTP initiator
                    final_sdp = self._force_setup_passive_in_sdp(final_sdp)
                    logger.info("Local SDP:\n%s", final_sdp)

                    await self._ws.send_json({
                      "sdp": {
                        "type": "answer",
                        "sdp": final_sdp,
                      }
                    })
                    logger.info("Sent SDP answer (local mode)")
                    return

                # ── CLOUD/RELAY MODE: apply filtering + STUN + trickle ──

                # Filter out the 192.168.x.x candidate - it won't work anyway
                filtered_lines = []
                for line in local_sdp.splitlines():
                    if line.startswith("a=candidate:") and "192.168." in line:
                        logger.debug("Stripping unreachable local candidate: %s", line[:60])
                        continue
                    if line.startswith("c=IN IP4 192.168."):
                        line = "c=IN IP4 0.0.0.0"
                    if line.startswith("m=video ") and "192.168." in local_sdp:
                        # Reset port to 9 if we're stripping the candidate
                        parts = line.split(" ", 2)
                        if len(parts) >= 3:
                            line = f"m=video 9 {parts[2]}"
                    if line.startswith("m=application ") and "192.168." in local_sdp:
                        parts = line.split(" ", 2)
                        if len(parts) >= 3:
                            line = f"m=application 9 {parts[2]}"
                    filtered_lines.append(line)
                local_sdp = "\r\n".join(filtered_lines)

                # ── Discover public address via STUN ──
                # Try aiortc's ICE socket first (correct NAT mapping),
                # then fall back to a separate socket (works for full-cone NAT)
                public_addr = await self._discover_public_from_ice_socket()
                if public_addr:
                    logger.info(
                        "Discovered public address via ICE socket: %s:%d",
                        *public_addr,
                    )
                else:
                    logger.warning(
                        "ICE socket STUN failed, trying fallback socket"
                    )
                    public_addr = self._discover_public_address()

                if public_addr:
                    public_ip, public_port = public_addr
                    local_sdp = self._inject_srflx_candidate(
                        local_sdp, public_ip, public_port
                    )
                else:
                    logger.warning(
                        "Could not discover public address — "
                        "connection will only work over VPN / LAN"
                    )

                # Remove a=end-of-candidates — we will trickle candidates
                # via WebSocket so the relay can process them
                local_sdp = "\r\n".join(
                    line for line in local_sdp.splitlines()
                    if line.strip() != "a=end-of-candidates"
                )

                logger.info("Local SDP (filtered):\n%s", local_sdp)

                final_sdp = self._force_sha256_in_sdp(local_sdp)
                # Fix SCTP collision: let VMS be the SCTP initiator
                final_sdp = self._force_setup_passive_in_sdp(final_sdp)

                await self._ws.send_json({
                  "sdp": {
                    "type": "answer",
                    "sdp": final_sdp,
                  }
                })
                logger.info("Sent SDP answer")

                # ── Trickle ICE: send each candidate via WebSocket ──
                # The NX cloud relay processes trickle ICE candidates
                # from the signaling channel, not just SDP-embedded ones.
                await self._trickle_local_candidates(final_sdp)

                return

        # Handle ICE candidates - only accept UDP candidates with valid IPs
        if "ice" in data and isinstance(data["ice"], dict):
            ice_data = data["ice"]
            candidate_str = ice_data.get("candidate", "")

            # Only accept UDP candidates with valid IPs (skip TCP, skip cloud relay UUIDs)
            if " UDP " in candidate_str.upper():
                parts = candidate_str.split()
                if len(parts) >= 5:
                    ip = parts[4]
                    # Check if it's a valid IP (not a UUID)
                    if "." in ip or ":" in ip:  # IPv4 or IPv6
                        try:
                            import ipaddress
                            ipaddress.ip_address(ip.split('%')[0])
                            # It's valid! Add it
                            candidate = self._parse_ice_candidate(
                                candidate_str,
                                ice_data.get("sdpMid"),
                                ice_data.get("sdpMLineIndex"),
                            )
                            if candidate:
                                await self._pc.addIceCandidate(candidate)
                                logger.info("Added UDP ICE candidate: %s", candidate_str)
                                return
                        except ValueError:
                            pass

            logger.debug("Ignoring non-UDP/invalid ICE candidate: %s", candidate_str[:60])
            return

        msg_type = data.get("type")
        logger.debug("Unhandled signaling message type=%s: %s", msg_type, str(data)[:200])

    # ── PeerConnection event handlers ─────────────────────────────────

    def _setup_pc_handlers(self) -> None:
        """Register callbacks on the RTCPeerConnection."""

        @self._pc.on("track")
        async def on_track(track):
            logger.info("Track received: kind=%s", track.kind)
            if track.kind == "video":
                asyncio.ensure_future(self._consume_video(track))

        @self._pc.on("datachannel")
        def on_datachannel(channel):
            logger.info("DataChannel opened: label=%s", channel.label)
            self._setup_datachannel(channel)

        @self._pc.on("connectionstatechange")
        async def on_state_change():
            if self._pc is None:
                return
            state = self._pc.connectionState
            logger.info("Connection state → %s", state)
            if state == "connected":
                # Request keyframe via WS — retry periodically
                asyncio.ensure_future(self._keyframe_request_loop())
            elif state in ("failed", "closed"):
                self._closed.set()

    async def _keyframe_request_loop(self):
        """Keep requesting keyframes via WS until video starts decoding."""
        for i in range(10):
            await asyncio.sleep(1 + i)
            if self._latest_frame is not None:
                logger.info("First frame decoded, stopping keyframe requests")
                return
            await self._request_keyframe_ws()

    def _setup_datachannel(self, channel) -> None:
        """Attach handlers to an incoming DataChannel."""

        @channel.on("message")
        def on_message(raw_data):
            try:
                metadata = json.loads(raw_data)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Non-JSON datachannel message (%d bytes)", len(raw_data)
                )
                return

            if self.on_metadata:
                self._fire_callback(self.on_metadata, metadata, self._latest_frame)
            else:
                logger.debug(
                    "Metadata received but no handler set: %s",
                    list(metadata.keys()),
                )

    def _fire_callback(self, cb, *args) -> None:
        """Invoke a callback that may be sync or async."""
        result = cb(*args)
        if asyncio.iscoroutine(result):
            asyncio.ensure_future(result)

    # ── video consumer ────────────────────────────────────────────────

    async def _consume_video(self, track) -> None:
        """Read frames from the video track and stash the latest."""
        from aiortc.mediastreams import MediaStreamError
        logger.info("Starting video consumer")
        frame_count = 0
        while True:
            try:
                frame: VideoFrame = await track.recv()
                frame_count += 1
                if frame_count % 30 == 1:
                    logger.info("Decoded frame #%d: %dx%d", frame_count, frame.width, frame.height)
                self._latest_frame = frame
                if self.on_frame:
                    self._fire_callback(self.on_frame, frame)
            except MediaStreamError:
                logger.info("Video track ended")
                break
            except Exception as e:
                # Decode errors — keep going, next IDR will fix it
                if frame_count == 0 and "no frame" not in str(e):
                    logger.warning("Video decode error (no frames yet): %s", e)

    # ── SDP / ICE helpers ─────────────────────────────────────────────

    @staticmethod
    def _force_sha256_in_sdp(sdp: str) -> str:
        """
        Rewrite any 'a=fingerprint:...' lines in the SDP to use
        sha-256. Leaves existing sha-256 lines untouched.
        """
        lines = []
        for line in sdp.splitlines():
            if line.startswith("a=fingerprint:") and "sha-256" not in line:
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    line = f"a=fingerprint:sha-256 {parts[1]}"
                    logger.debug("Rewrote SDP fingerprint line → sha-256")
            lines.append(line)
        return "\r\n".join(lines)

    @staticmethod
    def _force_setup_passive_in_sdp(sdp: str) -> str:
        """
        Rewrite 'a=setup:active' → 'a=setup:passive' in the SDP answer.

        Why: NX VMS (ice-lite) always initiates the SCTP association by
        sending InitChunk.  aiortc decides who is the SCTP initiator based
        on the DTLS role: DTLS-client → SCTP-client → also sends InitChunk.
        This creates an SCTP init collision that aiortc cannot resolve
        (it only handles incoming InitChunk when is_server=True).

        By making ourselves the DTLS *server* (setup:passive), aiortc sets
        is_server=True for SCTP, so it waits for the VMS's InitChunk and
        responds with InitAckChunk — no collision.
        """
        lines = []
        for line in sdp.splitlines():
            if line.strip() == "a=setup:active":
                line = "a=setup:passive"
                logger.info("SDP: rewrote a=setup:active → a=setup:passive (SCTP collision fix)")
            lines.append(line)
        return "\r\n".join(lines)

    @staticmethod
    def _discover_public_address(
        stun_host: str = "stun.l.google.com",
        stun_port: int = 19302,
        timeout: float = 3.0,
    ) -> tuple[str, int] | None:
        """
        Fallback: Send a STUN Binding Request from a fresh socket.
        The NAT port mapping will differ from aiortc's ICE socket, so
        this only works for endpoint-independent (full-cone) NATs.
        Prefer _discovered_public_from_ice when available.
        """
        import os
        txn_id = os.urandom(12)
        request = _struct.pack("!HHI", 0x0001, 0, 0x2112A442) + txn_id

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        try:
            sock.sendto(request, (stun_host, stun_port))
            data, _ = sock.recvfrom(1024)
        except (socket.timeout, OSError) as e:
            logger.warning("STUN discovery (fallback) failed: %s", e)
            return None
        finally:
            sock.close()

        return WebRTCClient._parse_stun_response(data)

    @staticmethod
    def _parse_stun_response(data: bytes) -> tuple[str, int] | None:
        """Parse a STUN Binding Success Response and extract the public address."""
        if len(data) < 20:
            return None

        resp_type = _struct.unpack_from("!H", data, 0)[0]
        if resp_type != 0x0101:
            return None

        magic = 0x2112A442
        pos = 20
        while pos + 4 <= len(data):
            attr_type, attr_len = _struct.unpack_from("!HH", data, pos)
            pos += 4

            if attr_type == 0x0020 and attr_len >= 8:
                family = data[pos + 1]
                if family == 0x01:
                    xport = _struct.unpack_from("!H", data, pos + 2)[0] ^ (magic >> 16)
                    xip_raw = _struct.unpack_from("!I", data, pos + 4)[0] ^ magic
                    ip = socket.inet_ntoa(_struct.pack("!I", xip_raw))
                    logger.info("STUN: parsed public address %s:%d (XOR-MAPPED)", ip, xport)
                    return (ip, xport)

            elif attr_type == 0x0001 and attr_len >= 8:
                family = data[pos + 1]
                if family == 0x01:
                    port = _struct.unpack_from("!H", data, pos + 2)[0]
                    ip_raw = _struct.unpack_from("!I", data, pos + 4)[0]
                    ip = socket.inet_ntoa(_struct.pack("!I", ip_raw))
                    logger.info("STUN: parsed public address %s:%d (MAPPED)", ip, port)
                    return (ip, port)

            pos += attr_len
            if attr_len % 4:
                pos += 4 - (attr_len % 4)

        return None

    async def _discover_public_from_ice_socket(self) -> tuple[str, int] | None:
        """
        Send a STUN Binding Request from aiortc's actual ICE socket,
        so the NAT mapping matches what the relay will use to reach us.

        Installs a temporary interceptor on the StunProtocol to capture
        the response without blocking the event loop or stealing packets.
        """
        try:
            # Dig into aiortc → aioice to find the ICE UDP socket
            # Try multiple paths for different aiortc versions
            ice_connection = None

            # Path 1: through transceivers
            for transceiver in self._pc.getTransceivers():
                for attr in ('_dtlsTransport', '_transport'):
                    dtls = getattr(transceiver, attr, None)
                    if dtls and hasattr(dtls, '_transport'):
                        ice = dtls._transport
                        if hasattr(ice, '_connection'):
                            ice_connection = ice._connection
                            break
                if ice_connection:
                    break

            # Path 2: RTCPeerConnection._dtlsTransports dict
            if not ice_connection:
                for attr in ('_dtlsTransports', '_sctp_transport'):
                    container = getattr(self._pc, attr, None)
                    if isinstance(container, dict):
                        for dtls in container.values():
                            ice = getattr(dtls, '_transport', None)
                            if ice and hasattr(ice, '_connection'):
                                ice_connection = ice._connection
                                break
                    elif container and hasattr(container, '_transport'):
                        ice = container._transport
                        if hasattr(ice, '_transport'):
                            ice2 = ice._transport
                            if hasattr(ice2, '_connection'):
                                ice_connection = ice2._connection
                    if ice_connection:
                        break

            if not ice_connection:
                # Log what we can see to help debug
                logger.warning(
                    "Cannot access ICE connection. PC attrs: %s",
                    [a for a in dir(self._pc) if 'ice' in a.lower() or 'dtls' in a.lower() or 'transport' in a.lower()]
                )
                for i, t in enumerate(self._pc.getTransceivers()):
                    logger.warning(
                        "  Transceiver[%d] attrs: %s",
                        i,
                        [a for a in dir(t) if 'transport' in a.lower() or 'dtls' in a.lower()]
                    )
                return None

            # Find the first StunProtocol
            protocol = None
            protocols = ice_connection._protocols
            if isinstance(protocols, dict):
                for proto in protocols.values():
                    protocol = proto
                    break
            elif isinstance(protocols, (list, tuple)):
                for proto in protocols:
                    protocol = proto
                    break
            else:
                logger.warning("Unknown _protocols type: %s", type(protocols))
                return None

            if not protocol or not protocol.transport:
                logger.warning("No ICE protocol/transport found")
                return None

            sock = protocol.transport.get_extra_info('socket')
            if not sock:
                logger.warning("Cannot get socket from ICE transport")
                return None

            local_addr = sock.getsockname()
            logger.info("ICE socket local address: %s", local_addr)

            # Resolve STUN server
            stun_info = socket.getaddrinfo(
                "stun.l.google.com", 19302, socket.AF_INET
            )[0][4]
            stun_ip = stun_info[0]

            # Build STUN Binding Request
            import os
            txn_id = os.urandom(12)
            request = _struct.pack("!HHI", 0x0001, 0, 0x2112A442) + txn_id

            # ── Intercept the response via a temporary protocol patch ──
            loop = asyncio.get_event_loop()
            result_future = loop.create_future()
            orig_datagram_received = protocol.datagram_received

            def _intercepting_datagram_received(data, addr):
                # Capture STUN Binding Success from the STUN server
                if (
                    addr[0] == stun_ip
                    and len(data) >= 20
                    and _struct.unpack_from("!H", data, 0)[0] == 0x0101
                ):
                    # Check transaction ID matches
                    if data[8:20] == txn_id:
                        parsed = WebRTCClient._parse_stun_response(data)
                        if parsed and not result_future.done():
                            result_future.set_result(parsed)
                        return  # consumed — don't pass to aioice
                # Everything else goes to the original handler
                return orig_datagram_received(data, addr)

            protocol.datagram_received = _intercepting_datagram_received

            # Send the request from the ICE socket
            sock.sendto(request, stun_info)
            logger.info("Sent STUN binding from ICE socket %s → %s", local_addr, stun_info)

            try:
                result = await asyncio.wait_for(result_future, timeout=3.0)
                logger.info(
                    "STUN: discovered public address via ICE socket: %s:%d",
                    result[0], result[1],
                )
                return result
            except asyncio.TimeoutError:
                logger.warning("STUN: no response on ICE socket within 3s")
                return None
            finally:
                # Always restore the original handler
                protocol.datagram_received = orig_datagram_received

        except Exception as e:
            logger.warning("STUN discovery via ICE socket failed: %s", e)
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def _inject_srflx_candidate(sdp: str, public_ip: str, public_port: int) -> str:
        """
        Inject a synthetic server-reflexive candidate into each m= section
        of the local SDP. Uses the first host candidate's foundation and
        component, with a slightly lower priority.
        """
        lines = sdp.splitlines()
        result = []
        for line in lines:
            result.append(line)
            # After each host candidate line, inject our srflx
            if line.startswith("a=candidate:") and "typ host" in line:
                parts = line.split()
                # a=candidate:<foundation> <component> <proto> <priority> <ip> <port> typ host
                if len(parts) >= 8:
                    foundation = parts[0].split(":")[1]
                    component = parts[1]
                    proto = parts[2]
                    # host candidate IP (raddr/rport for the srflx)
                    host_ip = parts[4]
                    host_port = parts[5]
                    # Slightly lower priority than host (typ srflx)
                    srflx_priority = 1694498815  # standard srflx priority
                    srflx = (
                        f"a=candidate:srflx{foundation} {component} {proto} "
                        f"{srflx_priority} {public_ip} {public_port} "
                        f"typ srflx raddr {host_ip} rport {host_port}"
                    )
                    result.append(srflx)
                    logger.info("Injected srflx candidate: %s", srflx)
        return "\r\n".join(result)

    async def _trickle_local_candidates(self, sdp: str) -> None:
        """
        Extract candidate lines from the local SDP and send each one
        to the server as a trickle ICE message over the WebSocket.

        NX VMS cloud relay processes trickle ICE candidates from the
        signaling channel. This tells the relay our addresses (including
        the srflx) so it can set up UDP forwarding to reach us.
        """
        if not self._ws or self._ws.closed:
            return

        # Determine the current sdpMid from the SDP — each m= section
        current_mid = None
        mline_index = -1

        for line in sdp.splitlines():
            if line.startswith("m="):
                mline_index += 1
            elif line.startswith("a=mid:"):
                current_mid = line.split(":", 1)[1].strip()
            elif line.startswith("a=candidate:"):
                candidate_str = line[2:]  # strip "a="
                ice_msg = {
                    "ice": {
                        "candidate": candidate_str,
                        "sdpMid": current_mid or str(mline_index),
                        "sdpMLineIndex": mline_index,
                    }
                }
                try:
                    await self._ws.send_json(ice_msg)
                    logger.info("Trickled ICE candidate: %s (mid=%s)",
                                candidate_str[:80], current_mid)
                except Exception as e:
                    logger.warning("Failed to trickle candidate: %s", e)

        # Signal end of candidates
        try:
            await self._ws.send_json({
                "ice": {
                    "candidate": "",
                    "sdpMid": "0",
                    "sdpMLineIndex": 0,
                }
            })
            logger.info("Sent end-of-candidates trickle")
        except Exception as e:
            logger.warning("Failed to send end-of-candidates: %s", e)

    @staticmethod
    def _parse_ice_candidate(candidate_str, sdp_mid, sdp_mline_index):
        """Parse a candidate string into an object aiortc can consume."""
        from aiortc.sdp import candidate_from_sdp

        try:
            candidate = candidate_from_sdp(candidate_str)
            candidate.sdpMid = sdp_mid
            candidate.sdpMLineIndex = sdp_mline_index
            return candidate
        except Exception:
            logger.warning("Failed to parse ICE candidate: %s", candidate_str)
            return None

    async def _request_keyframe_ws(self):
        """Request keyframe via WebSocket signaling (NX VMS protocol)."""
        if self._ws and not self._ws.closed:
            # Try the most common NX VMS WebSocket keyframe request formats
            for msg in [
                {"type": "keyframe"},
                {"type": "configure", "keyframe": True},
                {"keyframe": True},
            ]:
                try:
                    await self._ws.send_json(msg)
                    logger.info("Sent WS keyframe request: %s", msg)
                    return
                except Exception as e:
                    logger.warning("WS keyframe request failed: %s", e)
        """Send RTCP PLI to request a keyframe from the server."""
        import struct
        for transceiver in self._pc.getTransceivers():
            if transceiver.kind != "video" or not transceiver.receiver:
                continue
            receiver = transceiver.receiver
            # Get remote SSRC from the receiver's internal codec state
            remote_ssrc = getattr(receiver, '_ssrc', None)
            if not remote_ssrc and hasattr(receiver, '_track'):
                remote_ssrc = getattr(receiver._track, '_ssrc', None)
            if not remote_ssrc:
                # Fall back to parsing it from the stored offer SDP
                remote_ssrc = self._extract_video_ssrc()
            if not remote_ssrc:
                logger.warning("Cannot send PLI: no remote SSRC found")
                return

            local_ssrc = 0
            # RTCP PSFB PLI: V=2, P=0, FMT=1, PT=206, length=2
            pli = struct.pack('!BBH II', 0x81, 206, 2, local_ssrc, remote_ssrc)

            # Send via the DTLS transport
            dtls = getattr(receiver, '_dtls_transport', None)
            if dtls and hasattr(dtls, '_send_rtp'):
                await dtls._send_rtp(pli)
                logger.info("Sent PLI for SSRC %d", remote_ssrc)
            elif dtls and hasattr(dtls, 'transport'):
                # Try via ICE transport
                ice = dtls.transport
                if hasattr(ice, '_send'):
                    await ice._send(pli)
                    logger.info("Sent PLI via ICE for SSRC %d", remote_ssrc)

    def _extract_video_ssrc(self):
        """Extract video SSRC from stored offer SDP."""
        if not hasattr(self, '_offer_sdp') or not self._offer_sdp:
            return None
        in_video = False
        for line in self._offer_sdp.splitlines():
            if line.startswith('m=video'):
                in_video = True
            elif line.startswith('m='):
                in_video = False
            if in_video and line.startswith('a=ssrc:'):
                try:
                    return int(line.split(':')[1].split()[0])
                except (ValueError, IndexError):
                    pass
        return None
