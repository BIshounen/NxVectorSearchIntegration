"""
aiortc compatibility patches for NX VMS WebRTC streams.

Patches applied:
  1. SHA-256-only fingerprint  (VMS rejects sha-384/sha-512)
  2. Mixed RSA+ECDSA ciphers   (VMS has RSA cert; aiortc only offers ECDSA)
  3. H264 SPS/PPS extradata + auto-reset on decode failures
  4. PLI + FIR keyframe requests (best effort)
  5. SCTP stream count clamp   (OS=1, MIS=1024 like Chrome)
  6. SCTP packet-level tracing
  7. RTP IDR fragment diagnostics
  8. Jitter buffer capacity increase (128→512) — gives large IDR frames
     room to fully assemble, and gives NACK retransmissions time to arrive
  9. NACK diagnostics — logs when NACKs are sent so we can verify
     retransmissions are working

Usage: import BEFORE any aiortc usage:
    import aiortc_rsa_patch
"""

import asyncio
import base64
import logging
import random as _random
import struct
import time as _time_module

from OpenSSL import SSL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patch 1: SHA-256-only fingerprint
# ---------------------------------------------------------------------------
from aiortc.rtcdtlstransport import (
    RTCCertificate,
    RTCDtlsFingerprint,
    certificate_digest,
)


def _patched_getFingerprints(self):
    return [
        RTCDtlsFingerprint(
            algorithm="sha-256",
            value=certificate_digest(self._cert, "sha-256"),
        )
    ]


RTCCertificate.getFingerprints = _patched_getFingerprints

# ---------------------------------------------------------------------------
# Patch 2: Mixed RSA + ECDSA cipher suites
# ---------------------------------------------------------------------------
_MIXED_CIPHERS = (
    b"ECDHE-RSA-AES128-GCM-SHA256:"
    b"ECDHE-RSA-CHACHA20-POLY1305:"
    b"ECDHE-RSA-AES128-SHA:"
    b"ECDHE-RSA-AES256-SHA:"
    b"ECDHE-ECDSA-AES128-GCM-SHA256:"
    b"ECDHE-ECDSA-CHACHA20-POLY1305:"
    b"ECDHE-ECDSA-AES128-SHA:"
    b"ECDHE-ECDSA-AES256-SHA"
)


def _patched_create_ssl_context(self, srtp_profiles):
    ctx = SSL.Context(SSL.DTLS_METHOD)
    ctx.set_verify(
        SSL.VERIFY_PEER | SSL.VERIFY_FAIL_IF_NO_PEER_CERT, lambda *args: True
    )
    ctx.use_certificate(self._cert)
    ctx.use_privatekey(self._key)
    ctx.set_cipher_list(_MIXED_CIPHERS)
    ctx.set_tlsext_use_srtp(b":".join(x.openssl_profile for x in srtp_profiles))
    return ctx


RTCCertificate._create_ssl_context = _patched_create_ssl_context

# ---------------------------------------------------------------------------
# Patch 3: H264 SPS/PPS extradata + auto-reset
# ---------------------------------------------------------------------------
import av
from av.frame import Frame
from av.packet import Packet

from aiortc.codecs.h264 import H264Decoder as _OrigH264Decoder
from aiortc.jitterbuffer import JitterBuffer, JitterFrame
from aiortc.mediastreams import VIDEO_TIME_BASE
from aiortc.rtcrtpparameters import RTCRtpCodecParameters
import aiortc.codecs as _codecs
import aiortc.rtcrtpreceiver as _rtcrtpreceiver

ANNEX_B_START_CODE = bytes([0, 0, 0, 1])


def _parse_sprop_parameter_sets(sprop: str) -> bytes:
    extradata = b""
    for nal_b64 in sprop.split(","):
        nal_b64 = nal_b64.strip()
        if not nal_b64:
            continue
        try:
            nal_bytes = base64.b64decode(nal_b64)
            if nal_bytes:
                nal_type = nal_bytes[0] & 0x1F
                logger.info("sprop NAL type=%d  size=%d bytes", nal_type, len(nal_bytes))
                extradata += ANNEX_B_START_CODE + nal_bytes
        except Exception as e:
            logger.warning("Failed to decode sprop NAL %r: %s", nal_b64, e)
    return extradata


class H264DecoderPatched(_OrigH264Decoder):
    _MAX_FAILURES_BEFORE_RESET = 30

    def __init__(self, codec_params: RTCRtpCodecParameters = None) -> None:
        self.codec = av.CodecContext.create("h264", "r")
        self._extradata = b""
        self._first_frame = True
        self._consecutive_failures = 0
        self._total_decoded = 0

        if codec_params is None:
            return

        params = codec_params.parameters or {}
        sprop = (
            params.get("sprop-parameter-sets")
            or params.get(" sprop-parameter-sets")
            or ""
        )
        if not sprop:
            logger.warning("H264DecoderPatched: no sprop-parameter-sets")
            return

        self._extradata = _parse_sprop_parameter_sets(sprop)
        if self._extradata:
            self.codec.extradata = self._extradata
            logger.info("H264DecoderPatched: set extradata (%d bytes)", len(self._extradata))

    def _reset_codec(self) -> None:
        try:
            self.codec = av.CodecContext.create("h264", "r")
            if self._extradata:
                self.codec.extradata = self._extradata
            self._first_frame = True
            logger.info("H264DecoderPatched: decoder RESET (decoded so far: %d)",
                        self._total_decoded)
        except Exception as e:
            logger.error("H264DecoderPatched: reset failed: %s", e)

    @staticmethod
    def _ensure_annexb(data: bytes) -> bytes:
        SC = b"\x00\x00\x00\x01"
        if data[:4] == SC or data[:3] == b"\x00\x00\x01":
            return data
        if len(data) >= 4:
            maybe_len = int.from_bytes(data[:4], "big")
            if 0 < maybe_len <= len(data) - 4:
                out = bytearray()
                pos = 0
                valid = True
                while pos + 4 <= len(data):
                    nal_len = int.from_bytes(data[pos:pos + 4], "big")
                    pos += 4
                    if nal_len <= 0 or pos + nal_len > len(data):
                        valid = False
                        break
                    out += SC + data[pos:pos + nal_len]
                    pos += nal_len
                if valid and out:
                    return bytes(out)
        if data and (data[0] & 0x80):
            return b""
        return SC + data

    def decode(self, encoded_frame: JitterFrame) -> list[Frame]:
        try:
            data = bytes(encoded_frame.data)
            SC = b"\x00\x00\x00\x01"
            data = self._ensure_annexb(data)
            if not data:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._MAX_FAILURES_BEFORE_RESET:
                    self._reset_codec()
                    self._consecutive_failures = 0
                return []

            # Identify NAL types
            nal_types = []
            for i in range(len(data) - 4):
                if data[i:i + 4] == SC:
                    nal_types.append(data[i + 4] & 0x1F)

            has_idr = 5 in nal_types

            # Log when we see an IDR arrive at the decoder
            if has_idr:
                logger.info("H264DecoderPatched: IDR frame reached decoder! "
                            "(%d bytes, NALs: %s)", len(data), nal_types)

            # Log periodically when still no frames decoded
            if self._total_decoded == 0 and not has_idr:
                if self._consecutive_failures % 60 == 0:
                    logger.info("H264DecoderPatched: still waiting for IDR "
                                "(failures: %d, NALs: %s)", self._consecutive_failures, nal_types)

            # Prepend SPS/PPS to IDR frames that lack them
            if self._extradata and has_idr and 7 not in nal_types:
                data = self._extradata + data
                self._first_frame = False
                logger.info("H264DecoderPatched: prepended SPS/PPS to IDR")

            packet = Packet(data)
            packet.pts = encoded_frame.timestamp
            packet.time_base = VIDEO_TIME_BASE
            frames = list(self.codec.decode(packet))

            if frames:
                self._consecutive_failures = 0
                self._total_decoded += len(frames)
                if self._total_decoded <= 5 or self._total_decoded % 100 == 0:
                    logger.info("H264DecoderPatched: *** DECODED FRAME *** (total: %d)",
                                self._total_decoded)
                return frames
            else:
                self._consecutive_failures += 1

        except av.FFmpegError as e:
            self._consecutive_failures += 1
            if self._consecutive_failures <= 5 or self._consecutive_failures % 50 == 0:
                logger.warning("H264DecoderPatched: decode failed (#%d): %s",
                               self._consecutive_failures, e)

        if self._consecutive_failures >= self._MAX_FAILURES_BEFORE_RESET:
            self._reset_codec()
            self._consecutive_failures = 0

        return []


_original_get_decoder = _codecs.get_decoder
_sprop_cache: dict[int, str] = {}


def _cache_sprop_from_pc(pc) -> None:
    try:
        rd = pc.remoteDescription
        if rd is None:
            return
        from aiortc.sdp import SessionDescription
        sdp = SessionDescription.parse(rd.sdp)
        for media in sdp.media:
            for codec in media.rtp.codecs:
                sprop = (codec.parameters.get("sprop-parameter-sets")
                         or codec.parameters.get(" sprop-parameter-sets") or "")
                if sprop:
                    _sprop_cache[codec.payloadType] = sprop
    except Exception as e:
        logger.warning("Failed to cache sprop: %s", e)


def _patched_get_decoder(codec: RTCRtpCodecParameters):
    if codec.mimeType.lower() == "video/h264":
        sprop = (codec.parameters.get("sprop-parameter-sets")
                 or codec.parameters.get(" sprop-parameter-sets")
                 or _sprop_cache.get(codec.payloadType, ""))
        if sprop:
            import copy
            codec = copy.deepcopy(codec)
            codec.parameters["sprop-parameter-sets"] = sprop
        return H264DecoderPatched(codec)
    return _original_get_decoder(codec)


_codecs.get_decoder = _patched_get_decoder
_rtcrtpreceiver.get_decoder = _patched_get_decoder


def _install_sdp_hook():
    import aiortc.rtcpeerconnection as _rtcpc
    _orig = _rtcpc.RTCPeerConnection.setRemoteDescription

    async def _hooked(self, description):
        result = await _orig(self, description)
        _cache_sprop_from_pc(self)
        return result

    _rtcpc.RTCPeerConnection.setRemoteDescription = _hooked


_install_sdp_hook()

# ---------------------------------------------------------------------------
# Patch 4: PLI + FIR keyframe requests
# ---------------------------------------------------------------------------
from aiortc.rtcrtpreceiver import RTCRtpReceiver
from aiortc.rtp import RtcpRrPacket, RtcpReceiverInfo

_original_handle_rtp = RTCRtpReceiver._handle_rtp_packet
_orig_run_rtcp = RTCRtpReceiver._run_rtcp
_fir_seq = [0]


async def _send_fir(receiver, media_ssrc):
    _fir_seq[0] = (_fir_seq[0] + 1) % 256
    rtcp_ssrc = getattr(receiver, '_RTCRtpReceiver__rtcp_ssrc', None) or 0
    header = struct.pack('!BBH II', 0x80 | 4, 206, 4, rtcp_ssrc, 0)
    fci = struct.pack('!I BBH', media_ssrc, _fir_seq[0], 0, 0)
    try:
        await receiver._send_rtcp(header + fci)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Patch 8: Jitter buffer capacity increase (128 → 512)
#
# Root cause: IDR frames at 2048x1536 span ~90 RTP packets.  With the
# default capacity of 128, the buffer has very little headroom.  If even
# a few packets need NACK retransmission, the buffer fills up and evicts
# older packets before the IDR frame can be fully assembled.
#
# Also: when data volume increases (more analytics, more P-frames),
# the buffer fills faster and leaves less room for IDR assembly.
#
# Fix: increase to 512 (must be power of 2).  This gives ~5x more
# headroom for retransmissions and bursty traffic.
# ---------------------------------------------------------------------------
from aiortc.rtcrtpreceiver import NackGenerator

_original_receiver_init = RTCRtpReceiver.__init__

_JITTER_CAPACITY = 512  # was 128

_nack_send_count = [0]


def _patched_receiver_init(self, kind, transport):
    _original_receiver_init(self, kind, transport)
    if kind == "video":
        # Replace the jitter buffer with a larger one
        old_cap = self._RTCRtpReceiver__jitter_buffer._capacity
        self._RTCRtpReceiver__jitter_buffer = JitterBuffer(
            capacity=_JITTER_CAPACITY, is_video=True
        )
        logger.info("JitterBuffer capacity: %d → %d", old_cap, _JITTER_CAPACITY)


RTCRtpReceiver.__init__ = _patched_receiver_init

# ---------------------------------------------------------------------------
# Patch 9: NACK send diagnostics + aggressive keyframe requests
# ---------------------------------------------------------------------------
_orig_send_nack = RTCRtpReceiver._send_rtcp_nack


async def _patched_send_nack(self, media_ssrc, lost):
    _nack_send_count[0] += 1
    n = _nack_send_count[0]
    if n <= 10 or n % 50 == 0:
        logger.info("NACK #%d: requesting retransmit of %d packets for SSRC %d "
                     "(seqs: %s)", n, len(lost), media_ssrc,
                     list(lost)[:10] if len(lost) > 10 else list(lost))
    return await _orig_send_nack(self, media_ssrc, lost)


RTCRtpReceiver._send_rtcp_nack = _patched_send_nack


async def _patched_handle_rtp_packet(self, packet, arrival_time_ms):
    result = await _original_handle_rtp(self, packet, arrival_time_ms)
    if getattr(self, "_RTCRtpReceiver__kind", None) == "video":
        self._rtp_count = getattr(self, "_rtp_count", 0) + 1
        n = self._rtp_count
        if (n == 1 or n % 100 == 0) and not getattr(self, "_first_frame_decoded", False):
            try:
                await self._send_rtcp_pli(packet.ssrc)
            except Exception:
                pass
            try:
                await _send_fir(self, packet.ssrc)
            except Exception:
                pass
    return result


async def _fast_run_rtcp(self):
    self._RTCRtpReceiver__log_debug("- RTCP started")
    self._RTCRtpReceiver__rtcp_started.set()
    await asyncio.sleep(0.05)
    try:
        first = True
        while True:
            await asyncio.sleep(0.0 if first else 0.5 + _random.random())
            first = False
            rtcp_ssrc = self._RTCRtpReceiver__rtcp_ssrc
            remote_streams = self._RTCRtpReceiver__remote_streams
            lsr_dict = self._RTCRtpReceiver__lsr
            lsr_time_dict = self._RTCRtpReceiver__lsr_time
            reports = []
            for ssrc, stream in remote_streams.items():
                if stream.max_seq is None:
                    continue
                lsr = lsr_dict.get(ssrc, 0)
                dlsr = 0
                if ssrc in lsr_time_dict:
                    delay = _time_module.time() - lsr_time_dict[ssrc]
                    if 0 < delay < 65536:
                        dlsr = int(delay * 65536)
                reports.append(RtcpReceiverInfo(
                    ssrc=ssrc, fraction_lost=stream.fraction_lost,
                    packets_lost=stream.packets_lost,
                    highest_sequence=stream.max_seq,
                    jitter=stream.jitter, lsr=lsr, dlsr=dlsr,
                ))
            if rtcp_ssrc is not None and reports:
                rr = RtcpRrPacket(ssrc=rtcp_ssrc, reports=reports)
                await self._send_rtcp(rr)
    except asyncio.CancelledError:
        pass
    self._RTCRtpReceiver__log_debug("- RTCP finished")
    self._RTCRtpReceiver__rtcp_exited.set()


RTCRtpReceiver._handle_rtp_packet = _patched_handle_rtp_packet
RTCRtpReceiver._run_rtcp = _fast_run_rtcp

# ---------------------------------------------------------------------------
# Patch 5: SCTP stream count
# ---------------------------------------------------------------------------
from aiortc.rtcsctptransport import RTCSctpTransport as _RTCSctpTransport

_original_sctp_init = _RTCSctpTransport.__init__


def _patched_sctp_init(self, transport, port=5000):
    _original_sctp_init(self, transport, port)
    self._outbound_streams_count = 1
    self._inbound_streams_max = 1024


_RTCSctpTransport.__init__ = _patched_sctp_init

# ---------------------------------------------------------------------------
# Patch 6: SCTP tracing (minimal)
# ---------------------------------------------------------------------------
from aiortc.rtcsctptransport import (
    InitChunk, InitAckChunk, CookieEchoChunk, CookieAckChunk,
    DataChunk, SackChunk, AbortChunk, ErrorChunk, HeartbeatChunk,
    HeartbeatAckChunk, ShutdownChunk,
)

_orig_send_chunk = _RTCSctpTransport._send_chunk
_orig_receive_chunk = _RTCSctpTransport._receive_chunk
_sctp_send_count = [0]
_sctp_recv_count = [0]


def _chunk_name(chunk):
    for cls in (InitChunk, InitAckChunk, CookieEchoChunk, CookieAckChunk,
                DataChunk, SackChunk, AbortChunk, ErrorChunk,
                HeartbeatChunk, HeartbeatAckChunk, ShutdownChunk):
        if isinstance(chunk, cls):
            return cls.__name__
    return type(chunk).__name__


async def _traced_send_chunk(self, chunk):
    _sctp_send_count[0] += 1
    n = _sctp_send_count[0]
    name = _chunk_name(chunk)
    if n <= 20 or name not in ("SackChunk", "DataChunk"):
        print(f"[SCTP-TX #{n}] {name}", flush=True)
    return await _orig_send_chunk(self, chunk)


async def _traced_receive_chunk(self, chunk):
    _sctp_recv_count[0] += 1
    n = _sctp_recv_count[0]
    name = _chunk_name(chunk)
    extra = ""
    if isinstance(chunk, CookieAckChunk):
        extra = " *** HANDSHAKE COMPLETE ***"
    elif isinstance(chunk, AbortChunk):
        extra = " *** ABORTED ***"
    if n <= 50 or name not in ("SackChunk", "DataChunk"):
        print(f"[SCTP-RX #{n}] {name}{extra}", flush=True)
    return await _orig_receive_chunk(self, chunk)


_RTCSctpTransport._send_chunk = _traced_send_chunk
_RTCSctpTransport._receive_chunk = _traced_receive_chunk

# ---------------------------------------------------------------------------
# Patch 7: RTP IDR fragment diagnostics
# ---------------------------------------------------------------------------
from aiortc.codecs.h264 import H264PayloadDescriptor

_orig_h264_parse = H264PayloadDescriptor.parse
_idr_rtp_count = [0]
_fu_a_start_count = [0]


@classmethod
def _patched_h264_parse(cls, data):
    if data:
        nal_type = data[0] & 0x1F
        if nal_type == 28 and len(data) >= 2:
            fu_type = data[1] & 0x1F
            fu_start = (data[1] & 0x80) != 0
            if fu_type == 5:
                _idr_rtp_count[0] += 1
                if fu_start:
                    _fu_a_start_count[0] += 1
                    logger.info("RTP: IDR FU-A START #%d (total IDR pkts: %d)",
                                _fu_a_start_count[0], _idr_rtp_count[0])
        elif nal_type == 5:
            logger.info("RTP: Single-packet IDR!")
    return _orig_h264_parse.__func__(cls, data)


H264PayloadDescriptor.parse = _patched_h264_parse

# ---------------------------------------------------------------------------
print("[PATCH] aiortc_rsa_patch: sha-256 + RSA + H264 + PLI/FIR + SCTP"
      f" + JitterBuffer({_JITTER_CAPACITY}) + NACK diag", flush=True)