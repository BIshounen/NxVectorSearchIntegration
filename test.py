import asyncio
from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer

async def test():
    pc = RTCPeerConnection(configuration=RTCConfiguration(
        iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")]
    ))
    pc.addTransceiver("audio", direction="recvonly")
    await pc.setLocalDescription(await pc.createOffer())
    await asyncio.sleep(5)
    for line in pc.localDescription.sdp.split("\r\n"):
        if "candidate" in line:
            print(line)
    await pc.close()

asyncio.run(test())