import urllib.parse
from threading import Thread

from huggingface_hub.cli.cache import verify
from websocket import create_connection, WebSocketBadStatusException
import ssl

WS_PATH = "/jsonrpc"


def _concat_url(server_url, path):
    initial_url = urllib.parse.urlparse(server_url)
    result = str(urllib.parse.urlunparse(initial_url._replace(path=path, scheme='wss')))
    return result


class WSClass:

  def __init__(self, on_message, server_url):
    self.on_message_callback = on_message
    self.ws = create_connection(_concat_url(server_url=server_url, path=WS_PATH), sslopt={"cert_reqs": ssl.CERT_NONE})

    self.listen_thread = Thread(target=self.listen)
    self.listen_thread.start()

  def listen(self):
    while True:
      print('receiving')
      raw_message = self.ws.recv()
      print("received")
      print(raw_message)
      self.on_message_callback(raw_message)

  def send(self, message):
    print('sending')
    print(message)
    self.ws.send(message)
    print('sent', message)
