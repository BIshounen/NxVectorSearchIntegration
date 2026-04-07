import urllib.parse
import json
import uuid
from enum import verify
from threading import Thread
from AnalyticsAPIInterface import AnalyticsAPIInterface
import ssl

import websocket
from websocket import create_connection, WebSocketBadStatusException

WS_PATH = "/jsonrpc"

METHOD_CREATE_SESSION = "rest.v4.login.sessions.create"
METHOD_SUBSCRIBE_USERS = "rest.v4.users.subscribe"
METHOD_UPDATE_USERS = "rest.v4.users.update"
METHOD_SUBSCRIBE_ANALYTICS = 'rest.v4.analytics.subscribe'
METHOD_CREATE_DEVICE_AGENT = 'rest.v4.analytics.engines.deviceAgents.create'
METHOD_DELETE_DEVICE_AGENT = 'rest.v4.analytics.engines.deviceAgents.delete'
METHOD_CREATE_DEVICE_AGENT_MANIFEST = 'rest.v4.analytics.engines.deviceAgents.manifest.create'
METHOD_GET_INTEGRATION_ENGINE_SIDE_SETTINGS = 'rest.v4.analytics.engines.integrationSideSettings.get'
METHOD_GET_INTEGRATION_DEVICE_AGENT_SIDE_SETTINGS = 'rest.v4.analytics.engines.deviceAgents.integrationSideSettings.get'
METHOD_UPDATE_DEVICE_AGENT_SETTINGS = 'rest.v4.analytics.engines.deviceAgents.settings.update'
METHOD_NOTIFY_AGENT_ACTIVE_SETTINGS_CHANGE = 'rest.v4.analytics.engines.deviceAgents.settings.notifyActiveSettingChanged'
METHOD_UPDATE_ENGINE_SETTINGS = 'rest.v4.analytics.engines.settings.update'
METHOD_NOTIFY_ENGINE_ACTIVE_SETTINGS_CHANGE = 'rest.v4.analytics.engines.settings.notifyActiveSettingChanged'
METHOD_CREATE_OBJECT_METADATA = "rest.v4.analytics.engines.deviceAgents.metadata.object.create"
METHOD_CREATE_BEST_SHOT_METHOD = "rest.v4.analytics.engines.deviceAgents.metadata.bestShot.create"
METHOD_CREATE_TITLE_IMAGE_METHOD = "rest.v4.analytics.engines.deviceAgents.metadata.title.create"


def _concat_url(server_url, path):
  initial_url = urllib.parse.urlparse(server_url)
  result = str(urllib.parse.urlunparse(initial_url._replace(path=path, scheme='wss')))
  return result


class RequestAwaitable:

  def __init__(self):
    self.respond = None

  def __await__(self):
    while self.respond is None:
      yield

    return self.respond


class NxJSONRPC:

  def __init__(self, server_url: str, integration: AnalyticsAPIInterface):

    self.requests_queue: dict[str, RequestAwaitable] = dict()
    self.server_url = server_url
    self.integration = integration
    self.ws = create_connection(_concat_url(server_url=server_url, path=WS_PATH), sslopt={"cert_reqs": ssl.CERT_NONE})
    self.listen_thread = Thread(target=self.listen)
    self.listen_thread.start()

  def on_ws_message(self, raw_message):
    message = json.loads(raw_message)
    if 'id' in message:
      if 'result' in message:
        self.parse_response(message)
      else:
        self.parse_request(message)
    else:
      self.parse_notification(message)

  def parse_response(self, message: dict):
    if message['id'] in self.requests_queue:
      self.requests_queue[message['id']].respond = message['result']
      self.requests_queue.pop(message['id'])


  def parse_request(self, message: dict):
    if 'method' not in message:
      return

    if message['method'] == METHOD_CREATE_DEVICE_AGENT:
      self.react_on_device_agent_creation(message)
    if message['method'] == METHOD_GET_INTEGRATION_ENGINE_SIDE_SETTINGS:
      self.react_on_integration_engine_side_settings(message)
    if message['method'] == METHOD_GET_INTEGRATION_DEVICE_AGENT_SIDE_SETTINGS:
      self.react_on_integration_device_agent_side_settings(message)
    if message['method'] == METHOD_UPDATE_DEVICE_AGENT_SETTINGS:
      self.react_on_agent_settings_update(message)
    if message['method'] == METHOD_NOTIFY_AGENT_ACTIVE_SETTINGS_CHANGE:
      self.react_on_agent_active_settings(message)
    if message['method'] == METHOD_UPDATE_ENGINE_SETTINGS:
      self.react_on_engine_settings_update(message)
    if message['method'] == METHOD_NOTIFY_ENGINE_ACTIVE_SETTINGS_CHANGE:
      self.react_on_engine_active_settings(message)

  def parse_notification(self, message: dict):
    if 'method' in message:
      if message['method'] == METHOD_UPDATE_USERS:
        self.integration.set_parameters(message['params'])
      if message['method'] == METHOD_DELETE_DEVICE_AGENT:
        self.react_on_device_agent_deletion(message)

  def listen(self):
    while True:
      print('listening')
      raw_message = self.ws.recv()
      print("received:")
      print(raw_message)
      self.on_ws_message(raw_message)


  @staticmethod
  def compose_request(message: str|dict|list, method: str, message_id: str):
    message_dict = {
      'id': message_id,
      'params': message,
      'method': method,
      'jsonrpc': '2.0'
    }

    return json.dumps(message_dict)

  @staticmethod
  def compose_respond(message: str|dict|list, message_id: str):
    message_dict = {
      'id': message_id,
      'result': message,
      'jsonrpc': '2.0'
    }

    return json.dumps(message_dict)

  @staticmethod
  def compose_notification(message: str|dict|list, method: str):
    message_dict = {
      'method': method,
      'params': message,
      'jsonrpc': '2.0'
    }

    return json.dumps(message_dict)

  async def make_request(self, message: str|dict|list, method: str):
    message_id = str(uuid.uuid4())
    message_string = self.compose_request(message=message, method=method, message_id=message_id)
    request = RequestAwaitable()
    self.requests_queue[message_id] = request
    self.send_message(message_string=message_string)
    return await request

  def send_message(self, message_string):
    print("sent: ", message_string)
    self.ws.send(message_string)

  def notify(self, message, method):
    notification = self.compose_notification(message=message, method=method)
    self.send_message(notification)

  def respond(self, message, message_id):
    respond = self.compose_respond(message_id=message_id, message=message)
    self.send_message(respond)

  async def authorize(self, credentials: dict):
    message = credentials
    message['setSession'] = True
    await self.make_request(method=METHOD_CREATE_SESSION, message=message)
    print("authorized")

  async def subscribe_on_users(self, credentials: dict):
    message = {"name": credentials['username']}
    parameters = await self.make_request(method=METHOD_SUBSCRIBE_USERS, message=message)
    self.integration.set_parameters(parameters[0])

  async def subscribe_to_analytics(self, integration_id: str):
    message = {"id": integration_id}
    await self.make_request(method=METHOD_SUBSCRIBE_ANALYTICS, message=message)

  def react_on_device_agent_creation(self, message):
    device_parameters = message['params']
    manifest = self.integration.get_device_agent_manifest(device_parameters)
    respond = {
      'type': 'ok',
      'data': {
        'deviceAgentManifest': manifest
      }
    }
    self.respond(message=respond, message_id=message['id'])
    self.integration.on_device_agent_created(device_parameters=device_parameters)

  def react_on_device_agent_deletion(self, message):
    device_id = message['params']['target']['deviceId']
    self.integration.on_device_agent_deletion(device_id=device_id)

  def react_on_integration_engine_side_settings(self, message):
    parameters = message['params']['parameters']
    settings = self.integration.get_integration_engine_side_settings(parameters)
    self.respond(message=settings, message_id=message['id'])

  def react_on_integration_device_agent_side_settings(self, message):
    parameters = message['params']['parameters']
    device_id = message['params']['target']['deviceId']
    settings = self.integration.get_integration_device_agent_side_settings(parameters, device_id)
    self.respond(message=settings, message_id=message['id'])

  def react_on_agent_settings_update(self, message):
    parameters = message['params']['parameters']
    device_id = message['params']['target']['deviceId'].strip('{}')
    data = self.integration.on_agent_settings_update(parameters, device_id)
    respond = {
      'type': 'ok',
      'data': data
    }
    self.respond(message=respond, message_id=message['id'])

  def react_on_agent_active_settings(self, message):
    parameters = message['params']['parameters']
    device_id = message['target']['deviceId']
    data = self.integration.on_agent_active_settings_change(parameters, device_id)
    respond = {
      'type': 'ok',
      'data': data
    }
    self.respond(message=respond, message_id=message['id'])

  def react_on_engine_settings_update(self, message):
    parameters = message['params']['parameters']
    data = self.integration.on_engine_settings_update(parameters)
    respond = {
      'type': 'ok',
      'data': data
    }
    self.respond(message=respond, message_id=message['id'])

  def react_on_engine_active_settings(self, message):
    parameters = message['params']['parameters']
    data = self.integration.on_engine_active_settings_change(parameters)
    respond = {
      'type': 'ok',
      'data': data
    }
    self.respond(message=respond, message_id=message['id'])

  def send_object(self, object_data):
    data = object_data

    self.notify(message=data, method=METHOD_CREATE_OBJECT_METADATA)

  def send_best_shot(self, best_shot):
    data = best_shot

    self.notify(message=data, method=METHOD_CREATE_BEST_SHOT_METHOD)

  def send_title_image(self, title_image):
    data = title_image

    self.notify(message=data, method=METHOD_CREATE_TITLE_IMAGE_METHOD)
