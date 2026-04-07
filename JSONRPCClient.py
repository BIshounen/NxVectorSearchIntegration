import json
from ws_class import WSClass

METHOD_CREATE_SESSION = "rest.v3.login.sessions.create"
METHOD_SUBSCRIBE_USERS = "rest.v3.users.subscribe"
METHOD_UPDATE_USERS = "rest.v3.users.update"
METHOD_SUBSCRIBE_ANALYTICS = 'rest.v4.analytics.subscribe'
METHOD_CREATE_DEVICE_AGENT = 'rest.v4.analytics.engines.deviceAgents.create'
METHOD_CREATE_DEVICE_AGENT_MANIFEST = 'rest.v4.analytics.engines.deviceAgents.manifest.create'


class JSONRPCClient:

  def __init__(self, server_url, integration):
    self.integration = integration
    self.ws_connect = WSClass(server_url=server_url, on_message=self.on_message)
    self.current_id = 0

    self.reply_queue = {}

  def on_message(self, message):
    jsn = json.loads(message)
    if 'id' in jsn:
      if jsn['id'] > self.current_id:
        self.current_id = jsn['id']

    if 'jsonrpc' in jsn:
      if 'error' in jsn:
        self.integration.print_message(message=jsn['error'].get('message', ''))
      else:
        if 'id' in jsn and jsn['id'] in self.reply_queue:
          self.reply_queue[jsn['id']](jsn.get('result'))
          self.reply_queue.pop(jsn['id'])
        if 'method' in jsn:
          if jsn['method'] == METHOD_UPDATE_USERS:
            self.set_parameters(jsn.get('params', {}))
          if jsn['method'] == METHOD_CREATE_DEVICE_AGENT:
            pass ## add device agent and reply with manifest
          else:
            self.integration.print_message(message=jsn)
        else:
          self.integration.print_message(message=jsn.get('result',''))
    else:
      print('Unknown message')

  def set_parameters(self, result):
    self.integration.set_parameters(result)

  def send(self, method, payload):
    self.current_id += 1
    message = {"method": method,
               "params": payload,
               "jsonrpc": "2.0",
               'id': self.current_id
               }
    self.ws_connect.send(json.dumps(message))

  def authorize(self, credentials):
    payload = {
      'username': credentials['user'],
      'password': credentials['password'],
      'setSession': True
    }
    self.send(method=METHOD_CREATE_SESSION, payload=payload)

  def subscribe_to_users(self, user_name):
    payload = {
      'id': user_name
    }
    self.reply_queue[self.current_id + 1] = self.set_parameters
    self.send(method=METHOD_SUBSCRIBE_USERS, payload=payload)

  def subscribe_to_analytics(self):
    payload = {}
    self.send(method=METHOD_SUBSCRIBE_ANALYTICS, payload=payload)
