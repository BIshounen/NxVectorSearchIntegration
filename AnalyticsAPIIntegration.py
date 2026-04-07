import abc
import asyncio
import os
import rest_utils
import json
import httpx

from AnalyticsAPIInterface import AnalyticsAPIInterface
from NxJSONRPC import NxJSONRPC


class AnalyticsAPIIntegration(AnalyticsAPIInterface):

  class ApprovalAwaitable:

    def __init__(self, outer):
      self.outer = outer

    def __await__(self):
      while not self.outer.is_approved:
        yield

      return True


  def __init__(self, server_url: str, integration_manifest: dict, engine_manifest: dict, credentials_path: str):
    self.server_url = server_url
    self.credentials = dict()
    self.integration_manifest = integration_manifest
    self.engine_manifest = engine_manifest
    self.credentials_path = credentials_path
    self.integration_id = None
    self.is_approved = False

    self.register()

    self.JSONRPC = None

  def register(self):
    if not os.path.isfile(self.credentials_path):
      creds = rest_utils.register_integration(integration_manifest=self.integration_manifest,
                                              engine_manifest=self.engine_manifest,
                                              server_url=self.server_url)
      with open(self.credentials_path, 'w') as f:
        json.dump(creds, f)
    with open(self.credentials_path, 'r') as f:
      temp_credentials = json.load(f)
      self.credentials = {'username': temp_credentials['username'], 'password': temp_credentials['password']}


  def set_parameters(self, parameters: dict):
    self.is_approved = parameters.get('parameters', {}).get('integrationRequestData', {}).get('isApproved', False)
    self.integration_id = parameters.get('parameters', {}).get('integrationRequestData', {}).get('integrationId', None)
    print("is approved: ", self.is_approved)
    print("integration id", self.integration_id)

  @abc.abstractmethod
  def get_device_agent_manifest(self, device_agent_id):
    raise NotImplemented

  @abc.abstractmethod
  def on_device_agent_created(self, device_parameters):
    raise NotImplemented

  @abc.abstractmethod
  def on_device_agent_deletion(self, device_id):
    raise NotImplemented

  @abc.abstractmethod
  def on_agent_settings_update(self, parameters, device_id):
    raise NotImplemented

  @abc.abstractmethod
  def on_agent_active_settings_change(self, parameters, device_id):
    raise NotImplemented

  @abc.abstractmethod
  def on_engine_settings_update(self, parameters):
    raise NotImplemented

  @abc.abstractmethod
  def on_engine_active_settings_change(self, parameters):
    raise NotImplemented

  def get_integration_engine_side_settings(self, parameters):
    return {}

  def get_integration_device_agent_side_settings(self, parameters, device_id):
    return {}

  async def main(self):

    self.JSONRPC = NxJSONRPC(server_url=self.server_url, integration=self)
    await self.JSONRPC.authorize(credentials=self.credentials)
    await self.JSONRPC.subscribe_on_users(credentials=self.credentials)
    approval = self.ApprovalAwaitable(self)
    await approval
    print('approved')
    device_agents = rest_utils.get_device_agents(server_url=self.server_url,
                                         credentials=self.credentials,
                                          integration_id=self.integration_id)

    for device_agent in device_agents:
      if device_agent['isEnabled']:
        device_parameters = {
          "parameters": {
            'id': device_agent['id']
          },
          "target": {
            "engineId": device_agent['engineId']
          }
        }
        self.on_device_agent_created(device_parameters=device_parameters)

    await self.JSONRPC.subscribe_to_analytics(self.integration_id)
    print('subscribed')


  def run(self):
    asyncio.run(self.main())
