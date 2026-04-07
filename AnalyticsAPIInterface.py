import abc
from collections.abc import Hashable


class AnalyticsAPIInterface(metaclass=abc.ABCMeta):

  @classmethod
  def __subclasshook__(cls, subclass):
    return (hasattr(subclass, 'set_parameters') and
            callable(subclass.set_parameters) and
            hasattr(subclass, 'get_device_agent_manifest') and
            callable(subclass.get_device_agent_manifest) or
            NotImplemented)

  @abc.abstractmethod
  def set_parameters(self, parameters: dict):
    raise NotImplemented

  @abc.abstractmethod
  def get_device_agent_manifest(self, device_agent_id: Hashable) -> dict:
    raise NotImplemented

  @abc.abstractmethod
  def on_device_agent_created(self, device_parameters: dict):
    raise NotImplemented

  @abc.abstractmethod
  def on_device_agent_deletion(self, device_id: str):
    raise NotImplemented

  @abc.abstractmethod
  def get_integration_engine_side_settings(self, parameters) -> dict:
    raise NotImplemented

  @abc.abstractmethod
  def get_integration_device_agent_side_settings(self, parameters, device_id) -> dict:
    raise NotImplemented

  @abc.abstractmethod
  def on_agent_settings_update(self, parameters, device_id) -> dict:
    raise NotImplemented

  @abc.abstractmethod
  def on_agent_active_settings_change(self, parameters, device_id) -> dict:
    raise NotImplemented

  @abc.abstractmethod
  def on_engine_settings_update(self, parameters) -> dict:
    raise NotImplemented

  @abc.abstractmethod
  def on_engine_active_settings_change(self, parameters) -> dict:
    raise NotImplemented
