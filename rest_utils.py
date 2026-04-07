import requests
import urllib.parse
import hashlib
import base64

REGISTER_PATH = "/rest/v4/analytics/integrations/*/requests"
LOGIN_PATH = "/rest/v4/login/sessions"
ENGINES_PATH = "/rest/v4/analytics/engines"
DEVICE_AGENTS_PATH = "/rest/v4/analytics/engines/{engine_id}/deviceAgents"
DEVICE_STREAM_PATH = "/rest/v4/devices/{device_id}/media.{video_format}"
NONCE_PATH = "/api/getNonce"
RTSP_PATH = "/{device_id}"
WEBRTC_PATH = "/rest/v3/devices/{device_id}/webrtc"
SITE_INFO_PATH = "/rest/v4/site/info"


def _concat_url(server_url, path, scheme='https') -> str:
    initial_url = urllib.parse.urlparse(server_url)
    print(initial_url)
    result = urllib.parse.urlunparse(initial_url._replace(path=path, scheme=scheme))
    print(result)
    return str(result)


def register_integration(server_url, integration_manifest, engine_manifest):
    params = {"integrationManifest": integration_manifest,
              "engineManifest": engine_manifest,
              "pinCode": "9876"}
    result = requests.post(_concat_url(server_url=server_url, path=REGISTER_PATH), json=params, verify=False)
    if result.status_code == 200:
        creds = {"username" : result.json()['username'], "password": result.json()['password']}
        return creds
    else:
        raise RuntimeError(result.status_code, result.text)


def get_device_agents(server_url: str, credentials: dict, integration_id: str) -> list:
    token = authorize(server_url, credentials)
    print('received auth token: ', token)
    parameters = 'integrationId="{integration_id}"'.format(integration_id=integration_id)

    header = {'Authorization': 'Bearer ' + token}
    result = requests.request(method='GET',
                              params=parameters,
                              url=_concat_url(server_url=server_url, path=ENGINES_PATH),
                              verify=False,
                              headers=header)
    if result.status_code != 200:
        raise RuntimeError('Unable to get engine id')

    engine_id = result.json()[0]['id']
    print('engine id: ', engine_id)

    result = requests.request(method='GET',
                              url=_concat_url(server_url=server_url,
                                              path=DEVICE_AGENTS_PATH.format(engine_id=engine_id)),
                              verify=False,
                              headers=header)
    if result.status_code != 200:
        raise RuntimeError('Unable to get device agents')

    device_agents = result.json()
    return device_agents


def authorize(server_url, credentials: dict):
    params = {
        "username": credentials['username'],
        "password": credentials['password']
    }

    result = requests.request(method='POST', url=_concat_url(server_url=server_url,path=LOGIN_PATH),
                              json=params,
                              verify=False)

    if result.status_code == 200:
        return result.json()['token']
    else:
        raise RuntimeError('Unable to authorize')

def create_auth(server_url: str, credentials: dict, method: str):
  response = requests.request(method='GET', url=_concat_url(server_url=server_url, path=NONCE_PATH), verify=False).json()
  realm = response['reply']['realm']
  nonce = response['reply']['nonce']

  data = f"{credentials['username']}:{realm}:{credentials['password']}".encode()

  md = hashlib.md5()
  md.update(data)
  digest = md.hexdigest()

  md = hashlib.md5()
  data = f"{method}:".encode()
  md.update(data)
  method = md.hexdigest()

  md = hashlib.md5()
  data = f"{digest}:{nonce}:{method}".encode()
  md.update(data)
  auth_digest = md.hexdigest()
  auth = f"{credentials['username']}:{nonce}:{auth_digest}".encode()
  return str(base64.b64encode(auth), 'utf-8')

def get_stream_link(server_url, credentials, device_id: str, video_format: str):
  auth = create_auth(server_url, credentials, 'GET')
  link = _concat_url(server_url=server_url,
                     path=DEVICE_STREAM_PATH).format(device_id=device_id, video_format=video_format)

  link += '?auth={auth}'.format(auth=auth)

  return link


def get_rtsp_link(server_url, credentials, device_id: str):
  auth = create_auth(server_url, credentials, 'PLAY')
  link = _concat_url(server_url=server_url, scheme='rtsp',
                     path=RTSP_PATH).format(device_id=device_id)

  link += '?auth={auth}'.format(auth=auth)
  link += '&fps=30'

  return link


def get_site_id(server_url, credentials):
  token = authorize(server_url, credentials)

  header = {'Authorization': 'Bearer ' + token}
  result = requests.request(method='GET',
                            url=_concat_url(server_url=server_url, path=SITE_INFO_PATH),
                            verify=False,
                            headers=header)
  if result.status_code != 200:
    raise RuntimeError('Unable to get site id')

  site_id = result.json()['cloudId']
  print('engine id: ', site_id)

  return site_id
