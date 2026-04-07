import config
import json
from VectorizerIntegration import VectorizerIntegration
import requests

INTEGRATION_MANIFEST_PATH = "manifests/integration_manifest.json"
ENGINE_MANIFEST_PATH = "manifests/engine_manifest.json"
AGENT_MANIFEST_PATH = "manifests/agent_manifest.json"
CREDENTIALS_PATH = ".credentials"

if __name__ == "__main__":

  with (open(INTEGRATION_MANIFEST_PATH, 'r') as f_i,
        open(ENGINE_MANIFEST_PATH, 'r') as f_e,
        open(AGENT_MANIFEST_PATH, 'r') as f_a):
    integration_manifest = json.load(f_i)
    engine_manifest = json.load(f_e)
    agent_manifest = json.load(f_a)

  server_url = config.server_url

  resp = requests.head(f'{server_url}/ui', allow_redirects=False, verify=False)

  if resp.status_code in (301, 302, 307, 308):
    redirect_url = (resp.headers["Location"].replace("https://", "wss://").
                    replace("http://", "ws://").
                    replace("/ui", ""))
  else:
    redirect_url = server_url

  # redirect_url = server_url

  print(f"Connecting to: {redirect_url}")

  integration = VectorizerIntegration(server_url=redirect_url,
                                      integration_manifest=integration_manifest,
                                      engine_manifest=engine_manifest,
                                      credentials_path=CREDENTIALS_PATH,
                                      device_agent_manifest=agent_manifest)

  integration.run()
