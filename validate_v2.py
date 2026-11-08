import json
from pathlib import Path
import run
from advanced_models import BlendModel
ROOT = Path(__file__).resolve().parent
config_path = ROOT / 'model_config_v2.json'
config = json.loads(config_path.read_text())
destination = ROOT / 'research_v2' / 'official'
destination.mkdir(parents=True, exist_ok=True)
(destination / 'selected_config.json').write_text(json.dumps(config, indent=2))

def make_model(track):
    members = config['tracks'][track]
    print(f'Training {track}: ' + ', '.join((m['name'] for m in members)), flush=True)
    return BlendModel(track, members)
if __name__ == '__main__':
    run.models.make_model = make_model
    run.ROOT = destination
    run.validate()
    run.report()
