import hashlib
import json
from pathlib import Path
import urllib.request
REVISION = '96a258439e2a85f650699be8339e52d8ac5614b7'
ROOT = Path(__file__).resolve().parent

def main():
    folder = ROOT / 'data_local'
    folder.mkdir(exist_ok=True)
    expected = {'listings': '64c780c97e395e20c9c5fb93ceb50fe0c1ff94604b2f8fc5f173f998c5b96ecb', 'test_off_market': '5840818b2045bcfc4eaaf8eb6b1fe9c1f2f45616e9698b573304799d140dbe58', 'test_list_aware': 'e2e317b6a63a7b591c460c8a3ff18264dc8ac3259b3a5a6d94c4d86f336515a5'}
    hashes = {}
    for name in ['listings', 'test_off_market', 'test_list_aware']:
        path = folder / f'{name}.csv'
        url = f'https://huggingface.co/datasets/Baliyang/SSU_data_dash/resolve/{REVISION}/{name}.csv'
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]:
            temporary = path.with_suffix('.download')
            with urllib.request.urlopen(url, timeout=120) as source, temporary.open('wb') as target:
                while (block := source.read(1024 * 1024)):
                    target.write(block)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected[name]:
                raise ValueError(f'Unexpected checksum for {name}')
            temporary.replace(path)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(name, path.stat().st_size, hashes[name], flush=True)
    (folder / 'manifest.json').write_text(json.dumps({'revision': REVISION, 'sha256': hashes}, indent=2))
if __name__ == '__main__':
    main()
