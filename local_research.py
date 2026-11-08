import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'research_v2'

def package():
    subprocess.run([sys.executable, str(ROOT / 'select_research.py'), 'summary'], cwd=ROOT, check=True)
    path = ROOT / 'research_results.zip'
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for pattern in ['*.csv', '*.json', '*.jsonl', '*.md', '*.log']:
            for f in OUT.glob(pattern):
                archive.write(f, f.relative_to(ROOT))
        for f in (OUT / 'predictions').glob('*'):
            if f.suffix in ['.json', '.csv']:
                archive.write(f, f.relative_to(ROOT))
    print(f'\nSEND THIS FILE: {path}', flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--package-only', action='store_true')
    args = parser.parse_args()
    if args.package_only:
        package()
        return
    subprocess.run([sys.executable, str(ROOT / 'download_data.py')], cwd=ROOT, check=True)
    catalogue = json.loads((OUT / 'catalogue.json').read_text())
    env = os.environ.copy()
    env.update(DASH_THREADS=str(args.threads), OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    log = OUT / 'local_run.log'
    with log.open('a', encoding='utf-8') as stream:
        for track in ['off_market', 'list_aware']:
            names = []
            for name, cfg in catalogue.items():
                if track == 'off_market' and (cfg.get('features') == 'market' or name == 'lgb_list_calibrated'):
                    continue
                if track == 'list_aware' and (cfg.get('area_offset') or name == 'lgb_l2_calibrated'):
                    continue
                names.append(name)
            print(f'\n{track}: {len(names)} configurations. Completed runs are resumed automatically.', flush=True)
            command = [sys.executable, 'research.py', '--stage', 'screen', '--track', track, '--names', ','.join(names)]
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    stream.write(line)
                    stream.flush()
                if process.wait() != 0:
                    print('A process stopped. Rerun this command to resume.', flush=True)
                    break
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                print('\nStopped. Completed results are preserved. Rerun to resume.', flush=True)
                break
    package()
if __name__ == '__main__':
    main()
