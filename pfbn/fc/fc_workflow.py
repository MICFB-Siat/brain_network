import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import os, sys, json, time, subprocess, traceback, hashlib
from pathlib import Path
ROOT = Path(settings.out('fc'))
ROOT.mkdir(parents=True, exist_ok=True)
CODE = Path(settings.code('pfbn', 'fc'))
os.environ['TMPDIR'] = str(ROOT / 'tmp')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

def status(phase, **kwargs):
    (ROOT / 'results/status.json').write_text(json.dumps(dict(phase=phase, time_unix=time.time(), **kwargs), indent=2))

def stage(mode, tasks):
    cards = []
    q = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    for line in q.splitlines():
        (card, mem, util) = [x.strip() for x in line.split(',')]
        if int(mem) < 1000 and int(util) < 5:
            cards.append(card)
    if not cards:
        raise RuntimeError('No idle GPU')
    queue = list(tasks)
    running = []
    done = []
    while queue or running:
        occupied = {x['gpu'] for x in running}
        for card in cards:
            if card in occupied or not queue:
                continue
            (branch, seed) = queue.pop(0)
            name = f'{mode}_{branch}_seed{seed}'
            log = open(ROOT / 'logs' / f'{name}.log', 'x')
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = card
            p = subprocess.Popen([sys.executable, str(CODE / 'train_fc_adapter.py'), '--root', str(ROOT), '--mode', mode, '--branch', branch, '--seed', str(seed)], env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append(dict(p=p, log=log, gpu=card, name=name))
        for job in running[:]:
            rc = job['p'].poll()
            if rc is not None:
                job['log'].close()
                running.remove(job)
                if rc != 0:
                    raise RuntimeError(job['name'] + ' failed: ' + str(rc))
                done.append(job['name'])
        status(mode, done=done, pending=queue, running=[dict(name=x['name'], gpu=x['gpu'], pid=x['p'].pid) for x in running])
        time.sleep(3)

def main():
    for d in ['logs', 'results', 'models', 'selection', 'tmp']:
        (ROOT / d).mkdir(exist_ok=True)
    if (ROOT / 'results/COMPLETE.json').exists():
        raise FileExistsError('Already complete')
    stage('select', [(b, 0) for b in ['pretrained', 'random']])
    subprocess.run([sys.executable, str(CODE / 'train_fc_adapter.py'), '--root', str(ROOT), '--mode', 'choose'], check=True)
    stage('final', [(b, s) for s in range(3) for b in ['pretrained', 'random']])
    status('scoring')
    subprocess.run([sys.executable, str(CODE / 'score_fc.py'), '--root', str(ROOT)], check=True)
    status('complete')
if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        traceback.print_exc()
        status('failed', error=repr(e))
        raise
