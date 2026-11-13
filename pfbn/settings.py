import os
import re
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(os.environ.get('BRAINNET_CONFIG', ROOT / 'configs' / 'paths.yaml'))
with open(CONFIG) as _f:
    _RAW = yaml.safe_load(_f) or {}
_REF = re.compile(r'\{(\w+)\}')


def get(key):
    value = os.environ.get('BRAINNET_' + key.upper(), _RAW[key])
    value = _REF.sub(lambda m: get(m.group(1)), str(value))
    return os.path.expanduser(os.path.expandvars(value))


def path(key, *parts):
    p = Path(get(key))
    if not p.is_absolute():
        p = ROOT / p
    return str(p.joinpath(*parts))


def code(*parts):
    return str(ROOT.joinpath(*parts))


def out(*parts):
    return path('outputs', *parts)


def tmp(*parts):
    os.makedirs(path('tmp'), exist_ok=True)
    return path('tmp', *parts)


def scratch(*parts):
    return path('scratch', *parts)


if __name__ == '__main__':
    key, parts = sys.argv[1], sys.argv[2:]
    print(get(key) if key in ('wb_command', 'python') and not parts else path(key, *parts))
