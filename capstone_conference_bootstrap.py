# Safe bootstrap for the complete capstone conference-paper pipeline.
# This avoids upgrading Colab's core NumPy/Pandas stack inside a live runtime.

import re
import sys
import subprocess
import urllib.request

print('Preparing Colab environment safely...')

# IMPORTANT: Do not upgrade numpy/pandas/scikit-learn here.
# Colab ships a mutually compatible scientific Python stack. Replacing NumPy
# inside an already-running kernel can leave mixed binary/Python modules loaded
# and produce errors such as: cannot import name `_slice` from numpy._core.umath.
subprocess.run(
    [
        sys.executable, '-m', 'pip', 'install', '-q',
        'transformers>=4.51.0',
        'accelerate',
        'bitsandbytes',
        'sentence-transformers',
        'huggingface_hub',
        'openpyxl',
        'xlrd'
    ],
    check=True
)

ORIGINAL_URL = 'https://raw.githubusercontent.com/m-i-shakib/basic-ai/main/capstone_conference_pipeline.py'
print('Downloading complete conference-paper pipeline...')
with urllib.request.urlopen(ORIGINAL_URL) as r:
    source = r.read().decode('utf-8')

# Remove the original pipeline's package-install command because the bootstrap
# above has already installed only the non-core dependencies safely.
source = re.sub(
    r"import subprocess,sys\s*\nsubprocess\.run\(\[sys\.executable,'-m','pip','install'.*?\],check=True\)\s*\n",
    "",
    source,
    count=1,
    flags=re.S,
)

# Safety check: fail rather than accidentally run the old NumPy-upgrading line.
if "'pandas','numpy','openpyxl'" in source or '"pandas","numpy","openpyxl"' in source:
    raise RuntimeError('Safety patch did not remove the old package installer. Please refresh the notebook and try again.')

print('Running full conference-paper pipeline...')
exec(compile(source, '<capstone_conference_pipeline>', 'exec'), globals())
