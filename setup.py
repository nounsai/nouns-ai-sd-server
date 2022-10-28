from setuptools import setup, find_packages
import os

def _read_reqs(relpath):
    fullpath = os.path.join(os.path.dirname(__file__), relpath)
    with open(fullpath) as f:
        return [s.strip() for s in f.readlines() if (s.strip() and not s.startswith("#"))]

setup(
    name='stable-diffusion',
    version='0.0.1',
    description='',
    packages=find_packages(),
    install_requires=_read_reqs("requirements.txt"),
)
