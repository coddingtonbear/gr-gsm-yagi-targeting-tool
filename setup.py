import os
from setuptools import setup, find_packages
import uuid


requirements_path = os.path.join(
    os.path.dirname(__file__),
    'requirements.txt',
)
try:
    from pip.req import parse_requirements
    requirements = [
        str(req.req) for req in parse_requirements(
            requirements_path,
            session=uuid.uuid1()
        )
    ]
except ImportError:
    requirements = []
    with open(requirements_path, 'r') as in_:
        requirements = [
            req for req in in_.readlines()
            if not req.startswith('-') and
            not req.startswith('#')
        ]


setup(
    name='gr-gsm-yagi-targeting-tool',
    version='0.1.2',
    url='https://github.com/coddingtonbear/gr-gsm-yagi-targeting-tool',
    description=(
        'Not sure where to point your yagi antenna?  Find out.'
    ),
    author='Adam Coddington',
    author_email='me@adamcoddington.net',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'yagi-targeting-tool = yagi_targeting_tool:cmdline'
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    packages=find_packages(),
)
