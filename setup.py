from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    with open(file_path, "r") as file:
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements


setup(
    name='SentinelAI',
    version='0.0.1',
    author='Naman',
    author_email='namangupta2132@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)