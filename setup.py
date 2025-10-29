from setuptools import setup, find_packages

setup(
    name='gridbot_env',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'numpy',
        'pygame',
        'stable-baselines3',
    ],
    python_requires='>=3.8',
)