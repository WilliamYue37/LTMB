from setuptools import setup, find_packages

setup(
    name='LTMB',
    version='1.0.0',
    author='William Yue',
    author_email='william.yue@utexas.edu',
    description='A Long-Term Memory Benchmark for Sequential Decision Making',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/WilliamYue37/LTMB',
    packages=find_packages(),
    install_requires=[
        'minigrid==2.3.1',
    ],
    python_requires='>=3.7',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords=[
    'gym', 
    'benchmark', 
    'reinforcement learning', 
    'imitation learning', 
    'machine learning', 
    'long-term memory', 
    'sequential decision making', 
    'minigrid', 
    'gridworld'
    ],
    license='MIT',
    entry_points={
        'console_scripts': [
            'play_hallway=ltmb.envs.hallway:main',
        ],
    },
)
