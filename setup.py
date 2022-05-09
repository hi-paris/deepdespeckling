#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]
test_requirements = [ ]

setup(
    author="emanuele dalsasso, youcef kemiche, pierre blanchard",
    author_email='y.kemiche06@hotmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Python Boilerplate contains all the boilerplate you need to create a Python package.",
    entry_points={
        'console_scripts': [
            'deepdespeckling=deepdespeckling.cli:main',
        ],
    },
    install_requires=["numpy","Pillow","scipy","torch","opencv-python","tqdm"],
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='deepdespeckling',
    name='deepdespeckling',
    packages=find_packages(include=['deepdespeckling', 'deepdespeckling.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ykemiche/deepdespeckling',
    version='0.1.0',
    zip_safe=False,
)
