#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages


setup(
    author="Emanuele Dalsasso, Youcef Kemiche, Pierre Blanchard, Hadrien Mariaccia",
    author_email='engineer@hi-paris.fr',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Environment :: Console',
        'Operating System :: OS Independent',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="Python Boilerplate contains all the boilerplate you need to create a Python package.",
    entry_points={
        'console_scripts': [
            'deepdespeckling=deepdespeckling.cli:main',
        ],
    },
    install_requires=["numpy", "Pillow", "scipy",
                      "torch", "opencv-python", "tqdm"],
    license="MIT license",
    include_package_data=True,
    keywords='deepdespeckling',
    name='deepdespeckling',
    packages=find_packages(include=['deepdespeckling', 'deepdespeckling.*']),
    url='https://github.com/hi-paris/deepdespeckling',
    version='0.3',
    zip_safe=False,
)
