# This file is part of pyRDDLGym.

# pyRDDLGym is free software: you can redistribute it and/or modify
# it under the terms of the MIT License as published by
# the Free Software Foundation.

# pyRDDLGym is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# MIT License for more details.

# You should have received a copy of the MIT License
# along with pyRDDLGym. If not, see <https://opensource.org/licenses/MIT>.

from setuptools import setup, find_packages


setup(
      name='pyRDDLGym-worldmodel',
      version='0.1',
      author="Michael Gimelfarb, Scott Sanner",
      author_email="mike.gimelfarb@mail.utoronto.ca, ssanner@mie.utoronto.ca",
      description="Training and evaluation of world models for RDDL-based environments.",
      license="MIT License",
      url="https://github.com/pyrddlgym-project/pyRDDLGym-worldmodel",
      packages=find_packages(),
      install_requires=['matplotlib==3.10.8', 'numpy==2.4.4', 'Pillow==12.2.0', 'pyrddlgym==2.7', 'torch==2.11.0+cu126', 'tqdm==4.67.3'],
      python_requires=">=3.13,<3.15",
      package_data={'': ['*.cfg']},
      include_package_data=True,
      classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
