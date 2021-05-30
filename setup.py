from setuptools import setup, find_packages

# Setting up
setup(
      # the name must match the folder name 'verysimplemodule'
      name="bayesianadaptivemodels",
      version="0.0.1",
      author="Hugo Dolan",
      author_email="<hugojdolan@gmail.com>",
      description="Bayesian adaptive time series models",
      long_description="See readme file",
      packages=['bayesianadaptivemodels'],
      install_requires=[], # add any additional packages that
      # needs to be installed along with your package. Eg: 'caer'
      
      keywords=['python', 'first package'],
      classifiers= [
                    "Development Status :: 3 - Alpha",
                    "Intended Audience :: Education",
                    "Programming Language :: Python :: 2",
                    "Programming Language :: Python :: 3",
                    "Operating System :: MacOS :: MacOS X",
                    "Operating System :: Microsoft :: Windows",
                    ]
      )
