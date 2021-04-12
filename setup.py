from setuptools import setup

setup(name='laaac',
      version='0.0.1',
      description='Landscape-Aware Automated Algorithm Configuration',
      author='Yasha Pushak',
      author_email='ypushak@cs.ubc.ca',
      license='BSD-3',
      packages=['laaac'],
      install_requires=[
          'numpy>=1.19.1',
          'pandas>=1.1.1',
          'ray>1.1.0',
          'scipy>=1.6.1',
          'scikit-learn>=0.23.2',
          'ConfigSpace>=0.4.16'],
      extras_require={
          'plot': ['streamlit>=0.79.0']},
      zip_safe=False)
