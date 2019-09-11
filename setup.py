from setuptools import setup, find_packages

setup(name='pytorch_lib',
      version='0.2.1',
      description='PyTorch wrapper for Deep Learning',
      url='https://github.com/raharth/PyTorch_Lib',
      author='Jonas Goltz',
      author_email='goltz.jonas@googlemail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'tqdm', 'numpy', 'torchvision', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn'
      ],
      zip_safe=False)

# build *.tar.gz by
# $python setup.py sdist