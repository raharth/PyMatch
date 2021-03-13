from setuptools import setup, find_packages

setup(name='pymatch',
      version='1.3.4',
      description='PyTorch wrapper for Deep Learning',
      url='https://github.com/raharth/PyMatch',
      author='Jonas Goltz',
      author_email='goltz.jonas@googlemail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'tqdm', 'numpy', 'torch', 'torchvision', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn', 'tables'
      ],
      zip_safe=False)
