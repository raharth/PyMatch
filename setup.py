from setuptools import setup, find_packages

setup(name='pymatch',
      version='1.7.6',
      description='PyTorch wrapper for Deep Learning',
      url='https://github.com/raharth/PyMatch',
      author='Jonas Goltz',
      author_email='goltz.jonas@googlemail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'tqdm', 'numpy', 'torch', 'torchvision', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn', 'tables', 'wandb',
            'gym', 'box2d-py'
      ],
      zip_safe=False)
