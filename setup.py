from setuptools import setup

setup(name='pytorch_lib',
      version='0.1',
      description='PyTorch wrapper for Deep Learning',
      url='https://github.com/raharth/PyTorch_Lib',
      author='Jonas Goltz',
      author_email='goltz.jonas@googlemail.com',
      license='MIT',
      packages=['torch', 'numpy', 'gym'],
      zip_safe=False)
