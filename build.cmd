set /p mode="enter mode:"
bumpversion %mode%
python setup.py sdist