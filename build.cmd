set /p mode="enter mode [major, minor, patch]:"
bumpversion %mode%
python setup.py sdist