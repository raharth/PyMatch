set /p mode="enter mode [major, minor, patch]:"
git add -u
bumpversion %mode%
python setup.py sdist