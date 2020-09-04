class TerminationException(Exception):
    def __int__(self, *args, **kwargs):
        """
        Used for control of early termination.

        Args:
            *args:
            **kwargs:

        Returns:
        """
        super(TerminationException, self).__init__(*args, **kwargs)


class OverwriteException(Exception):
    def __init__(self, *args, **kwargs):
        """
        Used to stop user from overwriting already existing experiments.

        Args:
            *args:
            **kwargs:
        """
        super(OverwriteException, self).__init__(*args, **kwargs)
