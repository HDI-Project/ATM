from btb.tuning import Tuner


class MyTuner(Tuner):
    """
    Very bare_bones tuner that returns a random set of parameters each time.
    """
    def propose(self):
        """
        Generate and return a random set of parameters.
        """
        return self.create_candidates(1)[0, :]
