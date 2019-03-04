class OOBGenerator:
    """A bootstrap Out-of-Bag resampler.

    Args:
        n_splits (int): The number of resamplings to perform.
        random_state (int): Seed for the pseudo-random number generator.

    """

    def __init__(self, n_splits, random_state):

        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, y, **kwargs):
        """Generates Out-of-Bag samples.

        Args:
            X (array-like): The predictor data.
            y (array-like): The target data.

        Returns:
            (generator): An iterable with X and y sample indicators.

        """
        rgen = np.random.RandomState(self.random_state)

        nrows, _ = np.shape(X)
        sample_indicators = np.arange(nrows)
        for _ in range(self.n_splits):
            train_idx = rgen.choice(
                sample_indicators, size=nrows, replace=True
            )
            test_idx = np.array(
                list(set(sample_indicators) - set(train_idx)), dtype=int
            )
            yield train_idx, test_idx
