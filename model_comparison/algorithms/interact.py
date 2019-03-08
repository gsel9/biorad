


class Interact(base.BaseSelector):
    """Identifies interacting features???

    """

    def __init__(self, delta=0000.1):

        self.delta = delta

        # NOTE:
        self.support = None

    def fit(self, X, y=None, **kwargs):

        _, num_cols = np.shape(X)

        ranking_support = np.arange(num_cols, dtype=int)

        ranks = np.zeros(num_cols, dtype=float)
        # NOTE: May be replaced by univariate feature selection methods???
        for num in range(num_cols):
            ranks[num] = self.feature_ranking(X[:, num], y)

        # Order in descending values of ranking.
        sorting_idx = np.argsort(ranks)
        sorted_ranking_support = ranking_support[sorting_idx]

        # Start with the feature at the end of the list.
        support = np.zeros(num_cols, dtype=int)
        for num, F in enumerate(sorted_ranking_support[::-1]):
            p = self.consistency_contribution(F, support)
            if p > self.delta:
                support.append(num)

        self.support = self.check_support(support)

    def predict(self, X, y=None, **kwargs):

        return X[:, self.S_best]

    def consistency_contribution(self):
        pass

    # NOTE: Could use alternative univariate methods.
    def feature_ranking(self, x, y):

        entropy_x = None
        entropy_y = None

        mutual_info = None

        return 2 * (mutual_info / (entropy_x + entropy_y))
