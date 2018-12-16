def _check_estimator(nfeatures, hparams, estimator, random_state):

    # Using all available features after feature selection.
    if 'n_components' in hparams:
        if nfeatures - 1 < 1:
            hparams['n_components'] = 1
        else:
            hparams['n_components'] = nfeatures - 1

    # If stochastic algorithms.
    try:
        model = estimator(**hparams, random_state=random_state)
    except:
        model = estimator(**hparams)

    return model


def _update_prelim_results(results, path_tempdir, random_state, *args):
    # Update results <dict> container and write preliminary results to disk.
    (
        estimator, selector, best_params, avg_test_scores, avg_train_scores,
        best_features
    ) = args
    # Update results dict.
    results.update(
        {
            'model': estimator.__name__,
            'selector': selector['name'],
            'best_params': best_params,
            'avg_test_score': avg_test_scores,
            'avg_train_score': avg_train_scores,
            'best_features': best_features,
            'num_features': np.size(best_features)
        }
    )
    # Write preliminary results to disk.
    path_case_file = os.path.join(
        path_tempdir, '{}_{}_{}'.format(
            estimator.__name__, selector['name'], random_state
        )
    )
    ioutil.write_prelim_results(path_case_file, results)

    return results
