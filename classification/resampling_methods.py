from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import KMeansSMOTE, SMOTE


def make_resampler(method, random_seed = 24):
    if method=='CC':
        return ClusterCentroids(random_state=random_seed)
    elif method=='SMOTE':
        return SMOTE(random_state=random_seed)
    elif method=='kSMOTE':
        #resample = KMeansSMOTE(cluster_balance_threshold=0.00001, random_state=42)
        return KMeansSMOTE(cluster_balance_threshold=0.00001, random_state=random_seed)
    print('Resampling method not valid, choose one among CC, SMOTE, kSMOTE')


def describe_resampler(resample):
    """The parameters a fitted resampler actually used (defaults resolved by imblearn)."""
    d = {'class': type(resample).__name__, 'random_state': resample.random_state}
    if isinstance(resample, KMeansSMOTE):
        d.update(k_neighbors=resample.nn_k_.n_neighbors - 1, kmeans_estimator=repr(resample.kmeans_estimator_),
                 kmeans_n_clusters=resample.kmeans_estimator_.n_clusters,
                 cluster_balance_threshold=resample.cluster_balance_threshold_)
    elif isinstance(resample, SMOTE):
        d.update(k_neighbors=resample.nn_k_.n_neighbors - 1)
    elif isinstance(resample, ClusterCentroids):
        d.update(estimator=repr(resample.estimator_), voting=resample.voting_)
    return d


def resample_dataset(X_train, Y_train, method, random_seed = 24, return_sampler = False):
    resample = make_resampler(method, random_seed)
    try:
        X_train, Y_train = resample.fit_resample(X_train, Y_train)

    except:
        X_train = None
        Y_train = None

    if return_sampler:
        return X_train, Y_train, resample
    return X_train, Y_train
