# -*- coding: utf-8 -*-
"""
Created on Wed May 07 10:15:53 2014
@author: Anil Sener
"""
import collections
import numpy as np
from scipy import linalg, mat, dot
from sklearn.externals import six
from sklearn.base import BaseEstimator, ClassifierMixin,ClusterMixin, TransformerMixin
from abc import ABCMeta
from sklearn.linear_model import LogisticRegression as LR
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.utils.extmath import row_norms, squared_norm
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.cluster.k_means_ import _labels_inertia_precompute_dense
from sklearn.cluster import _k_means
class semiKMeans(six.with_metaclass(ABCMeta, BaseEstimator,ClusterMixin, TransformerMixin)):
    def __init__(self,maxiter=100,fixedprec=1e-9,verbose=False):
        self.maxiter=maxiter
        self.verbose=verbose
        self.fixedprec=fixedprec
        self.labels=None
        self.plattlr = None
        self.cluster_centers_=None
    def predict(self, X):
            """Predict the closest cluster each sample in X belongs to.

            In the vector quantization literature, `cluster_centers_` is called
            the code book and each value returned by `predict` is the index of
            the closest code in the code book.

            Parameters
            ----------
            X : {array-like, sparse matrix}, shape = [n_samples, n_features]
                New data to predict.

            Returns
            -------
            labels : array, shape [n_samples,]
                Index of the cluster each sample belongs to.
            """
            #check_is_fitted(self, 'cluster_centers_')

            X = self._check_test_data(X)
            x_squared_norms = row_norms(X, squared=True)
            return _labels_inertia(X, x_squared_norms, self.cluster_centers_)[0]

    def _check_test_data(self, X):
        X = check_array(X, accept_sparse='csr', dtype=FLOAT_DTYPES,
                        warn_on_dtype=True)
        n_samples, n_features = X.shape
        expected_n_features = self.cluster_centers_.shape[1]
        if not n_features == expected_n_features:
            raise ValueError("Incorrect number of features. "
                             "Got %d features, expected %d" % (
                                 n_features, expected_n_features))

        return X
    def fit_transform(self,texts,labels):
        # Initialize clusters with labeled data
        clust_names = [x for x, y in collections.Counter(labels).items() if y > 0]
        clust_names = sorted(clust_names)[1:]
        #print(clust_names)
        #centroids = np.zeros((len(clust_names),len(texts[1,:])))
        centroids = np.zeros((len(clust_names),texts.shape[1]))

        for unique_name in clust_names:
            indices = [i for i, x in enumerate(labels) if x == unique_name]

            aux = texts[indices,:]
            #print(np.mean(aux,axis=0).shape)
            #print(centroids[clust_names.index(unique_name),:].shape)
            centroids[clust_names.index(unique_name),:] = np.mean(aux,axis=0)

        texts = mat(texts)
        centroids = mat(centroids)
        new_labels = labels

        cnt = 0
        # Main loop
        while cnt<self.maxiter:
            cnt +=1

            if self.verbose:
                print('Iter: '+str(cnt))

            # Assign data to nearest centroid (cosine distance)
            dist = dot(texts,centroids.T)/linalg.norm(texts)/linalg.norm(centroids)
            n_lab = dist.argmax(axis=1)

            for ii in range(len(n_lab)):
                new_labels[ii] = clust_names[n_lab[ii,0]]

    #        print [y for x, y in collections.Counter(new_labels).items() if y > 0]

            # Recalculate clusters
            new_centroids = np.zeros((len(clust_names),len(texts.T)))
            for unique_name in clust_names:
                indices = [i for i, x in enumerate(new_labels) if x == unique_name]

                if len(indices)>0:
                    aux = texts[indices,:]
                    new_centroids[clust_names.index(unique_name),:] = aux.mean(0)
                else:
                    new_centroids[clust_names.index(unique_name),:] = centroids[clust_names.index(unique_name),:]

            # Check exit condition
            difference = np.power((centroids-new_centroids),2)

            if difference.sum()<self.fixedprec:
                break;
            else:
                self.labels = new_labels
                centroids = new_centroids

        self.cluster_centers_=new_centroids
        self.plattlr = LR()
        preds = self.predict(texts[labels!=-1,:])
        self.plattlr.fit( preds.reshape( -1, 1 ), labels[labels!=-1])

        return (labels)

    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X.

        The model need to have probability information computed at training
        time: fit with attribute `probability` set to True.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """


        preds = self.predict(X)
        #################Should be Added by me################

        #########################
        return self.plattlr.predict_proba(preds.reshape( -1, 1 ))




def _labels_inertia(X, x_squared_norms, centers,
                    precompute_distances=True, distances=None):
    """E step of the K-means EM algorithm.

    Compute the labels and the inertia of the given samples and centers.
    This will compute the distances in-place.

    Parameters
    ----------
    X: float64 array-like or CSR sparse matrix, shape (n_samples, n_features)
        The input samples to assign to the labels.

    x_squared_norms: array, shape (n_samples,)
        Precomputed squared euclidean norm of each data point, to speed up
        computations.

    centers: float64 array, shape (k, n_features)
        The cluster centers.

    precompute_distances : boolean, default: True
        Precompute distances (faster but takes more memory).

    distances: float64 array, shape (n_samples,)
        Pre-allocated array to be filled in with each sample's distance
        to the closest center.

    Returns
    -------
    labels: int array of shape(n)
        The resulting assignment

    inertia : float
        Sum of distances of samples to their closest cluster center.
    """
    n_samples = X.shape[0]
    # set the default value of centers to -1 to be able to detect any anomaly
    # easily
    labels = -np.ones(n_samples, np.int32)
    if distances is None:
        distances = np.zeros(shape=(0,), dtype=np.float64)
    # distances will be changed in-place
    if sp.issparse(X):
        inertia = _k_means._assign_labels_csr(
            X, x_squared_norms, centers, labels, distances=distances)
    else:
        if precompute_distances:
            return _labels_inertia_precompute_dense(X, x_squared_norms,
                                                    centers, distances)
        inertia = _k_means._assign_labels_array(
            X, x_squared_norms, centers, labels, distances=distances)
    return labels, inertia


