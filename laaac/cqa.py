import copy 
import abc

from scipy.optimize import linprog, minimize
import numpy as np
from numpy.linalg import LinAlgError


class CQA(metaclass=abc.ABCMeta):

    def __init__(self, norm=1):
        self._res = None
        self._n = None
        self._norm = norm
        self._attrs_to_save = ['_res', '_n', '_norm']
        
    @abc.abstractmethod
    def fit(self, X, y, sample_weight=None):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass

    @abc.abstractmethod
    def get_model(self):
        pass

    def to_dict(self):
        attr_dict = {}
        for attr in self._attrs_to_save:
            attr_dict[attr] = getattr(self, attr)
        return attr_dict

    def from_dict(self, attr_dict):
        for attr, value in attr_dict.items():
            setattr(self, attr, value)

    def get_minimizer(self):
        self._verify_fit('get_minimizer')
        return _get_minimizer(*self.get_model())

    def score(self, X, y, norm=None):
        y_pred = self.predict(X)
        if norm is None:
            norm = self._norm
        return np.mean(np.abs(y-y_pred)**norm)

    def _verify_fit(self, function):
        if self._res is None:
            raise Exception(f'fit must be called before {function}.')

    def _verify_X_y_w(self, X, y, w):
        X = np.array(X)
        y = np.array(y)
        if X.ndim != 2:
           raise ValueError(f'X is the wrong shape. Must be two dimensional. '
                            f'Provided X with shape {X.shape}.')
        if X.shape[0] < self.min_samples(X.shape[1]):
           raise ValueError(f'Insufficient number of training examples '
                            f'({X.shape[0]}) for problem with {X.shape[1]} '
                            f'dimensions. Please provide at least '
                            f'{self.min_samples(X.shape[1])} training '
                            f'examples.')
        if y.ndim != 1:
            raise ValueError(f'y must be one dimensional. Provided y with '
                             f'shape {y.shape}.')
        if X.shape[0] != y.shape[0]:
            raise ValueError(f'X and y must have the same length. Provided '
                             f'X with shape {X.shape} and y with shape '
                             f'{y.shape}.')
        if w is not None:
            w = np.array(w)
            if w.ndim != 1:
                raise ValueError(f'sample_weight must be one dimensional. Provided '
                                 f'sample_weight with shape {w.shape}.')
            if X.shape[0] != w.shape[0]:
                raise ValueError(f'X and sample_weight must have the same length. '
                                 f'Provided X with shape {X.shape} and '
                                 f'sample_weight with shape {w.shape}.')

    def _shape(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = np.array([X])
        return X

    @staticmethod
    def min_samples(n_dims):
       return int(np.ceil((n_dims+1)*(n_dims+2)/2))


class AutoCQA(CQA):
    def __init__(self, norm=1, eps=1e-6):
        self._eps = eps
        super().__init__(norm)
        self._attrs_to_save.append('_eps')

    def fit(self, X, y, sample_weight=None, X_val=None, y_val=None):
        self._verify_X_y_w(X, y, sample_weight)
        models = [GCQA(method='DDCQA', norm=self._norm, eps=self._eps),
                  GCQA(method='SCQA', norm=self._norm, eps=self._eps)]
        if X_val is None or y_val is None:
            X_val = X
            y_val = y
        best_score = np.inf
        best_model = None
        for model in models:
            try:
                model.fit(X, y, sample_weight)
            except LinAlgError:
                continue
            for m in [model, model._prefit_model]:
                score = m.score(X_val, y_val, self._norm)
                if score <= best_score:
                    best_model = m
                    best_score = score
        self._res = best_model

    def predict(self, X):
        return self._res.predict(X)

    def get_model(self):
        return self._res.get_model()

    def score(self, X, y, norm=None):
        return self._res.score(X, y, norm)


class GCQA(CQA):
    def __init__(self, method='DDCQA', norm=1, eps=1e-6, lambda_=0):
        self._method = method
        self._eps = eps
        self._prefit_model = None
        self._lambda = lambda_
        super().__init__(norm)
        self._attrs_to_save.extend(['_method', '_eps', '_prefit_model', '_lambda'])

    def fit(self, X, y, sample_weight=None):
        self._verify_X_y_w(X, y, sample_weight)
        n = X.shape[1]
        self._n = n
        # First fit the easier type of model to the data
        # using a linear program.
        if self._method == 'SCQA':
            sqcu = SCQA()
            sqcu.fit(X, y, sample_weight)
            self._prefit_model = sqcu
            c0, c, H = sqcu.get_model()
            d = np.array([H[i,i] for i in range(len(H))])
            L = np.diag((d - self._eps)**0.5)
        elif self._method == 'DDCQA':
            ddqcu = DDCQA()
            ddqcu.fit(X, y, sample_weight)
            self._prefit_model = ddqcu
            c0, c, H = ddqcu.get_model()
            H = H - np.eye(n)*self._eps
            try:
                L = np.linalg.cholesky(H)
            except np.linalg.LinAlgError:
                # Sometimes the epsilon subtraction makes the
                # Hessian not be positive definite anymore.
                H = H + np.eye(n)*self._eps
                L = np.linalg.cholesky(H)
        else:
            raise NotImplementedError('Method not Support. Use SCQA or DDCQA.')
        l = [L[i,j] for (i,j) in _L_ij(n)]
        x0 = [c0] + list(c) + l
        # Now fit the more general model using that model's solution as
        # the initial guess.
        self._res = _nonlin_prog(X, y, x0, norm=self._norm, eps=self._eps, sample_weight=sample_weight, lambda_=self._lambda)

    def predict(self, X):
        self._verify_fit('predict')
        X = self._shape(X)
        return np.array(np.array([_q(x, self._res.x)[0] for x in X])).squeeze()

    def get_model(self):
        self._verify_fit('get_model')
        x = self._res.x
        n = self._n
        c0 = x[0]
        c = x[1:1+n]
        l = x[1+n:]
        L = np.zeros((n, n))
        for idx, (i, j) in enumerate(_L_ij(n)):
            L[i,j] = l[idx]
        H = np.dot(L, L.transpose()) + np.eye(n)*self._eps
        return c0, c, H

 
class DDCQA(CQA):

    def fit(self, X, y, sample_weight=None):
        self._verify_X_y_w(X, y, sample_weight)
        self._n = X.shape[1]
        res = _lin_prog_2(X, y, sample_weight=sample_weight)
        self._res = res

    def predict(self, X):
        self._verify_fit('predict')
        X = self._shape(X)
        return np.array([np.sum(self._res.x*_q_lp2(X[i:i+1,:])) for i in range(len(X))]).squeeze()

    def get_model(self):
        self._verify_fit('get_model')
        x0 = self._res.x
        n = self._n
        c0 = x0[0]
        c = x0[1:n+1]
        h = x0[n+1:]
        _H_ij = [(j,j) for j in range(n)] + [(i,j) for j in range(n) for i in range(j+1, n)]
        H = np.zeros((n, n))
        for idx, (i, j) in enumerate(_H_ij):
            H[i,j] = h[idx]
            H[j,i] = h[idx]
        H = H
        return c0, c, H


class SCQA(CQA):

    def fit(self, X, y, sample_weight=None):
        self._verify_X_y_w(X, y, sample_weight)
        self._n = X.shape[1]
        res = _lin_prog_1(X, y, sample_weight=sample_weight)
        self._res = res

    def predict(self, X):
        self._verify_fit('predict')
        X = self._shape(X)
        return np.array([np.sum(self._res.x*_q_lp1(X[i:i+1,:])) for i in range(len(X))]).squeeze()

    def get_model(self):
        self._verify_fit('get_model')
        x0 = self._res.x
        n = self._n
        c0 = x0[0]
        c = x0[1:n+1]
        d = x0[n+1:]
        H = np.diag(d)
        return c0, c, H

    @staticmethod
    def min_samples(n_dims):
       return int(n_dims*2 + 2)


def _get_minimizer(c0, c, H):
    return - 2*np.dot(np.linalg.inv(H + H.transpose()), c)


def _q_lp1(X, sample_weight=None):
    """
    q(c0, c, D; x) = c0 + sum_i(c_i*x_i) + 0.5*sum_i(d_(i,i)*x_i**2)

    Take the dot product of the output vector with the parameter vector 
    [c0, c, D] to get  sum_k q(c0, c, D; x^(k))

    To be used in lin_prog_1.

    Parameters
    ----------
    X : np.array
      The array of data points
    
    Returns
    -------
    np.array
      A vector of weights which can be dot producted with  fitted parameter
      vector to obtain the sum of the predicted values for X.
    """
    if sample_weight is not None:
        sample_weight = np.array(sample_weight)
        assert np.all(sample_weight > 0)
    else:
        sample_weight = np.ones(len(X))    
    return np.array([np.sum(sample_weight)]
                  + list(np.sum(X.transpose()*sample_weight, axis=1))
                  + list(0.5*np.sum(sample_weight*(X**2).transpose(), axis=1)))


def _q_lp2(X, sample_weight=None):
    """
    q(c0, c, H; x) = c0 + sum_i(c_i*x_i) + 0.5*sum_i(H_(i,i)*x_i**2)
                     + sum_j(sum_{i=j+1..n}(H_[i,j]*x_i*x_j))

    This vector also includes the n*(n-1)/2 beta variables that are used as
    constraints but are otherwise not included in the objective function.

    Take the dot product of the output vector with the parameter vector 
    [c0, c, H] to get  sum_k q(c0, c, H; x^(k))

    To be used in lin_prog_2.

    Parameters
    ----------
    X : np.array
      The array of data points
    
    Returns
    -------
    np.array
      A vector of weights which can be dot producted with  fitted parameter
      vector to obtain the sum of the predicted values for X.
    """    
    n = X.shape[1]
    if sample_weight is not None:
        sample_weight = np.array(sample_weight)
        assert np.all(sample_weight > 0)
    else:
        sample_weight = np.ones(len(X))
    return np.array(list(_q_lp1(X, sample_weight)) + [np.sum(sample_weight*X[:,i]*X[:,j]) for j in range(n) for i in range(j+1, n)] + 
                    [0 for j in range(n) for i in range(j+1, n)])


def _constraint_1(f, X, _q):
    """
    Constraint type 1
    f^(k) >= _q(x^(k)).transpose()*[c0, c, D]
    """
    A_ub = []
    # The copy is necessary because lin_prog_2 extends b_ub
    b_ub = list(copy.deepcopy(f))
    for k in range(len(X)):
        A_ub.append(_q(X[k:k+1, :]))
    return A_ub, b_ub


def _lin_prog_1(X, f, sample_weight=None):
    # We will refer to the parameter vector as
    # [c0, c, D] (where c is a vector and D is a diagonal matrix) 
    # which is short-hand for 
    # [c_0, c_1, ..., c_n , D_(1,1), ..., D_(n,n)]

    n = X.shape[1]
    
    # q(c0, c, D; x) = c0 + sum_i(c_i*x_i) + 0.5*sum_i(d_(i,i)*x_i**2)
    _q = _q_lp1

    # objective is min_{c0, c, D} sum_k(f^(k) - _q(x).tranpose()*[c0, c, D]
    # but since the sum of f is constant with respect to the parameter vector we 
    # can drop it and use:
    objective = - _q(X, sample_weight)

    # Constraint type 1
    # f^(k) >= _q(x^(k)).transpose()*[c0, c, D]
    A_ub, b_ub = _constraint_1(f, X, _q)
   
    # Constraint type 2
    # D_(j,j) >= 0
    bounds = [(None, None)]*(1 + n) + [(0, None)]*n

    res = linprog(objective, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    return res


def _lin_prog_2(X, f, sample_weight=None):
    # We will refer to the parameter vector as
    # [c0, c, H] (where c is a vector and H is a diagonally dominant matrix) 
    # which is short-hand for 
    # [c_0, c_1, ..., c_n , H_(1,1), ..., H_(n,n), H_(1, 2), ..., H_(1,n), H_(2, 3), ..., H_(2, n), ..., H_(n-1,n)]

    n = X.shape[1]
  
    # q(c0, c, H; x) = c0 + sum_i(c_i*x_i) + 0.5*sum_i(H_(i,i)*x_i**2)
    #                  + sum_j(sum_{i=j+1..n}(H_[i,j]*x_i*x_j))
    _q = _q_lp2

    objective = - _q(X, sample_weight)

    # Constraint type 1
    # f^(k) >= _q(x^(k)).transpose()*[c0, c, H]
    A_ub, b_ub = _constraint_1(f, X, _q)

    # Constraint type 2
    # H_(j,j) >= sum_{i=1..n, i!=j} beta_(i,j)
    for j in range(n):
        A_ub.append(
            [0]*(1 + n)  # no constraints on c
          + [-1 if i == j else 0 for i in range(n)]
          + [0 for k in range(n) for i in range(k+1, n)]  # no constraints on H_(i, j) where i!=j
          + [1 if k==j and i!=j else 0 for k in range(n) for i in range(k+1, n)])
        b_ub.append(0)

    # Constraint type 3
    # H_(i,j) >= -beta_(i,j)
    for j in range(n):
        for i in range(j+1, n):
            A_ub.append(
                [0]*(1 + n)  # no constraints on c
              + [0 for l in range(n)]  # no constraints on H_(j, j)
              + [-1 if j==k and i==l else 0 for k in range(n) for l in range(k+1, n)]  
              + [-1 if j==k and i==l else 0 for k in range(n) for l in range(k+1, n)])
            b_ub.append(0)

    # Constraint type 4
    # H_(i,j) <= beta_(i,j)
    for j in range(n):
        for i in range(j+1, n):
            A_ub.append(
                [0]*(1 + n)  # no constraints on c
              + [0 for l in range(n)]  # no constraints on H_(j, j)
              + [1 if j==k and i==l else 0 for k in range(n) for l in range(k+1, n)]  
              + [-1 if j==k and i==l else 0 for k in range(n) for l in range(k+1, n)])
            b_ub.append(0)

    return linprog(objective, A_ub=A_ub, b_ub=b_ub)


def _L_ij(n):
    return [(i, j) for j in range(n) for i in range(j, n)]


def _q(x, args, eps=1e-3):
    x = np.array(x).squeeze()
    c0, c, l = _args_to_parts(args, len(x))
    H = _hessian(l, len(x), eps=eps)
    return c0 + np.dot(c.reshape((1,-1)), x) + 0.5*np.linalg.multi_dot([x.reshape((1, -1)), H, x.reshape((-1, 1))])


def _args_to_parts(args, n):
    c0 = args[0]
    c = np.array(args[1:n+1])
    l = np.array(args[n+1:])
    return c0, c, l


def _hessian(l, n, eps=1e-3):
    L = np.zeros((n,n))
    for idx, (i, j) in enumerate(_L_ij(n)):
       L[i,j] = l[idx]
    return np.dot(L, L.transpose()) + np.eye(n)*eps


def _nonlin_prog(X, f, x0, eps=1e-3, norm=1, sample_weight=None, lambda_=0):
    n = X.shape[1]

    L_ij = _L_ij(n)

    if sample_weight is not None:
        sample_weight = np.array(sample_weight)
        assert np.all(sample_weight > 0)
    else:
        sample_weight = 1

    objective = lambda args: (np.sum(sample_weight*np.abs(f - np.array([_q(x, args, eps=eps)[0][0]
                                                                        for idx, x in enumerate(X)]).squeeze())**norm)
                              + np.sum(lambda_*np.abs(args))) # Regularization term

    constraints = []
    for idx, x in enumerate(X):
        constraints.append({
            'type': 'ineq',
            'fun': lambda args: f[idx] - _q(x, args, eps=eps)[0][0],
        })

    # Bounds the maximum gradient (not including contribution from the hessian)
    # Is 10 reasonable? Who knows...
    beta_c = max(-np.min(f), 10000)
    # Bounds the maximum eigen value of hessian.
    # is 10 reasonable? Who knows...
    beta_l = 10000
    
    bounds = np.array([(-beta_c, beta_c)]*(n+1)
                    + [(-beta_l, beta_l) if i!=j else (0, beta_l) for (i,j) in L_ij])

    res = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')

    return res
