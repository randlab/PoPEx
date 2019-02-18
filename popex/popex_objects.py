# -*- coding: utf-8 -*-
""" `popex_objects.py` contains the PoPEx-specific class definitions.

Main structure
    - :class:`PoPEx`: Main class for any PoPEx simulation. It contains the
      model chain and any corresponding probability measures.

Sampling definitions
    - :class:`Problem`: Defines the sampling parameters and functions
    - :class:`Learning`: Learning scheme for learning the likelihood values
    - :class:`Prediction`: Defines the prediction parameters and functions

Classes associated to a model type
    - :class:`MType`: (`abstract`) Parent class for each map associated to a
      model type
    - :class:`ContParam`: (inherits from :class:`MType`) Class for each map that
      is associated to the model types but not to categories (e.g.
      `kld[imtype]`, `entropy[imtype]`)
    - :class:`CatMType`: (`abstract`, inherits from :class:`MType`) Parent class
      for each map that is associated to categories
    - :class:`CatProb`: (inherits from :class:`CatMType`) This class is used for
      the representation of probability distributions over categories
      (e.g. `p_cat[imtype]`, `q_cat[imtype]`)
    - :class:`CatParam`: (inherits from :class:`CatMType`) This class is used
      for the representation of categorized parameter values (e.g.
      `model[j][imtype]`)

"""

# -------------------------------------------------------------------------
#   Authors: Christoph Jaeggli, Julien Straubhaar and Philippe Renard
#   Year: 2019
#   Institut: University of Neuchatel
#
#   Copyright (c) 2019 Christoph Jaeggli
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

# General imports
import abc
import numpy as np
import reprlib
from copy import deepcopy
import pickle

# Package imports
import popex.popex_messages as msg


# Main structure
class PoPEx:
    """ Main class for any PoPEx simulation.

    This class is the main object for the PoPEx algorithm. It contains all
    the models, likelihood, log-prior and log-generation information of a PoPEx
    run.

    Parameters
    ----------
    model : list
        List of models

        model[j] : m-tuple
            Tuple of ``MType`` instances
    log_p_lik : ndarray, shape=(nmod,)
        Natural logarithm of likelihood measure
    cmp_log_p_lik : ndarray, shape=(nmod,)
        Boolean indicator whether the log-likelihood value has been computed
        (True) or predicted (False)
    log_p_pri : ndarray, shape=(nmod,)
        Natural logarithm of prior measure value
    log_p_gen : ndarray, shape=(nmod,)
        Natural logarithm of sampling measure value
    ncmax : m-tuple
        Maximum number of conditioning points for each model type
    nc : list
        List of m-tuples

        nc[j] : m-tuple
            Contains the number of conditioning used conditioning points in the
            generation of `model[j]`
    nmtype : int
        Number of model types
    path_res : str
        Path of the results

    """

    def __init__(self,
                 model=None,
                 log_p_lik=np.empty(0),     # Don't change this, because
                                            # sum(...) must be zero initially
                                            # this ensures that we take zero
                                            # hard conditioning for the first
                                            # models
                 cmp_log_p_lik=np.empty(0, dtype=bool),
                 log_p_pri=np.empty(0),
                 log_p_gen=np.empty(0),
                 ncmax=(0,),
                 nc=None,
                 nmtype=1,
                 path_res='~/'):

        # Initialisation of an instance
        if model is None:
            self.model = []
        else:
            self.model = deepcopy(model)
        self.log_p_lik = deepcopy(log_p_lik)
        self.cmp_log_p_lik = deepcopy(cmp_log_p_lik)
        self.log_p_pri = deepcopy(log_p_pri)
        self.log_p_gen = deepcopy(log_p_gen)
        self.ncmax = ncmax
        if nc is None:
            self.nc = []
        else:
            self.nc = deepcopy(nc)
        self.nmtype = nmtype
        self.path_res = path_res

    def __repr__(self):
        rlib = reprlib.Repr()
        rlib.maxlist = 3

        # Define repr string
        class_name = type(self).__name__
        model_str = rlib.repr(self.model)
        log_p_lik_str = rlib.repr(self.log_p_lik)
        cmp_log_p_lik_str = rlib.repr(self.cmp_log_p_lik)
        log_p_pri_str = rlib.repr(self.log_p_pri)
        log_p_gen_str = rlib.repr(self.log_p_gen)
        nc_str = rlib.repr(self.nc)
        return '{}(model={},\nlog_p_lik={},\ncmp_log_p_lik={},\n' \
               'log_p_pri={},\nlog_p_gen={},\nncmax={!r},\nnc={},\n' \
               'nmtype={},\npath_res={})' \
               .format(class_name,
                       model_str,
                       log_p_lik_str,
                       cmp_log_p_lik_str,
                       log_p_pri_str,
                       log_p_gen_str,
                       self.ncmax,
                       nc_str,
                       self.nmtype,
                       self.path_res)

    def __str__(self):
        return self.__repr__()

    def add_model(self, imod, model, log_p_lik, cmp_log_p_lik,
                  log_p_pri, log_p_gen, ncmod):
        """ Appends a new model at the end of the model list and updates the
        measure arrays.


        Parameters
        ----------
        imod : int
            Model index
        model : m-tuple
            Tuple of ``MType`` instances defining a new model
        log_p_lik : float
            Log-likelihood value of model
        cmp_log_p_lik : bool
            Indicates if the log-likelihood has been computed (True) or
            predicted (False)
        log_p_pri : float
            Log-prior value of model
        log_p_gen : float
            Log-generation value of model
        ncmod : m-tuple
            Defines the number of conditioning points that have been used in the
            generation of the model


        Returns
        -------
        None

        """
        # Pickle the model to save memory
        path_mod = 'model/mod{:06d}.mod'.format(imod)
        with open(self.path_res + path_mod, 'wb') as mfile:
            pickle.dump(model, mfile)

        # Add model information to self
        self.model.append(path_mod)
        self.log_p_lik     = np.append(self.log_p_lik, log_p_lik)
        self.cmp_log_p_lik = np.append(self.cmp_log_p_lik, cmp_log_p_lik)
        self.log_p_pri     = np.append(self.log_p_pri, log_p_pri)
        self.log_p_gen     = np.append(self.log_p_gen, log_p_gen)
        self.nc.append(ncmod)

    def insert_model(self, loc, imod, model, log_p_lik, cmp_log_p_lik,
                     log_p_pri, log_p_gen, ncmod):
        """ Inserts a new model at `loc` of the model list and  updates the
        measure arrays.


        Parameters
        ----------
        loc :  int
            Location of the insertion
        imod : int
            Model index
        model : m-tuple
            Tuple of ``MType`` instances defining a model
        log_p_lik : float
            Log-likelihood value of model
        cmp_log_p_lik : bool
            Indicates if the log-likelihood has been computed (True) or
            predicted (False)
        log_p_pri : float
            Log-prior value of model
        log_p_gen : float
            Log-generation value of model
        ncmod : m-tuple)
            Defines the number of conditioning points that have been used in the
            generation of the model


        Returns
        -------
        None

        """
        # Pickle the model to save memory
        path_mod = 'model/mod{:06d}.mod'.format(imod)
        with open(self.path_res + path_mod, 'wb') as mfile:
            pickle.dump(model, mfile)

        # Insert the model and the values at 'loc'
        self.model.insert(loc, path_mod)
        self.log_p_lik     = np.insert(self.log_p_lik,     loc, log_p_lik)
        self.cmp_log_p_lik = np.insert(self.cmp_log_p_lik, loc, cmp_log_p_lik)
        self.log_p_pri     = np.insert(self.log_p_pri, loc, log_p_pri)
        self.log_p_gen     = np.insert(self.log_p_gen, loc, log_p_gen)
        self.nc.insert(loc, ncmod)

    @property
    def nmod(self):
        """ Number of models

        Returns
        -------
        int
            Number of models in `model`

        """
        return len(self.model)


# Problem definitions
class Problem:
    """ Defines the sampling problem that should be addressed by the PoPEx
    method.

    The user must provide function definitions for `generate_m` and
    `compute_log_p_lik`. For the definition of the model space, we can also
    provide 'prior hard conditioning' through the function `get_hd_pri`.
    Optionally, a likelihood learning scheme can be defined in
    `learning_scheme`. Furthermore, one also must define how to compute the
    ratio in the importance sampling weights. For this, the functions
    `compute_log_p_pri` and `compute_log_p_gen` can also be defined manually. If
    they are left empty, the default version that only considers the hard
    conditioning data points is used.


    Parameters
    ----------
    generate_m : function
        Generates a new model m from a set of hard conditioning data.

        `generate_m(hd_param_ind, hd_param_val, imod)`

        Parameters:
            - hd_param_ind : m-tuple
                For each instance in the model tuple, this variable defines the
                hard conditioning INDICES (where to apply HD). `hd_param_ind[i]`
                is an ``ndarray`` of `shape=(nhd_i,)`
            - hd_param_val : m-tuple
                For each instance in the model tuple, this variable defines the
                hard conditioning VALUES (what to imposed). `hd_param_val[i]` is
                an ``ndarray`` of `shape=(nhd_i,)`
            - imod : int
                Model index

        Returns:
            m-tuple
                New model such as `(CatParam_1, ..., CatParam_m)`
    compute_log_p_lik : function
        Computes the natural logarithm of the likelihood of a model. It usually
        runs an expensive forward operation and compares the response to a given
        set of observations.

        `compute_log_p_lik(model, imod)`

        Parameters:
            - model : m-tuple
                Model such as `(CatParam_1, ..., CatParam_m)` (cf. output of
                `generate_m()`)
            - imod : int
                Model index

        Returns:
            float
                Log-likelihood value of the model
    get_hd_pri : function
        Provides the 'prior hard conditioning' that is used in the definition of
        the model space (i.e. parameter values that are known without
        uncertainty).

        `get_hd_pri()`

        Returns
            - hd_pri_ind : m-tuple
                For each instance in the model tuple, this variable defines the
                hard conditioning INDICES.
            - hd_pri_val : m-tuple
                For each instance in the model tuple, this variable defines the
                hard conditioning VALUES.
    compute_log_p_pri : function, optional
        This function computes the log-prior probability of a model that has
        been generated from a given set of hard conditioning data. Note that
        it's definition is OPTIONAL. If it is left undefined, a default
        implementation will be used (see remark below).

        `compute_log_p_pri(model, hd_p_pri, hd_param_ind)`

        Parameters:
            - model : m-tuple
                Model such as `(CatParam_1, ..., CatParam_m)` (cf. output of
                `generate_m()`)
            - hd_p_pri : m-tuple
                Tuple of the hard conditioning probability values for a given
                model. Each probability value describes the prior probability of
                observing the category of the model value at the corresponding
                conditioning location.
                hd_p_pri[i] is an ``ndarray`` of `shape=(nhd_i,)`
            - hd_param_ind : m-tuple
                Hard conditioning indices
        Returns:
            float
                Log-prior probability value

    compute_log_p_gen : function, optional
        This function computes the log-probability of generating a model in the
        PoPEx sampling from a given set of hard conditioning. Note that it's
        definition is OPTIONAL. If it is left undefined, a default
        implementation will be used (see remark below).

        `compute_log_p_gen(model, hd_p_gen, hd_param_ind)`

        Parameters:
            - model : m-tuple
                Model such as `(CatParam_1, ..., CatParam_m)` (cf. output of
                `generate_m()`)
            - hd_p_gen : m-tuple
                Tuple of the hard conditioning probability values for a given
                model. Each probability value describes the prior probability of
                observing the category of the model value at the corresponding
                conditioning location. `hd_p_gen[i]` is an ``ndarray`` of
                `shape=(nhd_i,)`
            - hd_param_ind : m-tuple
                Hard conditioning indices
        Returns:
            float
                Log-generation probability value
    learning_scheme : Learning, optional
        Learning scheme for log_p_lik (concrete sublcass of ``Learning``)
    meth_w_hd : dict, optional
        Defines the method for computing the learning weights that are used in
        the computation of the hard conditioning points (cf.
        :meth:`compute_w_lik`)
    nmtype : int
        Number of model types
    seed : int
        Initial seed


    Notes
    -----

    (1) Let us provide a simple example for hard conditioning data in
        `hd_param_ind` and `hd_param_val`. It is important to note that PoPEx
        does NOT use any parameter locations. They might be defined by the
        user. If so, they have to follow a certain structure. Let the parameter
        locations be such that::

                                        x    y    z
            param_loc[0] = np.array([[0.5, 1.5, 0.5],   # Parameter 0
                                     [0.5, 2.5, 0.5],   # Parameter 1
                                     [0.5, 3.5, 0.5]]   # Parameter 2

        and the parameter indices (in `hd_param_ind`) are for example::

            hd_param_ind[0] = [0, 2]    # Condition parameter 0 and 2

        so we will use `param_loc[0][hd_param_ind[0], :]` for obtaining the
        array::

            np.array([[0.5, 1.5, 0.5],
                      [0.5, 3.5, 0.5]]).

        This array indicates the physical locations where hard conditioning
        should be applied for the model type `0`. Let the parameter values
        (in `hd_param_val`) be given by::

            hd_param_val[0] = np.array([1.2, 2.5]).

        Together with the conditioning locations above, this imposes hard
        conditioning data as follows::

             x     y     z     val
            0.5   1.5   0.5    1.2
            0.5   3.5   0.5    2.5

    (2) Note that it is possible to NOT define `compute_log_p_pri` and
        `compute_log_p_gen`. In this case, a predefined function will be used.
        This predefined implementation assumes that the quantities `p_pri` and
        `p_gen` are only used TOGETHER in the form of a RATIO

            `ratio(m) = rho(m) / phi(m)`.

        In other words, the default functions assume that we are only interested
        in the DIFFERENCE of the log values, i.e.

            `log_p_pri - log_p_gen`,

        and never in the exact values on their own. It is left to the user to
        implement a more suitable computation, whenever the above assumption is
        not sufficient. For more information also consult the theoretical
        description of the PoPEx method.
    (3) It is also possible to NOT define the `learning_scheme`. In this case,
        the log-likelihood value will ALWAYS be computed.

    """

    def __init__(self,
                 generate_m,
                 compute_log_p_lik,
                 get_hd_pri,
                 compute_log_p_pri=None,
                 compute_log_p_gen=None,
                 learning_scheme=None,
                 meth_w_hd=None,
                 nmtype=1,
                 seed=0):

        # Mandatory arguments
        self.generate_m = generate_m
        self.compute_log_p_lik = compute_log_p_lik
        self.get_hd_pri = get_hd_pri

        # Optional arguments
        if compute_log_p_pri is None:
            self.compute_log_p_pri = _default_log_p
        else:
            self.compute_log_p_pri = compute_log_p_pri
        if compute_log_p_gen is None:
            self.compute_log_p_gen = _default_log_p
        else:
            self.compute_log_p_gen = compute_log_p_gen
        self.learning_scheme = learning_scheme

        # Method
        if meth_w_hd is None:
            self.meth_w_hd = {'name': 'exact'}
        else:
            self.meth_w_hd = dict(meth_w_hd)

        # Constants
        self.nmtype    = int(nmtype)
        self.seed      = int(seed)

    def __repr__(self):
        class_name = type(self).__name__
        return '{}(generate_m={},\n' \
               'compute_log_p_lik={},\n' \
               'get_hd_pri={},\n' \
               'compute_log_p_pri={},\n' \
               'compute_log_p_gen={},\n' \
               'learning_scheme={!r},\n' \
               'meth_w_hd={!r},\n' \
               'nmtype={},\n' \
               'seed={})'\
               .format(class_name,
                       self.generate_m.__name__,
                       self.compute_log_p_lik.__name__,
                       self.get_hd_pri.__name__,
                       self.compute_log_p_pri.__name__,
                       self.compute_log_p_gen.__name__,
                       self.learning_scheme,
                       self.meth_w_hd,
                       self.nmtype,
                       self.seed)

    def __str__(self):
        return self.__repr__()


def _default_log_p(model, hd_prob, hd_ind):
    """ `_default_log_p` is a default definition of the function that computes
    the log-probability of sampling a model from a given set of hard
    conditioning data. In this definition, it is assumed that the hard
    conditioning locations and all the model types are INDEPENDENT. From these
    assumptions, it follows that the probability of sampling a model is obtained
    by

        `p = p_1 * ... * p_nmtype`

    where

        `p_i = hd_prob[i][1] * ... * hd_prob[i][nc_i]`.

    Therefore, the log-probability is then given by

        `log_p_gen = sum_i sum_j log(hd_prob[i][j])`

    where `i` runs through the model types and `j` through the conditioning
    points within a given model type.


    Parameters
    ----------
    model : m-tuple
        Tuple of 'MType' instances defining a model
    hd_prob : m-tuple
        Tuple of the hard conditioning probabilities for a given model. Each
        probability value quantifies the probability of observing the
        corresponding parameter category at the this location.

            hd_prob[i] : ndarray, shape=(nc_i,)
                Containing the category probability values
    hd_ind : m-tuple
        Hard conditioning indices (for a more detailed explanation see comments
        in 'generate_m')


    Returns
    -------
    float
        Log-probability of sampling a model

    """
    # Raise all numpy errors for this function
    old_settings = np.seterr(all='raise')

    # Compute default log probability
    log_val = 0.
    for hd_prob_i in hd_prob:
        if hd_prob_i is not None:
            try:
                log_val += np.sum(np.log(hd_prob_i))
            except FloatingPointError as err:
                print(hd_prob_i)
                raise FloatingPointError(err)

    # Set numpy settings back
    np.seterr(**old_settings)
    return log_val


class Learning(abc.ABC):
    """ `Learning` defines an abstract parent class for a learning scheme.

    Let's assume that we want to define a learning scheme that predicts the
    log-likelihood of a model. In this case we define an explicite sub-class
    of ``Learning`` and provide implementations of the methods

        - :meth:`train`
        - :meth:`compute_p_eval_for`
        - :meth:`learn_value_of`

    It is assumed that there is a choice between 'evaluating the exact
    answer' (which is very expensive) or 'predicting the  answer by a machine
    learning scheme' (which should be very fast). The learning scheme undergoes
    the following main steps

        - Update the learning scheme regularly by using the function
          :meth:`trian` (cf. `upd_ls_freq` in `algorithm.run_popex_mp`). Note
          that here you can choose to only use likelihood values that have
          effectively been computed (cf. `PoPEx.cmp_log_p_lik`).
        - For a given instance compute a probability `p in [0,1]` with which
          the log-likelihood is predicted or evaluated (cf.
          :meth:`compute_p_eval_for`) and then the value eventually is predicted
          (cf. :meth:`learn_value_of`).

    Notes
    -----
        In the PoPEx framework this can be used to learn the log-likelihood
        values for each model (=value of interest). In this regard, predicting a
        value rather than computing it can considerably improve the overall
        computational time.
    """

    @abc.abstractmethod
    def train(self, popex):
        """ This method creates a learning scheme.

        The learning scheme can be saved as class parameter.

        Parameters
        ----------
        popex : PoPEx
            PoPEx main structure (cf `popex_objects.PoPEx`)


        Returns
        -------
        None

        """

    @abc.abstractmethod
    def compute_p_eval_for(self, model):
        """ Computes and return a probability value in [0, 1] that determines
        whether a model should be evaluated exactly.


        The two extreme confidence values signify:

            `p=0`: Value can be learned from the learning scheme
            `p=1`: Value should be evaluated exactly.


        Parameters
        ----------
        model : m-tuple
            Tuple of ``Mtype`` instances that define the new model


        Returns
        -------
        float
            Probability value in `[0, 1]`

        """

    @abc.abstractmethod
    def learn_value_of(self, model):
        """ Uses the existing learning scheme to learn the value of interest for
        an instance.


        Notes
        -----
        This function should raise an error if there is no existing learning
        scheme.


        Parameters
        ----------
        model : m-tuple
            Tuple of ``Mtype`` instances that define the new model


        Returns
        -------
        float
            Predicted log-likelihood value

        """


class Prediction:
    """ Defines a prediction that should be computed based on an existing
    PoPEx instance.

    The user must provide function definitions for `compute_pred` that actually
    implements the prediction operator. Note that there is no return value
    expected from that function. Any important result can be saved under

        | <path_res>$
        |  └-- solution
        |      └-- pred_<name>_modXXXXXX


    Parameters
    ----------
    compute_pred : function
        Computes the prediction for a given model.

        `compute_pred(popex, imod)`

        Parameters:
            - popex : PoPEx
                PoPEx main structure (cf :class:`popex.popex_objects.PoPEx`)
            - imod : int
                Model index

        Returns:
            None
    meth_w_pred : dict
        Defines the method used for computing the prediction weights (cf.
        :meth:`popex.utils.compute_w_lik`)
    nw_min : float
        Minimum number of effective weights (= l_0)
    wfrac_pred : float
        Number in (0,1] defining the fraction of the total weight to be used for
        the prediction. If `p=1`, all predictions for any model that has
        non-zero weight is computed. If `p<1` we take the minimum number of
        weight to cover a ratio of `p` of the total sum of weights.
    """

    def __init__(self,
                 compute_pred=None,
                 meth_w_pred=None,
                 nw_min=None,
                 wfrac_pred=1.):
        self.compute_pred = compute_pred

        if meth_w_pred is None:
            self.meth_w_pred = {'name': 'exact'}
        else:
            self.meth_w_pred  = dict(meth_w_pred)
        self.nw_min     = float(nw_min)
        self.wfrac_pred = float(wfrac_pred)

    def __repr__(self):
        class_name = type(self).__name__
        return '{}(compute_pred={},\n' \
               'meth_w_pred={},\n' \
               'nw_min={},\n' \
               'wfrac_pred={})' \
            .format(class_name,
                    self.compute_pred.__name__,
                    self.meth_w_pred,
                    self.nw_min,
                    self.wfrac_pred)

    def __str__(self):
        return self.__repr__()


# Classes associated to a model type
class MType(abc.ABC):
    """ This class is the parent of any quantity associated to a model type.

    Parameters
    ----------
    dtype_val : str
        Type of the ``ndarray`` values (eg. 'int8', 'float32', 'float64', etc)
    param_val : ndarray
        Values associated to the parameters
    """

    def __init__(self, dtype_val='float64', param_val=None):
        # Assuring instance attributes 'param_val'
        if param_val is None:
            self.param_val = np.empty((0,), dtype=dtype_val)
        else:
            self.param_val = np.array(param_val, dtype=dtype_val)

    def __repr__(self):
        rlib = reprlib.Repr()
        rlib.maxlist = 3

        class_name = type(self).__name__
        param_val_str = rlib.repr(self.param_val)
        return '{}({!r}, {})'.format(class_name,
                                     self.param_val.dtype.__str__(),
                                     param_val_str)

    def __str__(self):
        class_name = type(self).__name__
        return '\n{}\n   dtype_val = {!r}\n   param_val = ({} ndarray)\n' \
            .format(class_name,
                    self.param_val.dtype.__str__(),
                    self.param_val.shape)

    @abc.abstractmethod
    def __setattr__(self, key, value):
        """ This method must be implemented in each subclass. The main idea is
        to allow updates for other attributes as well as a dimensionality
        checks.
        """

    @property
    def nparam(self):
        """ Number of parameters.

        Returns
        -------
        int
            Number of values in `param_val`
        """
        return self.param_val.shape[0]


class CatMType(MType):
    """ This class is the abstract parent of any quantity associated to a
    categorical model type.

    Parameters
    ----------
    dtype_val : str
        Type of the ``ndarray`` values (eg. 'int8', 'float32', 'float64', etc)
    param_val : ndarray
        Values associated to the parameters
    categories : list
        List of size `ncat`. Each instance of the list is again a list
        of 2-tuples that define the value range for the category.

        If `categories[i] = [(v_1, v_2), (v_3, v_4)]`, where `v_j` are real
        values, then the category `i` is defined by the union

            `[v_1, v_2) U [v_3, v_4)`

    """

    def __init__(self, dtype_val='float64', param_val=None, categories=None):
        # Assuring instance attributes 'param_val' and 'categories'
        if categories is None:
            self.categories = []
        else:
            # To save memory no copy is produced
            self.categories = categories
        super().__init__(dtype_val, param_val)

    def __repr__(self):
        rlib = reprlib.Repr()
        rlib.maxlist = 3

        class_name = type(self).__name__
        param_val_str = rlib.repr(self.param_val)
        cat_str = rlib.repr(self.categories)
        return '{}({!r}\n,{},\n{})'.format(class_name,
                                           self.param_val.dtype.__str__(),
                                           param_val_str,
                                           cat_str)

    def __str__(self):
        class_name = type(self).__name__
        return '\n{}\n   dtype_val = {!r}\n   categories = {!r}\n' \
               '   param_val = ({} ndarray)\n'\
            .format(class_name,
                    self.param_val.dtype.__str__(),
                    '  |  '.join(
                        ' U '.join(
                            '[{}, {})'.format(self.categories[icat][intvl][0],
                                              self.categories[icat][intvl][1])
                            for intvl in range(len(self.categories[icat])))
                        for icat in range(len(self.categories))),
                    self.param_val.shape)

    @abc.abstractmethod
    def __setattr__(self, key, value):
        """ This method must be implemented in each subclass. The main idea is
        to allow updates for other attributes as well as a dimensionality
        checks.
        """

    @property
    def ncat(self):
        """ Number of categories.

        Returns
        -------
        int
            Number of categories in `categories`
        """
        return len(self.categories)


class ContParam(MType):
    """ This class is used to define a map of continuous values where each value
    is associated to a model parameter (e.g. `entropy`, `kld`, etc.).


    Notes
    -----
        The shape of `param_val` is `(nparam,)`.
    """

    def __setattr__(self, key, value):
        """ The '__setattr__' method is customized for a check of
        dimensionality. In 'ContParam' the attribute 'param_val' must be a
        1-dimensional object compatible to an ndarray.
        """
        if key == 'param_val':
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            if len(value.shape) > 1:
                raise AttributeError(msg.err2002)
        self.__dict__[key] = value


class CatProb(CatMType):
    """ This class is used to define a map of continuous values for each
    category, where each value is associated to a model parameter (e.g. `p_cat`,
    `q_cat`, etc.).


    Notes
    -----
        The shape of `param_val` is `shape=(nparam, ncat)`.
    """

    def __setattr__(self, key, value):
        """ The '__setattr__' method is customized for a check of
        dimensionality. In 'CatProb' the attribute 'param_val' must be a
        2-dimensional object compatible to an ndarray.
        """
        if key == 'param_val':
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            v_shp = value.shape
            if len(v_shp) == 1 and v_shp[0] > 0:
                raise AttributeError(msg.err2003)
            # If len(v_shp) == 1 and v_shp[0] == 0, then param_val was 'None'
            # and has been set to np.empty((0,)) by default
        self.__dict__[key] = value


class CatParam(CatMType):
    """ This class is used to define a categorized 1-dimensional parameter map
    that is associated to a model type (e.g. `model`, etc). The categories of
    each model parameter in `param_val` is indicated in `param_cat`. These
    categories are automatically updated if `param_val` or `categories` change.


    Notes
    -----
        The shape of `param_val` and `param_cat` is `shape=(nparam,)`.

    """

    _nint_cat = 16
    _dtype_cat = 'int' + str(_nint_cat)

    def __init__(self, dtype_val='float64', param_val=None, categories=None):
        # Assuring instance attributes 'param_val', 'categories' and
        # 'param_cat'

        # Check if _dtype_cat is sufficient
        if categories is not None \
                and len(categories) > 2**(self._nint_cat - 1):
            raise AttributeError(msg.err2004(len(categories)))

        super().__init__(dtype_val, param_val, categories)

    def __setattr__(self, key, value):
        """ The '__setattr__' method is customized for updating the private
        '__param_cat' attribute whenever 'param_val' is set.
        """
        update = False
        if key == 'param_val':
            # Avoid wrong definitions
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            if len(value.shape) > 1:
                raise AttributeError(msg.err2002)
            # Check whether 'param_cat' can be updated
            if hasattr(self, 'categories') and len(self.categories) > 0:
                update = True
        elif key == 'categories':
            # Avoid wrong type
            if not isinstance(value, list):
                raise AttributeError(msg.err2005)
            # Check whether 'param_cat' can be updated
            if hasattr(self, 'param_val') and self.param_val is not None:
                update = True

        # Set value by '__setattr__' of 'Dict'
        self.__dict__[key] = value

        # Update 'param_cat' if needed
        if update:
            self._update_param_cat()

    def _update_param_cat(self):
        """ This function is used to update '__param_cat' which indicates the
        category index for each parameter value.

        Returns
        ------
        None

        """
        # Categorize the parameters according to self.categories
        self.__param_cat = np.zeros_like(self.param_val, dtype=self._dtype_cat)
        test_cat = np.zeros_like(self.param_val, dtype=bool)
        for icat, cat in enumerate(self.categories):
            ind = [np.logical_and(self.param_val >= intvl[0],
                                  self.param_val < intvl[1])
                   for intvl in cat]
            cat_ind = np.logical_or.reduce(ind)
            self.__param_cat += cat_ind * icat
            test_cat = np.logical_or(test_cat, cat_ind)

        # Check if all parameters have been categorized
        if not np.all(test_cat):
            ind = np.where(np.logical_not(test_cat))[0]
            raise ValueError(msg.err2001(ind,
                                         self.param_val[ind],
                                         self.categories))

    @property
    def param_cat(self):
        """ Category indicator array.

        Returns
        -------
        ndarray, shape=(nparam,)
            Category indicators of the values in `param_val`.
        """
        return self.__param_cat
