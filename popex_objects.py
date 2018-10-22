""" 'popex_objects.py' contains the PoPEx-specific class definitions:

Main structure
    - 'PoPEx':      Main class for any PoPEx simulation. It contains the model
                    chain and any corresponding probability measures.

Problem definitions
    - 'Problem':    Defines the problem parameters and functions
    - 'Learning':   Learning scheme for predicting if it is worth to compute
                    the likelihood
    - 'Prediction': Defines the prediction parameters and functions

Classes associated to a model type
    - 'MType':      (abstract) Parent class for each map associated to a model
                    type
    - 'CatMType'    (abstract, inherits from MType) Parent class for each map
                    that is associated to categories
    - 'CatProb':    (inherits from CatMType) CatProb is a class that is used
                    for probability distributions over categories
    - 'CatParam':   (inherits from CatMType) CatModel is a class that is used
                    for categorized parameter values
"""

# -------------------------------------------------------------------------
#   Authors: Christoph Jaeggli, Julien Straubhaar and Philippe Renard
#   Year: 2018
#   Institut: University of Neuchatel
#
#   Copyright (c) 2018 Christoph Jaeggli
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
    """ This class is the main object for the PoPEx algorithm. It contains all
    the models, likelihood, log-prior and log-generation information of a PoPEx
    run. The measure values are not necessarily normalized.

    INSTANCE ATTRIBUTES
    model:          (list) List of models
        model[i]        (m-tuple) Tuple of 'MType' instances
    log_p_lik:      (nmod, ndarray) Natural logarithm of likelihood measure
    cmp_log_p_lik:  (nmod, ndarray) Boolean indicator whether the
                    log-likelihood value has been computed (True) or predicted
                    (False)
    log_p_pri:      (nmod, ndarray) Natural logarithm of prior measure value
    log_p_gen:      (nmod, ndarray) Natural logarithm of sampling measure value
    ncmax:          (tuple) Maximum number of conditioning points for each
                    model type
    nc:             (list) List of m-tuples
        nc[i]           (tuple) m-tuple containing the number of conditioning
                        points imposed in model[i]
    nmtype:         (int) Number of model types
    path_res:       (str) Path of the results

    INSTANCE PROPERTIES
    nmod:           (int) Number of models

    INSTANCE METHODS
    insert_model(...)   Insert a new model to the PoPEx instance at a given
                        location
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
        """ Appends a new model at the ond of the PoPEx.model list and
        correspondingly updates the probability value arrays.

        :param imod:            (int) Simulation index
        :param model:           (m-tuple) Tuple of 'MType' instances defining
                                a new model
        :param log_p_lik:       (float) Log-likelihood value of model
        :param cmp_log_p_lik:   (bool) Indicates if the log-likelihood has been
                                computed (True) or predicted (False)
        :param log_p_pri:       (float) Log-prior value of model
        :param log_p_gen:       (float) Log-generation value of model
        :param ncmod:           (tuple) m-tuple defining the number of
                                conditioning points
        :return: None
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
        """ Inserts a new model at 'loc' of the PoPEx.model list and
        correspondingly updates the probability arrays.

        :param loc:             (int) Location of the insertion
        :param imod:            (int) Model index
        :param model:           (m-tuple) Tuple of 'MType' instances defining
                                new model
        :param log_p_lik:       (float) Log-likelihood value of model
        :param cmp_log_p_lik:   (bool) Indicates if the log-likelihood has been
                                computed (True) or predicted (False)
        :param log_p_pri:       (float) Log-prior value of model
        :param log_p_gen:       (float) Log-generation value of model
        :param ncmod:           (tuple) m-tuple defining the number of
                                conditioning points
        :return: None
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

        :return:            (int) Number of models in 'self.model'.
        """
        return len(self.model)


# Problem definitions
class Problem:
    """ Defines the problem that should be addressed by the PoPEx method. The
    user must provide function definitions for 'generate_m' and
    'compute_log_p_lik'. Optionally, a learning scheme can be defined in
    'learning_scheme'.

    Optionally

    INSTANCE ATTRIBUTES
    generate_m          (callable) Generates a new model m
    compute_log_p_lik   (callable) Computes the natural logarithm of the
                        likelihood of m
    get_hd_pri          (callable) Assembles the 'prior hard conditioning'

    ------------------------------ (optional) ---------------------------------
    compute_log_p_pri   (callable) Computes quantity for the ratio
                        corresponding to the log-prior probability of m
    compute_log_p_gen   (callable) Computes the quantity of the ratio
                        corresponding to the log-generation probability of m
            (for more details about the callable instance variables see
            'INSTANCE ATTRIBUTE DETAILS AND RECOMMENDATIONS' below)
    learning_scheme     (Learning) Learning scheme for log_p_lik
    ---------------------------------------------------------------------------
    meth_w_hd:          (dict) Defines the method used in the
                        computation of the hard conditioning points (HD)
                        (see 'utils.compute_w_lik')
    nmtype              (int) Number of model types
    seed                (int) Initial seed

    INSTANCE ATTRIBUTE DETAILS AND RECOMMENDATIONS
    (1) generate_m(hd_param_ind, hd_param_val, imod)
    ------------------------------------------------
        This function generates a model instance m_k being a m-tuple such that

                m_k = (CatParam_k1, ..., CatParam_km).


        param: hd_param_ind:    (m-tuple) For each instance in the model tuple,
                            this variable defines the hard conditioning INDICES
                            according to the parameter locations.
                hd_param_ind[i] (nhd_i, ndarray) Defining the locations indices
                                where the hard conditioning should be applied.
                EXAMPLE:
                |   It is important to note that PoPEx does NOT use any
                |   parameter locations, but they are only defined by the user.
                |   Nevertheless, they have to follow a certain structure, so
                |   let the parameter locations be such that
                |
                |                                  x    y    z
                |       param_loc[0] = np.array([[0.5, 1.5, 0.5],
                |                                [0.5, 2.5, 0.5],
                |                                [0.5, 3.5, 0.5]]
                |
                |   and the parameter indices (in 'hd_param_ind') are for
                |   example
                |
                |       hd_param_ind[0] = [0, 2]
                |
                |   so we will use param_loc[0][hd_param_ind[0], :] for
                |   obtaining the array
                |
                |           np.array([[0.5, 1.5, 0.5],
                |                     [0.5, 3.5, 0.5]]).
                |
                |   This array indicates WHERE a hard conditioning should be
                |   applied for the model type 0.
                |
        param: hd_param_val:    (m-tuple) For each model type, this variable
                                defines the hard conditioning VALUES.
                hd_param_val[i] (nhd_i, ndarray) Defining the hard conditioning
                                values that should be imposed in a new model.
                EXAMPLE (continuation):
                |   let the parameter values (in 'hd_param_val') be given by
                |
                |       hd_param_val[0] = np.array([1.2, 2.5]).
                |
                |   Together with the first part of the example (above), for
                |   the model type 0, this would impose hard conditioning data
                |   as follows:
                |
                |             x     y     z     val
                |            0.5   1.5   0.5    1.2
                |            0.5   3.5   0.5    2.5
                |
        param: imod:        (int) Model index

        return:             (m-tuple) Tuple of CatModel instances.

    (2) compute_log_p_lik(model, imod)
    ----------------------------------
        This function computes the natural logarithm of the likelihood of a
        given model. It usually runs the forward operator and compares the
        response to a given set of observations.

        --------------------------- !!! CAUTION !!! ---------------------------
        If you choose to compute the train the PoPEx algorithm (i.e. derive a
        set of hard conditioning data) according to the log-likelihood values
        (rather than the likelihood values) you must make sure to only return
        NON-POSITIVE log-likelihood values
        -----------------------------------------------------------------------

        param: model:   (m-tuple) Tuple of CatParam() instances.
        param: imod:    (int) Model index
        return:         (float64) Log-likelihood value of 'model'

    (3) get_hd_pri()
    ----------------
        This function gets the prior hard conditioning of the problem (i.e.
        parameter values that are known without uncertainty).

        return:         (2-tuple) (hd_pri_ind, hd_pri_val)
            return[1]   (m-tuple) For each instance in the model tuple, this
                        variable defines the prior hard conditioning indices
                        according to the parameter locations.
            return[2]   (m-tuple) For each instance in the model tuple, this
                        variable defines the prior hard conditioning values.

            For an example of the hard conditioning structure, see the comments
            in 'generate_m'.

    (4) compute_log_p_pri(model, hd_p_pri, hd_param_ind)
    ----------------------------------------------------
        This function computes the log-prior probability of a model that has
        been generated from a given set of hard conditioning data. Note that
        it's definition is OPTIONAL. If it is left undefined, a default
        implementation will be used (see remark below).

        param: model:       (m-tuple) Tuple of Model instances
        param: hd_p_pri:    (m-tuple) Tuple of the hard conditioning
                            probability values for a given model. Each
                            probability value describes the prior probability
                            of observing the category of the model value at the
                            corresponding conditioning location.
                hd_p_pri[i] (nc_i, ndarray) Containing the category probability
                            values.
        param: hd_param_ind:(m-tuple) Hard conditioning indices (for a more
                            detailed explanation see comments in
                            'generate_m')
        return:             (float) Log-prior probability value

    (5) compute_log_p_gen(model, hd_p_gen, hd_param_ind)
    ----------------------------------------------------
        This function computes the log-probability of generating a model in the
        PoPEx sampling from a given set of hard conditioning. Note that it's
        definition is OPTIONAL. If it is left undefined, a default
        implementation will be used (see remark below).

        param: model:       (m-tuple) Tuple of Model instances
        param: hd_p_gen:    (m-tuple) Tuple of the hard conditioning
                            probability values for a given model. Each
                            probability value describes the weighted
                            probability of observing the category of the model
                            value at the corresponding conditioning location.
                hd_p_gen[i] (nc_i, ndarray) Containing the category probability
                            values
        param: hd_param_ind:(m-tuple) Hard conditioning indices (for a more
                            detailed explanation see comments in
                            'generate_m')
        return:             (float) Log-probability of generating a model


    ---------------------------- !!! ATTENTION !!! ----------------------------
    (1) Note that it is possible to NOT define 'compute_log_p_pri' and
        'compute_log_p_gen'. In this case, a predefined function will be used.
        This predefined implementation assumes that the quantities 'p_pri' and
        'p_gen' are only used TOGETHER in the form of a RATIO

            rho(m) / phi(m).

        In other words, it assumes that we are only interested in the
        DIFFERENCE of the log values, i.e.

            log_p_pri - log_p_gen,

        and never in the exact values on their own. It is left to the user to
        implement a more suitable computation, whenever the above assumption is
        not sufficient. For more information also consult the theoretical
        description of the PoPEx method.
    (2) It is also possible to NOT define the 'learning_scheme'. In this case,
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
    """ _DEFAULT_LOG_P(...) is a default definition of a function that computes
    the log-probability of generating a model from a given set of hard
    conditioning data. In this definition, it is assumed that the hard
    conditioning locations and all the model types are INDEPENDENT. From these
    assumptions, it follows that the (prior or PoPEx) probability of sampling
    a model m is obtained by

        p = p_1 * ... * p_nmtype

    where

        p_i = hd_prob[i][1] * ... * hd_prob[i][nc_i].

    Therefore, the log-probability is then given by

        log_p_gen = sum_i sum_j log(hd_prob[i][j]).

    :param model:       (m-tuple) Tuple of 'MType' instances
    :param hd_prob:     (m-tuple) Tuple of the hard conditioning probabilities
                        for a given model. Each probability value describes the
                        probability of observing the model category at the
                        corresponding location.
            hd_prob[i]  (nc_i, ndarray) Containing the category probability
                        values
    :param hd_ind:      (m-tuple) Hard conditioning indices (for a more
                        detailed explanation see comments in 'generate_m')
    :return             (float) Log-probability of generating a model
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
    """ Defines a learning scheme that is dedicated to particular training
    sets. It is supposed that there is a choice between 'evaluating the exact
    answer' (which is potentially very expensive) or 'predicting the
    answer by a machine learning scheme' (which should be very fast). The
    function 'compute_p_eval_for' computes a probability that expresses if the
    value should be evaluated exactly. The two extreme probabilitys thus
    signify:

        0: Value will be predicted
        1: Value must be evaluated exactly.


    INSTANCE METHODS
    train(...)          (abstract) Trains the learning scheme
    compute_p_eval_for(...) (abstract) Returns the probability with which the
                        value of a given model is evaluated exactly
    learn_value_of(...) (abstract) Predicts the value of a model based on the
                        existing learning scheme
    """

    @abc.abstractmethod
    def train(self, data_set):
        """ This method creates a learning scheme.

        :return: None
        """

    @abc.abstractmethod
    def compute_p_eval_for(self, instance):
        """ Computes and return a probability value in [0, 1] that determines
        whether a model should be evaluated exactly. The two extreme confidence
        values signify the following:

            0: Value can be learned from the learning scheme
            1: Value should be evaluated exactly.


        :param instance:    ( ? ) Instance corresponding to the data set
        :return:            (float) Probability value in [0, 1]
        """

    @abc.abstractmethod
    def learn_value_of(self, instance):
        """ Uses the existing learning scheme to learn the value of interest
        corresponding to a given model. This function should raise an error if
        there is no existing learning scheme.

        :param instance:    ( ? ) Instance corresponding to the data set
        :return:            ( ? ) Predicted value
        """


class Prediction:
    """ Defines a prediction that should be computed based on an existing
    PoPEx instance.

    INSTANCE ATTRIBUTES
    compute_pred        (callable) Runs the prediction of a model m
        (for more details about the callable instance variables see
        'INSTANCE ATTRIBUTE DETAILS AND RECOMMENDATIONS' below)
    meth_w_hd:          (dict) Defines the method used for weighting the
                        predictions (see 'utils.compute_w_lik')
    nw_min              (float) Minimum number of effective weights
    wfrac_pred          (float) Number in (0,1] defining the fraction of the
                        total weight to be used for the prediction

    INSTANCE ATTRIBUTE DETAILS AND RECOMMENDATIONS
    (1) compute_pred(popex, imod)
    -----------------------------
        This function computes the prediction of a given model. There is no
        return value expected so that the function must save important results
        at

        param: popex:   (PoPEx) (see 'popex_objects.PoPEx')
        param: imod:    (int) Simulation index
        return: None
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
        self.nw_min       = float(nw_min)
        self.wfrac_pred   = float(wfrac_pred)

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

    INSTANCE ATTRIBUTES
    param_val:      (nparam x ? ndarray) Values associated to the parameters

    INSTANCE PROPERTIES
    nparam:         (int) Number of parameters
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

        :return:        (int) Number of values in 'param_val'
        """
        return self.param_val.shape[0]


class CatMType(MType):
    """ This class is the parent of any quantity associated to a model type and
    some categories.

    INSTANCE ATTRIBUTES
    param_val:      (nparam x ? ndarray) Values associated to the parameters
    categories:     (list) List of size ncat. Each item is a list of 2-tuples
                    that define the value range for the category.
                    EXAMPLE: If categories[i] = [(v1, v2), (v3, v4)], where vi
                    are real numbers, then the category i is defined by all the
                    values contained in the union
                            [v1, v2) U [v3, v4)

    INSTANCE PROPERTIES
    nparam:         (int) Number of parameters
    ncat:           (int) Number of categories
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

        :return:        (int) Number of categories in 'categories
        """
        return len(self.categories)


class ContParam(MType):
    """ This class is used to define a continuous 1-dimensional parameter map
    that is associated to a model type.

    INSTANCE ATTRIBUTES
    param_val:      (nparam, ndarray) 1-dimensional parameter array

    INSTANCE PROPERTIES
    nparam:         (int) Number of parameters in self.param_val
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
    """ This class is used to define multiple parameter maps that are
    associated to categories within a given model type.

    INSTANCE ATTRIBUTES
    param_val:      (nparam x ncat ndarray) Values associated to the parameters
                    and the categories.
    categories:     (list) List of size ncat. Each item is a list of 2-tuples
                    that define the value range for the category.
                    EXAMPLE: If categories[i] = [(v1, v2), (v3, v4)], where vi
                    are real numbers, then the category i is defined by all the
                    values contained in the union
                            [v1, v2) U [v3, v4)

    INSTANCE PROPERTIES
    nparam:         (int) Number of parameters
    ncat:           (int) Number of categories
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
    that is associated to a model type. At the same time, a category indicator
    array will be generated. This array can be used to learn the category of
    each value in param_val.

    INSTANCE ATTRIBUTES
    param_val:      (nparam, ndarray) Values associated to the parameters
    categories:     (list) List of size ncat. Each item is a list of 2-tuples
                    that define the value range for the category.
                    EXAMPLE: If categories[i] = [(v1, v2), (v3, v4)], where vi
                    are real numbers, then the category i is defined by all the
                    values contained in the union
                            [v1, v2) U [v3, v4)

    INSTANCE PROPERTIES
    param_cat:      (nparam, ndarray) Category index of the values in
                    'param_val'
    nparam:         (int) Number of parameters
    ncat:           (int) Number of categories
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

        :return:        (nparam, ndarray) Category index of the values in
                        'param_val'
        """
        return self.__param_cat
