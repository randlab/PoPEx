""" 'popex_messages.py' contains a collection of messages that are used for
warnings and errors appearing in the popex package.
"""

# ------------------------------- 1XXX utils.py -------------------------------
# Warnings
warn1001 = "\nWARNING 1001: The sum of the weights was equal to 0. The " \
           "weights have been redefined to be uniformly equal to 1/n.\n"


def warn1002(imtype):
    return "\nWARNING 1002: For the model type {} a location x has been " \
           "detected where q_i(x) = 0 and p_i(x) > 0.\nIf q was computed " \
           "from a set of N models, try to increase N.\n".format(imtype)


warn1003 = "\nWARNING 1003: The sum of the kld was equal to 0. The " \
           "values have been redefined to be uniformly equal to 1/n.\n"


warn1004 = "\nWARNING 1004: The sum of the likelihood was equal to 0. The " \
           "values have been redefined to be uniformly equal to 1/n.\n"


# Errors
err1001 = "\nERROR 1001: Negative weight(s) detected."


def err1002(imtype, ind):
    return "\nERROR 1002: Negative Kullback-Leibler-Divergence detected for " \
           "model type {} at location index {}.\n".format(imtype, ind)


def err1003(imod):
    return "\nERROR 1003: HD-Weight value(s) {} is/are negative.".format(imod)


def err1006(first, last):
    return "\nERROR 1006: The set of models [{}, {}) based on which the" \
           "category probability should be computed is empty."\
           .format(first, last)


def err1007(first, last):
    return '\nERROR 1007: The weights between {} and {} have not properly ' \
           'been normalized'.format(first, last)


err1008 = '\nERROR 1008: Probability value at hard conditioning location is 0'


err1009 = '\nERROR 1009: Wrong length of HD parameter indices.'


def err1010(ind):
    return "\nERROR 1010: The log-values {!r} are positive.".format(ind)


err1011 = "\nERROR 1011: The input parameter 'm_new' in " \
          "'utils.update_cat_prob' must be of type 'list'\n"

err1012 = "\nERROR 1012: The input parameter 'w_new' in " \
          "'utils.update_cat_prob' must be of type 'np.ndarray'\n"

err1013 = "\nERROR 1013: The input parameters 'm_new' and 'w_new' in " \
          "'utils.update_cat_prob' must have same length\n"

# ---------------------------- 2XXX popex_objects -----------------------------
# Warnings


# Errors
def err2001(ind, vals, cat):
    return '\nERROR 2001: param_val{} = {} was/were not properly ' \
           'categorized by the categories in {!r}.'.format(ind, vals, cat)


err2002 = "\nERROR 2003: Attribute 'param_val' must be a 1 dimensional array" \
          " or 'None'."


err2003 = "\nERROR 2004: Attribute 'param_val' must be a 2 dimensional array" \
          " or 'None'."


def err2004(ncat):
    return "\nERROR 2004: Too many categories ({})! Redefine default numpy " \
           "type of 'param_cat' in source code of " \
           "'popex.popex_objects.CatParam'"


err2005 = "\nERROR 2005: Attribute 'categories' must be a list."

# ------------------------------- 3XXX problem --------------------------------
# Warnings


# Errors
err3001 = "\nERROR 3001: Average value of the KLD map is larger then KLD_MAX."


# ------------------------------ 4XXX algorithm -------------------------------
# Warnings


# Errors
def err4001(imod):
    return "\nERROR 4001: Likelihood value(s) {!r} is/are negative."\
           .format(imod)


def err4002(imod, val):
    return "\nERROR 4002: Log-prior probability value {} is equal to {}."\
           .format(imod, val)


def err4003(imod, val):
    return "\nERROR 4003: Log-generation probability value {} is equal to {}."\
           .format(imod, val)


def err4004(imod, n_is, n_should):
    return '\nERROR 4004: Model {:d} defines {:d} instead of {:d} model types'\
           .format(imod, n_is, n_should)


def err4005(imod, imtype, type_i):
    return "\nERROR 4005: Wrong instance in model {:d}. All models types " \
           "must subclass 'MType', but type(m[{:d}] = {!r}"\
           .format(imod, imtype, type_i)


def err4006(imtype, type_i):
    return "\nERROR 4006: Model instances of type {!r} can not be " \
           "conditioned. Put ncmax[{:d}] equal to 0.".format(type_i, imtype)


def err4007(imod, hd_meth):
    return "\nERROR 4007: Log-likelihood value of model {} is positive, what" \
           " is not possible for 'hd_meth' = {}."\
           .format(imod, hd_meth)


def err4008(imtype, map_type):
    return "\nERROR 4008: The prior probability map in 'q_cat[{}]' is of " \
           "type {!r}.".format(imtype, map_type)


# ----------------------------- 5XXX predictions ------------------------------
# Warnings


# Errors
def err5001(sflag):
    return "\nERROR 4001: Not implemented save option for flag = {}"\
           .format(sflag)
