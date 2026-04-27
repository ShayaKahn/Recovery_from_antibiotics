
def create_cohort_dict(keys, special_key, cohort):
    """
    This function creates a dictionary that maps subjects to their corresponding samples,
    excluding the specific subject.
    Inputs:
    keys: list of strings that represent the identifiers of the subjects.
    special_key: string that represents the identifier of the subject to exclude.
    cohort: numpy matrix of shape (# subjects, # species) or list that contains numpy arrays of shape (# species,)
            that contains the samples of the subjects.
    Returns:
    cohort_dict: dictionary that maps subjects to their corresponding samples, excluding the specific subject.
    """
    assert len(keys) == cohort.shape[0], "The number of keys should be equal to the number of subjects."
    # initialize the cohort dictionary.
    cohort_dict = {}
    # iterate over the keys.
    for i, key in enumerate(keys):
        # exclude the specific subject.
        if key != special_key:
            if type(cohort) is not list:
                cohort_dict[key] = cohort[i, :].reshape(1, -1)
            else:
                cohort_dict[key] = cohort[i]
    return cohort_dict