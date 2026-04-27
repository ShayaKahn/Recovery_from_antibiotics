import numpy as np
import operator

def subset(post_matrix, base_sample, ABX_sample, strict, new):
    num_timepoints = post_matrix.shape[0]
    op_lst = [operator.ne] * num_timepoints
    if new:
        general_cond = [base_sample == 0, ABX_sample == 0, post_matrix[-1, :] != 0]
    else:
        general_cond = [base_sample != 0, ABX_sample == 0, post_matrix[-1, :] != 0]
    timepoints_vals = []
    if strict:
        for j in range(num_timepoints):
            inter_cond = [op_lst[i](post_matrix[i, :], 0) for i in range(j + 1)]
            special_cond = [op_lst[i](post_matrix[i, :], 0) for i in range(j + 1, num_timepoints)]
            cond = general_cond + inter_cond + special_cond
            timepoints_vals.append(np.logical_and.reduce(cond))
            if j != num_timepoints - 1:
                op_lst[j] = operator.eq
    else:
        # iterate over the time points.
        for j in range(num_timepoints):
            # define the intermediate condition.
            inter_cond = [op_lst[i](post_matrix[i, :], 0) for i in range(j + 1)]
            # combine the conditions.
            cond = general_cond + inter_cond
            # find the returned species at each time point.
            timepoints_vals.append(np.logical_and.reduce(cond))
            if j != num_timepoints - 1:
                # update the operators list.
                op_lst[j] = operator.eq
    return timepoints_vals
