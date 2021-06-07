"""
Run a policy evaluation
"""

import numpy as np

from classification.policies.common import BehaviorPolicyHelper
from classification.policies.common import Policy

PERCENT_TRAIN = 0.5
# RNG = np.random.default_rng(2021)


def compute_direct_method_their(policy: Policy, data_list: list, r_hat_matrix: np.ndarray, loss_index_map: dict):
    """
    Historical (x, a, l_a)
    sum_{(x,a) in Historical} l_a
    1/|test set| * sum_{(x) test set} \hat{r}_{pi(x)}(x)
    """
    feature_list = [t[0] for t in data_list]

    return np.mean([r_hat_row[loss_index_map[policy.predict(feature)]]
                    for feature, r_hat_row in zip(feature_list, r_hat_matrix)])


# TODO: clean up arguments
def compute_direct_method(policy: Policy, data_list: list):
    """
    This one is ours
    """
    dim_list = []
    for t in data_list:
        r_a = 1 - t[1]
        id_pi_a = int(policy.predict(t[0]) == t[2])
        val = r_a * id_pi_a
        dim_list.append(val)

    return np.mean(dim_list)


def compute_inverse_propensity_score(policy: Policy, scarce_data: list, behavior_policy: BehaviorPolicyHelper):
    """Only works for uniform behavior policy.

    Use the scarce interaction data which has the form
    (feature vector, observed loss component, observed action)
    t[0] -- feature vector
    t[1] -- observed loss component
    t[2] -- observed label

    There was a mistake previously
    return np.mean([int(policy.predict(t[0]) == t[2])/(1/num_labels) for t in scarce_data])
    this is wrong because the numerator evaluates to 1 when the policy matches the sampled action,
    instead of when the policy chooses the same action as was sampled.
    """
    ips_list = []
    for index, t in enumerate(scarce_data):
        r_a = 1 - t[1]
        id_pi_a = int(policy.predict(t[0]) == t[2])
        # TODO replace with flexible baseline
        p_hat_a = behavior_policy.p[index][t[2]]
        val = r_a * id_pi_a / p_hat_a
        ips_list.append(val)

    return np.mean(ips_list)


def compute_doubly_robust_estimator(policy: Policy,
                                    scarce_data: list,
                                    loss_index_map: dict,
                                    r_hat_matrix: np.ndarray,
                                    behavior_policy: BehaviorPolicyHelper):
    """
    The formula from the paper where $S$ is the scarce dataset
    \frac{1}{|S|} \sum_{(x,h,a,r_a) \in S} \frac{(r_a - \hat{\rho_a(x)}) \mathbb{I}(\pi(x) = a)}{\hat{p}(a|x,h)}
    + \hat{\rho_{\pi}}
    """
    # try for loop expression to make the computation more easily readable
    dre_list = []
    index = 0
    for t, r_hat_row in zip(scarce_data, r_hat_matrix):
        r_a = 1 - t[1]
        rho_hat_a = r_hat_row[loss_index_map[t[2]]]
        id_pi_a = int(policy.predict(t[0]) == t[2])
        p_hat_a = behavior_policy.p[index][t[2]]
        p_hat_pi_a = r_hat_row[loss_index_map[policy.predict(t[0])]]
        val = ((r_a - rho_hat_a)*id_pi_a)/p_hat_a + p_hat_pi_a
        dre_list.append(val)
        index += 1

    return np.mean(dre_list)


def compute_limited_data_estimator(policy: Policy,
                    scarce_data: list,
                    loss_index_map: dict,
                    r_hat_matrix: np.ndarray):

    lde_list = []
    for t, r_hat_row in zip(scarce_data, r_hat_matrix):
        r_a = 1 - t[1]
        rho_hat_a = r_hat_row[loss_index_map[t[2]]]
        id_pi_a = int(policy.predict(t[0]) == t[2])
        val = ((r_a - rho_hat_a)*id_pi_a) + rho_hat_a
        lde_list.append(val)

    return np.mean(lde_list)


def compute_classification_error(policy: Policy, test_list: list) -> float:
    """Computes the classification error of the policy on the test data.

    A test tuple is (feature_vec, loss_vec, label)
    """
    # compute error percent
    error_list = [int(policy.predict(t[0]) == t[2]) for t in test_list]

    # compute error by element-wise multiplication and summation
    return float(np.mean(error_list))
