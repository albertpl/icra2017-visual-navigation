import numpy as np
import logging
import random
from utils.trajectory import RaggedArray


def choose_action_label_smooth(config, action, epsilon):
    """ epsilon-greedy: P(k) =  (1-epsilon) * P_e +  e * 1/N """
    pi_values = [epsilon/float(config.action_size)] * config.action_size
    pi_values[action] += 1-epsilon
    return pi_values


def choose_action_greedy(pi_values):
    # greedy algorithm since this is supervised learning
    return np.argmax(pi_values, axis=0)


def choose_action(pi_values):
    values = []
    s = 0.0
    for rate in pi_values:
        s += rate
        values.append(s)
    r = random.random() * s
    for i in range(len(values)):
        if values[i] >= r:
            return i
    # fail safe
    return len(values) - 1


def get_key(scopes):
    return '/'.join(scopes)


def discount(r_N_T_D, gamma):
    """
    Computes Q values from rewards.
    q_N_T_D[i,t,:] == r_N_T_D[i,t,:] + gamma*r_N_T_D[i,t+1,:] + gamma^2*r_N_T_D[i,t+2,:] + ...
    """
    assert r_N_T_D.ndim == 2 or r_N_T_D.ndim == 3
    input_ndim = r_N_T_D.ndim
    if r_N_T_D.ndim == 2: r_N_T_D = r_N_T_D[...,None]

    discfactors_T = np.power(gamma, np.arange(r_N_T_D.shape[1]))
    discounted_N_T_D = r_N_T_D * discfactors_T[None,:,None]
    q_N_T_D = np.cumsum(discounted_N_T_D[:,::-1,:], axis=1)[:,::-1,:] # this is equal to gamma**t * (r_N_T_D[i,t,:] + gamma*r_N_T_D[i,t+1,:] + ...)
    q_N_T_D /= discfactors_T[None,:,None]

    # Sanity check: Q values at last timestep should equal original rewards
    assert np.allclose(q_N_T_D[:,-1,:], r_N_T_D[:,-1,:])

    if input_ndim == 2:
        assert q_N_T_D.shape[-1] == 1
        return q_N_T_D[:,:,0]
    return q_N_T_D


def compute_qvals(r, gamma):
    assert isinstance(r, RaggedArray)
    trajlengths = r.lengths
    # Zero-fill the rewards on the right, then compute Q values
    rewards_B_T = r.padded(fill=0.)
    qvals_zfilled_B_T = discount(rewards_B_T, gamma)
    assert qvals_zfilled_B_T.shape == (len(trajlengths), trajlengths.max())
    return RaggedArray([qvals_zfilled_B_T[i,:l] for i, l in enumerate(trajlengths)]), rewards_B_T


def linesearch(f, x, fullstep, expected_improve_rate):
    accept_ratio = .1
    max_backtracks = 10
    fval = f(x)
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        logging.debug("%(expected_improve)f %(actual_improve)f" % locals())
        if ratio > accept_ratio and actual_improve > 0:
            return xnew
    return x


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / (p.dot(z) + 1e-8)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / (rdotr + 1e-8)
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x

