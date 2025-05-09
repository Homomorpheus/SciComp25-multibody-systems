"""
ODE and DAE solvers for this project.
"""

import numpy as np
from scipy.optimize import root
import warnings
import sympy


def newmark(q0: np.ndarray, v0: np.ndarray, M: np.ndarray, C: np.ndarray, force, steps, t_end, callback=None):
    """
    Newmark method.
    For documentation on the algorithm, see:
        https://jschoeberl.github.io/IntroSC/ODEs/mechanical.html#the-newmark-method
        https://miaodi.github.io/finite%20element%20method/newmark-generalized/
    Input:
        q0 -- starting position
        v0 -- starting velocity
        M  -- mass matrix
        C  -- damping matrix
        force -- force, a function of q (position)
        steps -- amount of ODE/DAE solver iterations
        t_end -- end time
        callback -- function of step-index and current (q,v,a), gets called every step
    Output:
        last (q,v,a), step-number calculated
    """
    
    h = t_end/steps
    dim = len(q0)
    #starting value for a
    a0 = np.zeros(dim)

    # state from before the timestep
    old = np.concatenate([q0, v0, a0])

    # implicit solver equations
    # new is qnew, vnew, anew; each of length dim
    def eqs(new: np.array):
        qnew, vnew, anew = np.split(new, len(new)/dim)

        eq_eval = np.concatenate([(M@anew.transpose()).transpose() + (C@vnew.transpose()).transpose() - force(qnew, vnew),
                                  vnew - vold - h/2*(anew + aold),
                                  qnew - qold - h*vold - h**2/4*(anew + aold)
                                  ])
        return eq_eval
        
    for step in range(1,steps+1):
        # solve the solver's equations
        qold, vold, aold = np.split(old, len(old)/dim)
        sol = root(eqs, old)

        if not sol.success:
            if sol.status == 5:
                warnings.warn(sol.message, RuntimeWarning)
            else:
                callback(step, old, error=True)
                raise RuntimeError(F"fscipy.optimize.root did not converge (message: {sol.message})")
        
        old=sol.x
        
        if callback != None:
            terminate = callback(step, old, error=False)

        if terminate == True:
            break
            
    return old, step