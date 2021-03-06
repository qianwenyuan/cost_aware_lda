from __future__ import print_function
from __future__ import division

import numpy as np
from scipy.optimize import brute
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import inspect
from .helpers import (UtilityFunction, PrintLog, acq_max, ensure_rng)
from .target_space import TargetSpace
#from utils.reuse_DE1 import Reuse_DE


class BayesianOptimization(object):

    def __init__(self, f, pbounds, random_state=None, verbose=1, if_cost=False, cost_function=None, cost_max=0.0):
        """
        :param f:
            Function to be maximized.

        :param pbounds:
            Dictionary with parameters names as keys and a tuple with minimum
            and maximum values.

        :param verbose:
            Whether or not to print progress.

        :param cost_function:
            The cost function of f, proportional to the (appox) running time of f,
            should have exact same parameter list with f.
            This function should fairly cheap to evaluate.

        :param cost_init:
            init points cost, if neccessary.

        """
        # Store the original dictionary
        self.pbounds = pbounds

        self.random_state = ensure_rng(random_state)

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self.space = TargetSpace(f, pbounds, random_state)

        # Initialization flag
        self.initialized = False

        # Initialization lists --- stores starting points before process begins
        self.init_points = []
        self.x_init = []
        self.y_init = []

        # Counter of iterations
        self.i = 0

        # Internal GP regressor
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            n_restarts_optimizer=25,
            random_state=self.random_state,
        )

        # Utility Function placeholder
        self.util = None

        # PrintLog object
        self.plog = PrintLog(self.space.keys)

        # Output dictionary
        self.res = {}
        # Output dictionary
        self.res['max'] = {'max_val': None,
                           'max_params': None}
        self.res['all'] = {'values': [], 'params': []}

        # non-public config for maximizing the aquisition function
        # (used to speedup tests, but generally leave these as is)
        self._acqkw = {'n_warmup': 100000, 'n_iter': 250}

        # Verbose
        self.verbose = verbose

        # Cost function
        self.if_cost = if_cost

        if cost_function is not None:
            assert(inspect.getfullargspec(f).args == inspect.getfullargspec(cost_function).args)
        self.cost_function = cost_function

        self.cost_init = 0.0
        self.cost_max = cost_max
        self.cost_acc = 0.0


    def init(self, init_points):
        """
        Initialization method to kick start the optimization process. It is a
        combination of points passed by the user, and randomly sampled ones.

        :param init_points:
            Number of random points to probe.
        """
        # Concatenate new random points to possible existing
        # points from self.explore method.
        rand_points = self.space.random_points(init_points)
        self.init_with_params(rand_points)

    def init_with_params(self, params):
        def cost(x):
            if self.cost_function is None:
                return 0.0
            x_ = np.asarray(x).ravel()
            params = dict(zip(self.space.keys, x_))
            return self.cost_function(**params)

        self.init_points.extend(params)

        # Evaluate target function at all initialization points
        for x in self.init_points:
            y = self._observe_point(x)
            self.cost_init += cost(x)
        self.cost_acc = self.cost_init

        # Add the points from `self.initialize` to the observations
        if self.x_init:
            x_init = np.vstack(self.x_init)
            y_init = np.hstack(self.y_init)
            for x, y in zip(x_init, y_init):
                self.space.add_observation(x, y)
                if self.verbose:
                    self.plog.print_step(x, y)

        # Updates the flag
        self.initialized = True

    def _observe_point(self, x):
        y = self.space.observe_point(x)
        if self.verbose:
            self.plog.print_step(x, y)
        return y

    def explore(self, points_dict, eager=False):
        """Method to explore user defined points.

        :param points_dict:
        :param eager: if True, these points are evaulated immediately
        """
        if eager:
            self.plog.reset_timer()
            if self.verbose:
                self.plog.print_header(initialization=True)

            points = self.space._dict_to_points(points_dict)
            for x in points:
                self._observe_point(x)
        else:
            points = self.space._dict_to_points(points_dict)
            self.init_points = points

    def initialize(self, points_dict):
        """
        Method to introduce points for which the target function value is known

        :param points_dict:
            dictionary with self.keys and 'target' as keys, and list of
            corresponding values as values.

        ex:
            {
                'target': [-1166.19102, -1142.71370, -1138.68293],
                'alpha': [7.0034, 6.6186, 6.0798],
                'colsample_bytree': [0.6849, 0.7314, 0.9540],
                'gamma': [8.3673, 3.5455, 2.3281],
            }

        :return:
        """

        self.y_init.extend(points_dict['target'])
        for i in range(len(points_dict['target'])):
            all_points = []
            for key in self.space.keys:
                all_points.append(points_dict[key][i])
            self.x_init.append(all_points)

    def initialize_df(self, points_df):
        """
        Method to introduce point for which the target function
        value is known from pandas dataframe file

        :param points_df:
            pandas dataframe with columns (target, {list of columns matching
            self.keys})

        ex:
              target        alpha      colsample_bytree        gamma
        -1166.19102       7.0034                0.6849       8.3673
        -1142.71370       6.6186                0.7314       3.5455
        -1138.68293       6.0798                0.9540       2.3281
        -1146.65974       2.4566                0.9290       0.3456
        -1160.32854       1.9821                0.5298       8.7863

        :return:
        """

        for i in points_df.index:
            self.y_init.append(points_df.loc[i, 'target'])

            all_points = []
            for key in self.space.keys:
                all_points.append(points_df.loc[i, key])

            self.x_init.append(all_points)

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        :param new_bounds:
            A dictionary with the parameter name and its new bounds

        """
        # Update the internal object stored dict
        self.pbounds.update(new_bounds)
        self.space.set_bounds(new_bounds)

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ei',
                 kappa=2.576,
                 xi=0.0,
                 n_jobs=1,
                 acq_type='others',
                 threshhold=0.1,
                 **gp_params):
        """
        Main optimization method.

        Parameters
        ----------
        :param init_points:
            Number of randomly chosen points to sample the
            target function before fitting the gp.

        :param n_iter:
            Total number of times the process is to repeated. Note that
            currently this methods does not have stopping criteria (due to a
            number of reasons), therefore the total number of points to be
            sampled must be specified.

        :param acq:
            Acquisition function to be used, defaults to Expected Improvement.

        :param n_jobs:
            Maximum experiments on f that we can run in parallel.

            Main optimization function with parallel evaluation enabled
            when n_jobs > 1. This method enables parallel evaluation via
            batch Bayesian Optimization, which picks multiple samples per
            iteration. The target function f should be thread-safe to use
            this parallel method.

            Here're some methods available for picking samples according to
            the acquisition function:
            - q-EI method
            - Iterative maximize method

        :param gp_params:
            Parameters to be passed to the Scikit-learn Gaussian Process object

        :param acq_type:
            bo: nomal EI, others:prefer lower cost when similar EI

        :param threshhold:
            when acq_type!=bo

        Returns
        -------
        :return: Nothing

        Example:
        >>> xs = np.linspace(-2, 10, 10000)
        >>> f = np.exp(-(xs - 2)**2) + np.exp(-(xs - 6)**2/10) + 1/ (xs**2 + 1)
        >>> bo = BayesianOptimization(f=lambda x: f[int(x)],
        >>>                           pbounds={"x": (0, len(f)-1)})
        >>> bo.maximize(init_points=2, n_iter=25, acq="ucb", kappa=1)
        """
        # Reset timer
        self.plog.reset_timer()

        # Set acquisition function
        self.util = UtilityFunction(kind=acq, kappa=kappa, xi=xi)

        # Initialize x, y and find current y_max
        if not self.initialized:
            if self.verbose:
                self.plog.print_header()
            self.init(init_points)

        y_max = self.space.Y.max()

        # Set parameters if any was passed
        self.gp.set_params(**gp_params)

        # Find unique rows of X to avoid GP from breaking
        self.gp.fit(self.space.X, self.space.Y)

        # self.candidates = []
        # Finding argmax of the acquisition function.
        def ac(x, gp, y_max):
            x_ = np.asarray(x).ravel()
            params = dict(zip(self.space.keys, x_))
            #print("(1){}\n(2){}\n".format(x_,self.util.utility(x, gp, y_max)))
            #print("{}\n".format(self.util.utility(x, gp, y_max) / (
            #    self.cost_function(**params) if self.cost_function is not None else 1)))
            if self.if_cost is False or acq_type=='switch':
                return self.util.utility(x, gp, y_max)
            return self.util.utility(x, gp, y_max) / (
                    self.cost_function(**params) if acq_type=='div' else self.cost_function(**params)**((self.cost_max-self.cost_acc)/(self.cost_max-self.cost_init)))
            #return self.util.utility(x, gp, y_max) 

        def cost(x):
            x_ = np.asarray(x).ravel()
            params = dict(zip(self.space.keys, x_))
            return self.cost_function(**params)

        x_max = acq_max(ac=ac,
                        gp=self.gp,
                        y_max=y_max,
                        bounds=self.space.bounds,
                        random_state=self.random_state,
                        alpha = (self.cost_max-self.cost_acc)/(self.cost_max-self.cost_init),
                        cost_func=(cost if self.cost_function is not None and acq_type=='switch' else None),
                        **self._acqkw)
        print("acc:{}, init:{}\n".format(self.cost_acc, self.cost_init))
        if self.cost_function is not None:
            self.cost_acc += cost(x_max)
        #print("x_max_before:{}\n".format(x_max))

        #print("x_max_after:{}\n".format(x_max))
        # Print new header
        if self.verbose:
            self.plog.print_header(initialization=False)
        # Iterative process of searching for the maximum. At each round the
        # most recent x and y values probed are added to the X and Y arrays
        # used to train the Gaussian Process. Next the maximum known value
        # of the target function is found and passed to the acq_max function.
        # The arg_max of the acquisition function is found and this will be
        # the next probed value of the target function in the next round.
        for i in range(n_iter):
            # Test if x_max is repeated, if it is, draw another one at random
            # If it is repeated, print a warning
            pwarning = False
            while x_max in self.space:
                x_max = self.space.random_points(1)[0]
                pwarning = True

            # Append most recently generated values to X and Y arrays
            y = self.space.observe_point(x_max)
            if self.verbose:
                self.plog.print_step(x_max, y, pwarning)

            # Updating the GP.
            self.gp.fit(self.space.X, self.space.Y)

            # Update the best params seen so far
            self.res['max'] = self.space.max_point()
            self.res['all']['values'].append(y)
            self.res['all']['params'].append(dict(zip(self.space.keys, x_max)))

            # Update maximum value to search for next probe point.
            if self.space.Y[-1] > y_max:
                y_max = self.space.Y[-1]

            self.candidates = []
            # Maximize acquisition function to find next probing point
            x_max = acq_max(ac=ac,
                            gp=self.gp,
                            y_max=y_max,
                            bounds=self.space.bounds,
                            random_state=self.random_state,
                            alpha = (self.cost_max-self.cost_acc)/(self.cost_max-self.cost_init),
                            cost_func=(cost if self.cost_function is not None and acq_type=='switch' else None),
                            **self._acqkw)
            if self.cost_function is not None:
                self.cost_acc += cost(x_max)
                print("acc:{}\n".format(self.cost_acc))

            if self.cost_acc >= self.cost_max:
                break
            #print("x_max_before:{}\n".format(x_max))

            #print("x_max_after:{}\n".format(x_max))
            # Keep track of total number of iterations
            self.i += 1

        # Print a final report if verbose active.
        if self.verbose:
            self.plog.print_summary()

    def points_to_csv(self, file_name):
        """
        After training all points for which we know target variable
        (both from initialization and optimization) are saved

        :param file_name: name of the file where points will be saved in the csv
            format

        :return: None
        """

        points = np.hstack((self.space.X, np.expand_dims(self.space.Y, axis=1)))
        header = ','.join(self.space.keys + ['target'])
        np.savetxt(file_name, points, header=header, delimiter=',', comments='')

    # --- API compatibility ---

    @property
    def X(self):
        warnings.warn("use self.space.X instead", DeprecationWarning)
        return self.space.X

    @property
    def Y(self):
        warnings.warn("use self.space.Y instead", DeprecationWarning)
        return self.space.Y

    @property
    def keys(self):
        warnings.warn("use self.space.keys instead", DeprecationWarning)
        return self.space.keys

    @property
    def f(self):
        warnings.warn("use self.space.target_func instead", DeprecationWarning)
        return self.space.target_func

    @property
    def bounds(self):
        warnings.warn("use self.space.dim instead", DeprecationWarning)
        return self.space.bounds

    @property
    def dim(self):
        warnings.warn("use self.space.dim instead", DeprecationWarning)
        return self.space.dim

class Discret_BO(BayesianOptimization):
    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ei',
                 kappa=2.576,
                 xi=0.0,
                 n_jobs=1,
                 **gp_params):
        self.plog.reset_timer()

        # Set acquisition function
        self.util = UtilityFunction(kind=acq, kappa=kappa, xi=xi)

        # Initialize x, y and find current y_max
        if not self.initialized:
            if self.verbose:
                self.plog.print_header()
            self.init(init_points)

        y_max = self.space.Y.max()

        # Set parameters if any was passed
        self.gp.set_params(**gp_params)

        # Find unique rows of X to avoid GP from breaking
        self.gp.fit(self.space.X, self.space.Y)

        # Finding argmax of the acquisition function.
        def ac(x, gp, y_max):
            x_ = np.asarray(x).ravel()
            params = dict(zip(self.space.keys, x_))
            return self.util.utility(x, gp, y_max) / (
                self.cost_function(**params) if self.cost_function is not None else 1)

        # x_max = acq_max(ac=ac,
        #                 gp=self.gp,
        #                 y_max=y_max,
        #                 bounds=self.space.bounds,
        #                 random_state=self.random_state,
        #                 **self._acqkw)
        # x_max = brute(lambda x: -ac(x.reshape(1, -1), self.gp, y_max), self.space.bounds, Ns=16, finish=None)

        # Print new header
        if self.verbose:
            self.plog.print_header(initialization=False)
        # Iterative process of searching for the maximum. At each round the
        # most recent x and y values probed are added to the X and Y arrays
        # used to train the Gaussian Process. Next the maximum known value
        # of the target function is found and passed to the acq_max function.
        # The arg_max of the acquisition function is found and this will be
        # the next probed value of the target function in the next round.
        for i in range(n_iter):
            # Test if x_max is repeated, if it is, draw another one at random
            # If it is repeated, print a warning
            x_max = brute(lambda x: -ac(x.reshape(1, -1), self.gp, y_max), self.space.bounds, Ns=151, finish=None)
            pwarning = False
            while x_max in self.space:
                print("!!!! REPEAT!!!!! %s" % x_max.__repr__())
                x_max = self.space.random_points(1)[0]
                pwarning = True

            # Append most recently generated values to X and Y arrays
            y = self.space.observe_point(x_max)
            if self.verbose:
                self.plog.print_step(x_max, y, pwarning)

            # Updating the GP.
            self.gp.fit(self.space.X, self.space.Y)

            # Update the best params seen so far
            self.res['max'] = self.space.max_point()
            self.res['all']['values'].append(y)
            self.res['all']['params'].append(dict(zip(self.space.keys, x_max)))

            # Update maximum value to search for next probe point.
            if self.space.Y[-1] > y_max:
                y_max = self.space.Y[-1]

            # Maximize acquisition function to find next probing point
            # x_max = acq_max(ac=ac,
            #                 gp=self.gp,
            #                 y_max=y_max,
            #                 bounds=self.space.bounds,
            #                 random_state=self.random_state,
            #                 **self._acqkw)

            # Keep track of total number of iterations
            self.i += 1

        # Print a final report if verbose active.
        if self.verbose:
            self.plog.print_summary()

avg_overhead = 0.5
class VirCost:


    def __init__(self, virtual_cost, reuse_dim_name=None, reuse_cost=None):
        self.virtual_cost = virtual_cost
        # 对于通用的reuse模式来说，reuse_dim上的参数必须都相同才能被reuse
        self.reuse_dim_name = reuse_dim_name
        self.params_keys = None
        self.reuse_cost = reuse_cost
        self.dict_hist_params = []

    def set_params_keys(self, params_keys, reuse_dim):
        self.params_keys = params_keys
        self.reuse_dim = reuse_dim
        if self.reuse_cost is None:
            # 如果没有特殊的reuse模式，就使用通用的reuse_cost
            def f(hist_params, params):
                orig_cost = self.virtual_cost(**params)
                reuse_costs = []
                for hp in hist_params:
                    is_reuse = np.ones(len(params))
                    for dim in self.reuse_dim_name:
                        is_reuse[params_keys.index(dim)] = (hp[dim] == params[dim])
                    if is_reuse.all():
                        # 能被warm_start, reuse_cost是从零开始算的cost减去能被warm start掉的cost再加上overhead
                        reuse_costs.append(avg_overhead + max(0, orig_cost - self.virtual_cost(**hp)))
                if len(reuse_costs) <= 0:
                    return orig_cost
                return min(orig_cost, min(reuse_costs))
            self.reuse_cost = f

    def set_hist_params(self, X):
        """

        :param X: a numpy matrix, containing history params values
        :return: virtual cost
        """
        self.dict_hist_params = [dict(zip(self.params_keys, x)) for x in X]

    def add_hist_params(self, x):
        self.dict_hist_params.append(dict(zip(self.params_keys, x)))


    def cost_x(self, x):
        """

        :param x: a numpy array, containing params values
        :return: virtual cost
        """
        dict_params = dict(zip(self.params_keys, x))
        cost = self.reuse_cost(self.dict_hist_params, dict_params)
        return cost

class Reuse_BO(BayesianOptimization):
    def __init__(self, f, pbounds, random_state=None, verbose=1, virCost=None):
        """
        :param f:
            Function to be maximized. The parameters of this function is a dictionary with parameter
            names as keys : f(dict(p1=xxx, p2=xxx))

        :param pbounds:
            Dictionary with parameters names as keys and a tuple with minimum
            and maximum values.

        :param verbose:
            Whether or not to print progress.

        :param virCost:

            VirCost object, containing virtual_cost, reuse_dim and reuse_cost

            The reuse cost function of f, proportional to the (appox) running time of f,
            should have exact same parameter list with f.

            reuse_cost_function(hist_params, params, reuse_dim_name), params is a dictionary


            This function should fairly cheap to evaluate.

        :param reuse_dim:
            ["param_name0", "param_name2", ...], 这些参数必须保持固定才能被reuse，传进来的是param的原参数名
        """
        super().__init__(f, pbounds, random_state, verbose, virCost.virtual_cost)
        self.reuse_dim_name = virCost.reuse_dim_name
        self.reuse_dim = [self.space.keys.index(param_name) for param_name in self.reuse_dim_name]
        virCost.set_params_keys(self.space.keys, self.reuse_dim)
        self.virCost = virCost



        # def reuse_cost_no_key(hist_params, params, reuse_dim=self.reuse_dim):
        #     """
        #
        #     :param hist_params:  a numpy matrix
        #     :param params: a numpy array
        #     :param reuse_dim: e.g. [0, 2, 4]
        #     :return: virtual cost
        #     """
        #     if reuse_cost is None:
        #         return 1
        #     dict_params = dict(zip(self.space.keys, params))
        #     dict_hist_params = [dict(zip(self.space.keys, x)) for x in hist_params]
        #     virtual_cost = reuse_cost.reuse_cost(dict_hist_params, dict_params, self.reuse_dim_name)
        #     return virtual_cost
        #
        # self.reuse_cost_function = reuse_cost_no_key
        self.hist_vir_cost = None
        self.accum_vir_cost = None



    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ei',
                 kappa=2.576,
                 xi=0.0,
                 n_jobs=1,
                 **gp_params):
        self.plog.reset_timer()

        # Set acquisition function
        self.util = UtilityFunction(kind=acq, kappa=kappa, xi=xi)

        # Initialize x, y and find current y_max
        if not self.initialized:
            if self.verbose:
                self.plog.print_header()
            self.init(init_points)

        # 记录init points花的cost

        y_max = self.space.Y.max()

        # Set parameters if any was passed
        self.gp.set_params(**gp_params)

        # Find unique rows of X to avoid GP from breaking
        self.gp.fit(self.space.X, self.space.Y)


        # x_max = acq_max(ac=ac,
        #                 gp=self.gp,
        #                 y_max=y_max,
        #                 bounds=self.space.bounds,
        #                 random_state=self.random_state,
        #                 **self._acqkw)
        # x_max = brute(lambda x: -ac(x.reshape(1, -1), self.gp, y_max), self.space.bounds, Ns=16, finish=None)

        # Print new header


        if self.verbose:
            self.plog.print_header(initialization=False)
        # Iterative process of searching for the maximum. At each round the
        # most recent x and y values probed are added to the X and Y arrays
        # used to train the Gaussian Process. Next the maximum known value
        # of the target function is found and passed to the acq_max function.
        # The arg_max of the acquisition function is found and this will be
        # the next probed value of the target function in the next round.
        for i in range(n_iter):
            # Test if x_max is repeated, if it is, draw another one at random
            # If it is repeated, print a warning
            # x_max = brute(lambda x: -ac(x.reshape(1, -1), self.gp, y_max), self.space.bounds, Ns=151, finish=None)


            # Finding argmax of the acquisition function.
            self.virCost.set_hist_params(self.space.X)
            def ac(x, gp, y_max):
                x_ = np.asarray(x).ravel()
                return self.util.utility(x, gp, y_max) / (
                    self.virCost.cost_x(x_) if self.cost_function is not None else 1)
                # params = dict(zip(self.space.keys, x_))
                # 在acquisition functino里，params是以dict的形式存下来的
                # return self.util.utility(x, gp, y_max) / (
                #     self.reuse_cost_function(self.space.X, params, self.reuse_dim) if self.cost_function is not None else 1)

            rde = Reuse_DE(lambda x: -ac(x.reshape(1, -1), self.gp, y_max), self.space.bounds, self.space.X, reuse_dim=self.reuse_dim)
            (x, y, x_hist, y_hist) = rde.solve()
            x_max = x if y < y_hist else x_hist

            pwarning = False
            while x_max in self.space:
                print("!!!! REPEAT!!!!! %s" % x_max.__repr__())
                x_max = self.space.random_points(1)[0]
                pwarning = True

            # Append most recently generated values to X and Y arrays
            y = self.space.observe_point(x_max)
            if self.verbose:
                self.plog.print_step(x_max, y, pwarning)

            # Updating the GP.
            self.gp.fit(self.space.X, self.space.Y)

            # Update the best params seen so far
            self.res['max'] = self.space.max_point()
            self.res['all']['values'].append(y)
            self.res['all']['params'].append(dict(zip(self.space.keys, x_max)))

            # Update maximum value to search for next probe point.
            if self.space.Y[-1] > y_max:
                y_max = self.space.Y[-1]

            # Maximize acquisition function to find next probing point
            # x_max = acq_max(ac=ac,
            #                 gp=self.gp,
            #                 y_max=y_max,
            #                 bounds=self.space.bounds,
            #                 random_state=self.random_state,
            #                 **self._acqkw)

            # Keep track of total number of iterations
            self.i += 1

        # Print a final report if verbose active.
        if self.verbose:
            self.plog.print_summary()
