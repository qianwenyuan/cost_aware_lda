import numpy as np
import matplotlib.pyplot as plt
#from benchmarks import branin
import sys
sys.path.append('..')
from bys_opt import BayesianOptimization
#from lda import lda_train
from benchmarks import parabola
from benchmarks import branin
from svm import svm_train

t_f = svm_train

costtype = 2 # 0: linear, 1: cost high near x where tf is minimum, 2: cost low near x where tf is minimum
avg_overhead = 0.5
def virtual_cost(**params):
    return float(params['x0'])**2+float(params['x1'])**2+avg_overhead # (s)
    '''
    a=1
    b=5.1 / (4 * np.pi**2)
    c=5. / np.pi
    r=6
    s=10
    t=1. / (8 * np.pi)
    if costtype==1:
        return 1. / (a * (params['x1'] - b * params['x0'] ** 2 + c * params['x0'] - r) ** 2 + s * (1 - t) * np.cos(params['x0']) + s) + avg_overhead
    elif costtype==2:
        return (a * (params['x1'] - b * params['x0'] ** 2 + c * params['x0'] - r) ** 2 + s * (1 - t) * np.cos(params['x0']) + s) + avg_overhead
    else:
        return np.abs(params["x1"] ** 2 ) + np.abs(params["x0"] ** 2 ) + avg_overhead
    '''

def virtual_reuse_cost(params, hist_params, reuse_dim):
    orig_cost = virtual_cost(**params)
    reuse_costs = []
    for hp in hist_params:
        is_reuse = np.ones(len(params))
        for dim in reuse_dim:
            is_reuse[dim] = (hp[dim] == params[dim])
        if is_reuse.all():
            
            reuse_costs.append(avg_overhead + max(0, orig_cost - virtual_cost(**hp)))
    if len(reuse_costs) <= 0:
        return orig_cost
    return min(orig_cost, min(reuse_costs))


    # avg_overhead = 0.5

  # dim = len(params)
    # key = "x%d" % (dim - 1)
    # cared = params[key]
    # return np.abs(cared) ** 2 + avg_overhead
    # return np.abs(cared) * np.abs(params["x0"]) + avg_overhead
    # int_part = np.ceil(cared)
    # if int_part % 2:
    #     return 0.001
    # else:
    #     return np.abs(params[key]) ** 2 + avg_overhead

def best_value(Y):
    M = np.array(Y)
    maximum = -np.inf
    for i in range(len(Y)):
        maximum = max(maximum, Y[i])
        M[i] = maximum
    return M

def accum_cost(bo):
    x = [dict(zip(bo.space.keys, x)) for x in bo.space.X]
    cost = np.zeros(len(x))
    cost[0] = virtual_cost(**x[0])
    for i in range(len(x)):
        cost[i] = cost[i-1] + virtual_cost(**x[i])
    return cost

def bo_target(f):
    def bo_f(**params):
        # Attention! This function is only a hack trick.
        # The key of **params must be named by convention of "x0, x1, ..., xn"
        dim = len(params)
        x = np.zeros(dim)
        for i in range(dim):
            x[i] = params["x%d" % i]
        return -f(x)
    return bo_f

def run_bo(target_f, pbounds, n_iter=30, init_params=None, cost_max = 0.0):
    # bayesian optimization search
    print
    bo_target_f = bo_target(target_f)
    bo = BayesianOptimization(bo_target_f, pbounds, if_cost=False, cost_function=virtual_cost, cost_max=cost_max)
    init_points = 2
    if init_params is not None:
        bo.init_with_params(init_params)
        bo.maximize(n_iter=n_iter-len(init_params),acq_type='bo')
    else:
        bo.maximize(init_points=init_points, n_iter=n_iter-init_points,acq_type='bo')
    return [accum_cost(bo), best_value(bo.space.Y), bo]

def run_bo_cost(target_f, pbounds, n_iter=30, init_params=None, acq_type='div', cost_max = 0.0):
    print
    bo_target_f = bo_target(target_f)
    # cost = virtual_cost
    bo = BayesianOptimization(bo_target_f, pbounds, if_cost=True, cost_function=virtual_cost, cost_max=cost_max)
    init_points = 2
    if init_params is not None:
        bo.init_with_params(init_params)
        bo.maximize(n_iter=n_iter-len(init_params),acq_type=acq_type)
    else:
        bo.maximize(init_points=init_points, n_iter=n_iter-init_points,acq_type=acq_type)
    return [accum_cost(bo), best_value(bo.space.Y), bo]

def run_ex(target_f, n_iter):
    print
    x = []
    y = []
    acc_cost = 0.0
    for i in range(n_iter):
        maximum = -target_f(i+1) if i==0 else max(maximum, -target_f(i+1))
        y.append(maximum)
        acc_cost = acc_cost + 2.5* (i+1) + 25
        x.append(acc_cost)
        print('x:{}, y:{}\n'.format(x[i], y[i]))
    return [x, y]

pbounds = {
    "x0": (0.1, 10), #x0 represent for n_topics
    #"x1": (1, 4),
    "x1": (0.001, 0.01)
}



def main():
    n_iter = 20
    n_runs = 1

    x=[]
    y=[]
    x_cost_div=[]
    y_cost_div=[]
    x_cost_my=[]
    y_cost_my=[]
    for i in range(n_runs):
        print('\nrun {}\n'.format(i+1))
        init_points = 1
        _bo_target_f = bo_target(t_f)
        _bo = BayesianOptimization(_bo_target_f, pbounds)
        _init_params = _bo.space.random_points(init_points)

        #x_ex, y_ex = run_ex(t_f, n_iter)
        x_bo, y_bo, bo = run_bo(t_f, pbounds, n_iter,_init_params, cost_max=1000.0)
        x_bo_cost_div, y_bo_cost_div, bo_cost_div = run_bo_cost(t_f, pbounds, n_iter, _init_params, 'div', cost_max=1000.0)
        x_bo_cost_divca, y_bo_cost_divca, bo_cost_divca = run_bo_cost(t_f, pbounds, n_iter, _init_params, 'divca', cost_max=1000.0)
        x_bo_cost_my, y_bo_cost_my, bo_cost_my = run_bo_cost(t_f, pbounds, n_iter, _init_params, 'switch', cost_max=1000.0)

        '''
        x.append(i+1)
        y.append(y_bo[-1])
        x_cost_div.append(i+1)
        y_cost_div.append(y_bo_cost_div[-1])
        x_cost_my.append(i+1)
        y_cost_my.append(y_bo_cost_my[-1])
        
        
        for j,_ in enumerate(x_bo):
            if abs(y_bo[j] + 0.397887) <= 0.01:
                x.append(i+1)
                y.append(x_bo[j])
                break
        for j,_ in enumerate(x_bo_cost_div):
            if abs(y_bo_cost_div[j] + 0.397887) <= 0.01:
                x_cost_div.append(i+1)
                y_cost_div.append(x_bo[j])
                break
        for j,_ in enumerate(x_bo_cost_my):
            if abs(y_bo_cost_my[j] + 0.397887) <= 0.01:
                x_cost_my.append(i+1)
                y_cost_my.append(x_bo[j])
                break
        
        '''
        plot_start = init_points
        #plt.plot(x_ex[0:], y_ex[0:], marker="1", label="ex_search")
        plt.plot(x_bo[0:], y_bo[0:], label="bo")
        # plt.scatter(x_bo[plot_start:], bo.space.Y[plot_start:], marker="x", label="bo_tria")
        plt.plot(x_bo_cost_div[0:], y_bo_cost_div[0:], label="bo_cost_aware_div")
        plt.plot(x_bo_cost_divca[0:], y_bo_cost_divca[0:], label="bo_cost_aware_divca")
        plt.plot(x_bo_cost_my[0:], y_bo_cost_my[0:], label="bo_cost_aware_my")
        # plt.scatter(x_bo_cost[plot_start:], bo_cost.space.Y[plot_start:], marker="o", label="bo_cost_aware_trial")
        # plt.ylim(-5, 0)
        plt.legend()
        plt.savefig('./result/svm_cost_aware_BO_compare_tf_{}.png'.format(i))
        plt.close()
       
        '''
        x1 = [i for i in range(len(x_bo)-1)]
        x2 = x1
        y1 = [x_bo_cost_my[i+1]-x_bo_cost_my[i] for i in range(len(x_bo)-1)]
        y2 = [x_bo[i+1]-x_bo[i] for i in range(len(x_bo)-1)]
        plt.plot(x1, y1, marker='+', label='bo_cost_aware')
        plt.plot(x2, y2, marker='x', label='bo')
        plt.legend()
        plt.savefig('./result/branin_cost_aware_BO_compare_cost_{}.png'.format(i))
        plt.close()
        #plt.show()
        
    plt.plot(x[0:], y[0:], marker="x", label="bo")
    # plt.scatter(x_bo[plot_start:], bo.space.Y[plot_start:], marker="x", label="bo_tria")
    plt.plot(x_cost_div[0:], y_cost_div[0:], marker="+", label="bo_cost_aware_div")
    plt.plot(x_cost_my[0:], y_cost_my[0:], marker="^", label="bo_cost_aware_my")
    plt.legend()
    plt.savefig('./result/branin_cost_aware_BO_compare_when_close_to_max_target_value.png')
    plt.close()
    '''

def multi_exps():
    n_iter = 25
    n_runs = 20
    report_interval = 5

    init_points = 1
    _bo_target_f = bo_target(t_f)
    _bo = BayesianOptimization(_bo_target_f, pbounds)
    # _init_params = _bo.space.random_points(init_points)
    _init_params = [_bo.space.random_points(init_points) for _ in range(n_runs)]

    # x_bo_cost, y_bo_cost, bo_cost = run_bo_cost(branin, pbounds, n_iter, _init_params)
    # x_bo, y_bo, bo = run_bo(branin, pbounds, n_iter, _init_params)

    bo_runs = [run_bo(t_f, pbounds, n_iter, _init_params[i]) for i in range(n_runs)]
    bo_cost_runs = [run_bo_cost(t_f, pbounds, n_iter, _init_params[i], acq_type='div') for i in range(n_runs)]
    bo_cost_my_runs = [run_bo_cost(t_f, pbounds, n_iter, _init_params[i], acq_type='others') for i in range(n_runs)]

    plot_start = init_points

    def gen_line(runs):
        x_max = max([max(run[0]) for run in runs])
        x_start = max([min(run[0]) for run in runs])
        xs = np.arange(x_start, x_max, report_interval)

        def cal_max_avg(x):
            max_indexes = [len(list(filter(lambda li: li <= x, run[0]))) for run in runs]
            max_cals = [run[1][max_index - 1] for max_index, run in zip(max_indexes, runs)]
            return np.average(max_cals)

        ys = np.array(list(map(cal_max_avg, xs)))
        return xs, ys

    bo_xs, bo_ys = gen_line(bo_runs)
    bo_cost_xs, bo_cost_ys = gen_line(bo_cost_runs)
    bo_cost_my_xs, bo_cost_my_ys = gen_line(bo_cost_my_runs)

    plt.plot(bo_xs, bo_ys, marker="|", label="bo")
    plt.plot(bo_cost_xs, bo_cost_ys, marker="|", label="bo_cost_aware")
    plt.plot(bo_cost_my_xs, bo_cost_my_ys, marker="|", label="bo_cost_aware_my")
    #plt.plot(bo_xs, bo_ys, label="bo")
    #plt.plot(bo_cost_xs, bo_cost_ys, label="bo_cost_aware")
    plt.legend()
    plt.savefig('./cost_aware_BO_compare_parabola.png')
    plt.show()


if __name__ == '__main__':
    #pass
    main()
    #multi_exps()
