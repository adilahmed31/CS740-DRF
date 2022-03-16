import numpy as np
import cvxpy as cp
import sys
import math
import enum
import pandas as pd
from scipy.optimize import minimize
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import argparse

class Algorithm(enum.Enum):
    drf = 1
    af = 2
    ceei = 3


class Task(object):
    '''
    Denotes a single task that can be fed to the scheduler to add into its queue.

    Initializes demand and weight vector for each task
    '''
    def __init__(self, task_id, demands, weights=None):
        self.task_id = task_id
        self.demands = demands
        if weights is None:
            self.weights = [1]*len(demands)
        else:
            self.weights = weights
        self.cpu = demands[0]
        self.memory = demands[1]

class Scheduler(Task):
    '''
    Denotes a Scheduler that can be fed tasks to allocate amongst contending users.

    '''
    def __init__(self, capacities, algorithm = None):
        '''
        Initialize capacities vector, demands vector and set algorithm to be used for allocation.
        '''
        self.capacities = capacities
        self.num_users = 0
        self.demands = []
        self.num_resources = len(self.capacities)
        self.capacities = np.array(self.capacities, dtype=float)
        self.consumed = np.zeros((self.num_users, self.num_resources))
        self.allocations = np.zeros((self.num_users, self.num_resources))
        self.weights = []
        if algorithm is None:
            self.algorithm = Algorithm.drf
        else:
            self.algorithm = algorithm

    def insert_task(self, task):
        '''Insert a task into the scheduling queue. Increments the overall number of users and adds the new task's demand vectors and weights'''
        self.num_users += 1
        self.demands.append(task.demands)
        self.weights.append(task.weights)

    def compute_dominant_shares(self):
        '''Compute dominant shares as defined in the DRF paper for implementing the DRF algorithm. Does not return a vlue - simply updates the class object'''
        self.dominant_shares = np.zeros((self.num_users))
        resource_shares = self.demands / self.capacities
        self.dominant_resources = np.argmax(resource_shares, axis=0)
        for user in range(self.num_users):
            self.dominant_shares[user] = np.max(self.weights[user] * (self.demands[user] / self.capacities))
      
    def compute_fair_shares(self):
        '''Compute fair shares as defined in the DRF paper for implementing the Asset fairness algorithm. Does not return a vlue - simply updates the class object'''
        self.fair_shares = np.zeros((self.num_users))
        self.normalized_capacities = [share/(self.capacities[0]) for share in self.capacities]
        self.normalized_capacities = [1/i for i in self.normalized_capacities]
        self.normalized_capacities = np.array(self.normalized_capacities)

        demands = np.array(self.demands)
        for user in range(self.num_users):
            self.fair_shares[user] = np.matmul(self.weights[user] * demands[user],self.normalized_capacities)
    
    def solver(self):
        ''' Solver for the Nash Bargaining Solution for the CEEI algorithm implementation. returns the allocation vector'''
        X = cp.Variable(len(self.demands),"X")
        demands = np.array(self.demands)
        constraints = [demands.T@X <= self.capacities ,X>=0]

        prob = cp.Problem(cp.Maximize(cp.sum(cp.log(X))), constraints)
        prob.solve()
        return X.value

    def allocate(self):
        '''Function to allocate resources after all tasks have been fed into the scheduler'''
        if self.algorithm == Algorithm.drf:
            self.compute_dominant_shares()
            resource_shares = self.dominant_shares
        elif self.algorithm == Algorithm.af:
            self.compute_fair_shares()
            resource_shares = self.fair_shares
        elif self.algorithm == Algorithm.ceei:
            return self.solver()
        
        normalized_resource_shares = [share/(resource_shares[0]) for share in resource_shares]
        normalized_resource_shares = [1/i for i in normalized_resource_shares]
        var_1 = sys.maxsize
        for i, resource in enumerate(self.capacities):
            max_resource = 0
            for j in range(len(self.demands)):
                max_resource += self.demands[j][i]*normalized_resource_shares[j]
                
            var_1 = min(var_1, resource/max_resource)
        
        all_vars = [i*var_1 for i in normalized_resource_shares]
        return all_vars

    def plot(self,labels):
        '''Internally calls the allocation function and plots a bar graph to show the allocations'''
        tasks_distribution = self.allocate()
        tasks_allocation = np.transpose(self.demands) * np.array(tasks_distribution)
        tasks_allocation = tasks_allocation.tolist()
        tasks_allocation[0].append(self.capacities[0]-sum(tasks_allocation[0]))
        tasks_allocation[1].append(self.capacities[1]-sum(tasks_allocation[1]))
        tasks_allocation = np.array(tasks_allocation)
        df = pd.DataFrame(tasks_allocation)
        column_labels = []
        for i in range(self.num_users):
            column_labels.append("User " + str(i+1))
        column_labels.append("Unused")
        df.columns = column_labels
        df = df.div(df.sum(axis=1), axis=0)
        ax = df.plot(
            kind = 'bar',
            stacked = True,
            title = 'Allocations',
            mark_right = True)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1.0))
        indices = [x for x in range(len(self.capacities))]
        plt.xticks(indices, labels)
        plt.xticks(rotation=0)
        plt.savefig("charts/allocation" + str(self.algorithm) + ".png")
        plt.show()

def args_parser():
    parser = argparse.ArgumentParser(description="Run different algorithms for resource allocation and compare results")
    parser.add_argument('-i',"--input",dest="input",help="Provide the path of input CSV file")
    parser.add_argument('-a',"--algo",dest="algo",help="Specify resource sharing algorithm to use (int value). \n1. DRF (default)\n2.Asset Fairness\n3.CEEI")
    parser.add_argument('-g',"--graph",action='store_true',help="Specify if graph should be generated for the allocation")
    parser.add_argument('-w',"--weights",nargs='?',dest="weights",help="[Optional] Provide the path of weights CSV file")
    parser.set_defaults(input='tasks.csv',algo=1,graph=True)
    args = parser.parse_args()
    return args


def main():
    args = args_parser()
    df = pd.read_csv(args.input)
    labels = df.columns.values
    capacities = df.iloc[0].values.tolist()
    demands = df.loc[1:].values.tolist()

    allocator = Scheduler(capacities,algorithm=Algorithm(int(args.algo)))
    if args.weights:
        df2 = pd.read_csv(args.weights)
        weights = df2.loc[0:].values.tolist()
    for idx, demand in enumerate(demands):
        if args.weights:
            allocator.insert_task(Task(idx,demand,weights[idx]))
        else:
            allocator.insert_task(Task(idx,demand))
    allocations = allocator.allocate()
    for idx, allocation in enumerate(allocations):
        print("User " + str(idx) + " is allocated " + str(allocation) + " tasks")
    if args.graph:
        allocator.plot(labels)

    

if __name__=="__main__":
    main()