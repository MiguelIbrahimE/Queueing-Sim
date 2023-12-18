# Import additional libraries
from scipy.spatial.distance import pdist, squareform

import numpy as np
import simpy
import matplotlib.pyplot as plt
import copy
import random
import threading

from concurrent.futures import ThreadPoolExecutor, as_completed

class Gillespie(object):

    def __init__(self, p01, p10):
        self.p01 = p01
        self.p10 = p10

    def calculate_total_rate(self):
        return self.p01 + self.p10
    
    def Generate(self):
        rando = random.uniform(0, 1)
        if (rando < self.p01 / self.calculate_total_rate()):
            return 1
        else:
            return 0
        
    def setP01(self, p):
        self.p01 = p

    def setP10(self, p):
        self.p10 = p


class ExclusiveGillespie(Gillespie):

    def __init__(self, p01, p10):
        super().__init__(p01, p10)
        self.first = True
        self.lock = threading.Lock()
        self.current = None

    def Generate(self):
        with self.lock:
            if (self.first == True):
                self.first = False
                self.current = super().Generate()
            return self.current
            
    def next(self):
        with self.lock:
            self.first = True

    def setP01(self, p):
        with self.lock:
            return super().setP01(p)
        
    def setP10(self, p):
        with self.lock:
            return super().setP10(p)
        
class Energy_Collector:

    def __init__(self):
        self.energy = 0
        self.total_additions = 0
        self.total_usage = 0
        self.lock = threading.Lock()

    def getEnergy(self):
        with self.lock:
            return self.energy

    def addEnergy(self, value):
        with self.lock:
            self.energy += value
            self.total_additions += value
    
    def useEnergy(self, value):
        with self.lock:
            self.energy -= value
            self.total_usage -= value

    def Check(self):
        #debugging
        if (round(self.getEnergy(), 5) == round(self.getTotalAddition() + self.getTotalUsage(), 5)):
            s = 'True, final values are: Remaining Energy: {}, Total Balnce: {}, Total Addition: {}, Total Usage:  {}'.format(round(self.getEnergy(), 5), round(self.getTotalAddition() + self.getTotalUsage(), 5) ,round(self.getTotalAddition(), 5), round(self.getTotalUsage(),5))
            print(s)
            return 1
        else:
            s = 'False, final values are: Remaining Energy: {}, Total Balnce: {}, Total Addition: {}, Total Usage:  {}'.format(round(self.getEnergy(), 5), round(self.getTotalAddition() + self.getTotalUsage(), 5) ,round(self.getTotalAddition(), 5), round(self.getTotalUsage(),5))
            print(s)
            return 0
            
    def getTotalUsage(self):
        #debugging
        with self.lock:
            return self.total_usage
        
    def getTotalAddition(self):
        #debugging
        with self.lock:
            return self.total_additions

class Simulator:
    def __init__(self, max_lambda=1.0, lambda_step=0.1):
        self.processes = []
        self.Battery = Energy_Collector()
        self.max_lambda = max_lambda
        self.lambda_step = lambda_step
        self.lambdas = []

    def addUser(self, process):
        process.setBattery(self.Battery)  # Pass the Battery object to the process
        self.processes.append(process)

    def deleteProcess(self, process):
        self.processes.remove(process)

    def readLambdas(self):
        for i in self.lambdas:
            print(i)

    def simulate(self, max_capacity=2001,  min_capacity = 0, amount_of_times = 1, capacity_increment = 10, time = 20000):
        #Function to simulate different processes per user
        #need to multiproces this function
        #scenario ii

        max_lambda = self.max_lambda
        lambda_step = self.lambda_step
        while max_lambda > 0:
            for user in self.processes:
                user.gp.setP01(max_lambda)
                user.gp.setP10(max_lambda)
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(user.simulate, max_capacity, min_capacity, amount_of_times, capacity_increment, time) for user in self.processes]
                for future in futures:
                    future.result()
            backlog = False
            for user in self.processes:
                if user.getResults()[0] > 0:
                    backlog = True
                    break
            if backlog:
                max_lambda -= lambda_step
                break
            else:
                max_lambda += lambda_step
            self.Battery.Check()
        self.lambdas.append(max_lambda)

        for process in self.processes:
            process.getOutput()


    def simulate_all_same_on_off_process(self, gp = ExclusiveGillespie(0.5, 0.5), max_capacity=2001,  min_capacity = 0, amount_of_times = 1, capacity_increment = 10, time = 20000):
        #scenario i
        for process in self.processes:
            process.setGP(gp)
        self.simulate(max_capacity, min_capacity, amount_of_times, capacity_increment, time)
        
    def find_max_lambda(self, max_capacity=2001, min_capacity=0, amount_of_times=1, capacity_increment=10, time=20000):
        max_lambda = self.max_lambda
        lambda_step = self.lambda_step
        while max_lambda > 0:
            for user in self.processes:
                user.gp.setP01(max_lambda)
                user.gp.setP10(max_lambda)
            with ThreadPoolExecutor(max_workers=len(self.processes)) as executor:
                futures = [executor.submit(user.simulate, max_capacity, min_capacity, amount_of_times, capacity_increment, time) for user in self.processes]
                for future in futures:
                    future.result()
            backlog = False
            for user in self.processes:
                if user.getResults()[0] > 0:
                    backlog = True
                    break
            if backlog:
                max_lambda -= lambda_step
                break
            else:
                max_lambda += lambda_step
        return max_lambda
    
class Process:

    def __init__(self, alphas, gamma, p, e, gp = Gillespie(0.5, 0.5)):
        self.alphas = alphas
        self.gamma = gamma
        self.p = p
        self.e = e
        self.gp = gp
        self.Battery = None
        self.results = None
        self.amount_of_times = 0

    def getResults(self):
        if (self.results == None):
            return "Simulate the process first"
        else:
            return self.results

    def setBattery(self, battery):
        self.Battery = battery

    def setGP(self, gp):
        self.gp = gp

    def energy_source_process(self, env, alpha, gamma, p, e, job_times, capacity):

        state = self.gp.Generate()
        total_energy_demand = 0
        B = 0.1
        total_energy_bought = 0
        jobs = []
        my_time = 0

        while job_times != []:
            # Generate a new job
            state = self.gp.Generate()

            if job_times and job_times[0] <= my_time:
                tau_k = np.random.exponential(lambda_rate)
                W_k_new = np.random.exponential(2000)  # Random wattage for the job
                Wh_new = tau_k * W_k_new 
                job_times.pop(0)
                total_energy_demand += Wh_new
                jobs.append(Wh_new)
            my_time += 10

            if jobs == []:
                continue
            if state == 1:
                total_source = np.random.uniform(0, 10)
                # Consume energy from the intermittent source
                if jobs[0] - total_source < 0: # Source is enough to satisfy the job
                    remaining_source = total_source - jobs[0]
                    jobs[0] = 0
                    jobs.pop(0)
                    if B + remaining_source > capacity: # Excess source cannot be stored
                        self.Battery.addEnergy(remaining_source)
                        B = capacity
                    else:
                        B = B + e * remaining_source
                else: # Source is not enough, we need to check the battery
                    jobs[0] = jobs[0] - total_source
                    if B - jobs[0] > 0: # Battery is enough
                        B = B - jobs[0]
                        jobs[0] = 0
                        jobs.pop(0)
                        yield env.timeout(1 / tau_k)

                    else: # Battery is not enough
                        jobs[0] = jobs[0] - B
                        B = 0
                    yield env.timeout(1 / tau_k)
            # Check if the energy source is ON  
            else:
                # Energy source is OFF
                # Determine whether to reschedule or buy external energy
                if B - jobs[0] > 0: # Battery is enough
                        B = B - jobs[0]
                        jobs[0] = 0
                        jobs.pop(0)
                else: # Battery is not enough
                    jobs[0] = jobs[0] - B
                    B = 0
                    if (((jobs[0] + total_energy_bought) / total_energy_demand <= alpha) and (self.Battery.getEnergy() > jobs[0])): #Buy external energy
                        cost = p * jobs[0]
                        total_energy_bought += jobs[0]
                        self.Battery.useEnergy(jobs[0])
                        jobs[0] = 0
                    else:
                        yield env.timeout(np.random.exponential(gamma)) # Reschedule
            #print(test)


    def simulate(self, max_capacity= 2000, min_capacity=0, amount_of_times=1, capacity_increment=10, time=20000):
        results = []
        self.amount_of_times=amount_of_times
        for alpha in self.alphas:
            results.append(0)

        for i in range(1, amount_of_times + 1):
            for alpha in self.alphas:
                my_time = 0
                job_times = []

                while my_time < time:
                    new_job_time = np.random.poisson(lam=lambda_rate) * 100
                    my_time += new_job_time
                    if my_time < time:
                        job_times.append(my_time)
                
                for capacity in list(range(min_capacity, max_capacity, capacity_increment)):
                    job_times_copy = copy.copy(job_times)
                    env = simpy.Environment()
                    env.process(self.energy_source_process(env, alpha, self.gamma, self.p, self.e, job_times_copy, capacity))
                    env.run(until=time)  # Adjust simulation time accordingly

                    if not job_times_copy:
                        results[self.alphas.index(alpha)] += capacity
                        break  # Move on to the next alpha
                    else:
                        continue
            if (isinstance(self.gp, ExclusiveGillespie)):
                self.gp.next()

        self.results = results
        #self.getOutput()

    def getOutput(self):
        
        if (self.Battery.Check()):
            print("--------------------------------------------------")
            print("Simulation Succesfull, running the simulation {self.amount_of_times} the values are as follows:")
            for alpha in self.alphas:
                print(f"Alpha: {alpha}, Average Min Battery Capacity: {self.results[self.alphas.index(alpha)] / self.amount_of_times}")


class RingGraphSimulator(Simulator):
    def __init__(self, num_users, ring_rate, alphas=None, max_lambda=1.0, lambda_step=0.1, **kwargs):
        super().__init__(max_lambda, lambda_step)
        if alphas is None:
            alphas = []  # Set a default value or handle it as needed
        self.alphas = alphas
        self.num_users = num_users
        self.ring_rate = ring_rate
        self.distance_matrix = self.generate_ring_distance_matrix()

        self.processes = []
        self.Battery = Energy_Collector()
        self.gamma = gamma
        self.p = p
        self.e = e
        self.lambdas = []

    def generate_ring_distance_matrix(self):
        positions = np.linspace(0, 2 * np.pi, self.num_users, endpoint=False)
        positions = np.column_stack([np.cos(positions), np.sin(positions)])
        return squareform(pdist(positions))

    def generate_ring_correlated_states(self, env, user, alpha, gamma, p, e, job_times, capacity):
        state = user.gp.Generate()
        total_energy_demand = 0
        B = 0.1
        total_energy_bought = 0
        jobs = []
        my_time = 0

        while job_times:
            # Generate a new job
            state = user.gp.Generate()

            if job_times and job_times[0] <= my_time:
                tau_k = np.random.exponential(lambda_rate)
                W_k_new = np.random.exponential(2000)  # Random wattage for the job
                Wh_new = tau_k * W_k_new
                job_times.pop(0)
                total_energy_demand += Wh_new
                jobs.append(Wh_new)
            my_time += 10

            if not jobs:
                continue

            if state == 1:
                total_source = np.random.uniform(0, 10)
                # Consume energy from the intermittent source
                if jobs[0] - total_source < 0:  # Source is enough to satisfy the job
                    remaining_source = total_source - jobs[0]
                    jobs[0] = 0
                    jobs.pop(0)
                    if B + remaining_source > capacity:  # Excess source cannot be stored
                        user.Battery.addEnergy(remaining_source)
                        B = capacity
                    else:
                        B = B + e * remaining_source
                else:  # Source is not enough, we need to check the battery
                    jobs[0] = jobs[0] - total_source
                    if B - jobs[0] > 0:  # Battery is enough
                        B = B - jobs[0]
                        jobs[0] = 0
                        jobs.pop(0)
                        yield env.timeout(1 / tau_k)

                    else:  # Battery is not enough
                        jobs[0] = jobs[0] - B
                        B = 0
                    yield env.timeout(1 / tau_k)
            else:
                # Energy source is OFF
                # Determine whether to reschedule or buy external energy
                if B - jobs[0] > 0:  # Battery is enough
                    B = B - jobs[0]
                    jobs[0] = 0
                    jobs.pop(0)
                else:  # Battery is not enough
                    jobs[0] = jobs[0] - B
                    B = 0
                    if ((jobs[0] + total_energy_bought) / total_energy_demand <= alpha) and (
                            user.Battery.getEnergy() > jobs[0]):  # Buy external energy
                        cost = p * jobs[0]
                        total_energy_bought += jobs[0]
                        user.Battery.useEnergy(jobs[0])
                        jobs[0] = 0
                    else:
                        yield env.timeout(np.random.exponential(gamma))  # Reschedule

    def simulate_ring_graph(self, gp=ExclusiveGillespie(0.5, 0.5), max_capacity=2001, min_capacity=0,
                            amount_of_times=1, capacity_increment=10, time=20000):
        for process in self.processes:
            process.setGP(gp)

        for user in self.processes:
            env = simpy.Environment()
            user_processes = [self.generate_ring_correlated_states(env, user, user.gp.p01, self.gamma, self.p, self.e,
                                                                    [], max_capacity) for _ in range(amount_of_times)]
            env.process(self.ring_process(env, user_processes, max_capacity, min_capacity, amount_of_times,
                                          capacity_increment, time))
            env.run(until=time)

    def ring_process(self, env, user_processes, max_capacity, min_capacity, amount_of_times, capacity_increment, time):
        for alpha in self.alphas:
            results = []

            for i in range(amount_of_times):
                user_jobs = [list(process) for process in user_processes]

                my_time = 0
                job_times = []

                while my_time < time:
                    my_time += 10
                    yield env.timeout(1)
                    new_job_time = np.random.poisson(lam=lambda_rate) * 100
                    my_time += new_job_time
                    if my_time < time:
                        job_times.append(my_time)

                for capacity in range(min_capacity, max_capacity, capacity_increment):
                    job_times_copy = copy.copy(job_times)
                    user_jobs = [list(self.generate_ring_correlated_states(env, self.processes[i], alpha, self.gamma, self.p, self.e, job_times_copy, capacity)) for i in range(len(self.processes))]

                    env.run(until=time)  # Adjust simulation time accordingly

                    if not job_times_copy:
                        results.append(sum([self.processes[i].Battery.getEnergy() for i in range(len(self.processes))]))
                        break  # Move on to the next alpha
                    else:
                        continue

            average_result = sum(results) / len(results)
            print(f"Alpha: {alpha}, Average Min Battery Capacity: {average_result}")


    def calculate_temporal_correlation(self, user_processes):
        # Flatten the generator and convert it to a list
        flat_processes = [item for sublist in user_processes for item in sublist]

        # Check if the flattened list is not empty
        if flat_processes:
            # Calculate temporal correlation for a single user
            return np.corrcoef(flat_processes)
        else:
            print("User processes are empty.")
            return np.empty((0, 0))

    def calculate_spatial_correlation(self, user_results):
        # Check if user_results is empty
        if not user_results or not user_results[0]:
            print("Spatial correlation data is empty.")
            return np.empty((0, 0))

        # Calculate spatial correlation between different users
        return np.corrcoef(user_results)

    def plot_temporal_correlation(self, temporal_correlation_matrix):
        # Flatten the matrix to a 1D array
        if temporal_correlation_matrix.size > 0:
            plt.imshow(temporal_correlation_matrix, cmap='viridis', origin='lower', interpolation='none')
            plt.colorbar()
            plt.title('Temporal Correlation Matrix')
            plt.show()
        else:
            print("Temporal correlation matrix is empty.")

    def plot_spatial_correlation(self, spatial_correlation_matrix):
        # Ensure that the matrix is not empty
        if spatial_correlation_matrix.size > 0:
            plt.imshow(spatial_correlation_matrix, cmap='viridis', origin='lower', interpolation='none')
            plt.colorbar()
            plt.title('Temporal Correlation Matrix')
            plt.show()
        else:
            print("Temporal correlation matrix is empty.")


    def analyze_correlations(self, gp=ExclusiveGillespie(0.5, 0.5), max_capacity=2001, min_capacity=0,
                         amount_of_times=1, capacity_increment=10, time=20000):
        for process in self.processes:
            process.setGP(gp)

        user_processes = []

        for user in self.processes:
            env = simpy.Environment()
            processes = [self.generate_ring_correlated_states(env, user, user.gp.p01, self.gamma, self.p, self.e,
                                                              [], max_capacity) for _ in range(amount_of_times)]
            env.process(self.ring_process(env, processes, max_capacity, min_capacity, amount_of_times,
                                          capacity_increment, time))
            env.run(until=time)
            user_processes.append(processes)

        # Analyze temporal correlation
        temporal_correlation_matrix = self.calculate_temporal_correlation(user_processes[0][0])
        self.plot_temporal_correlation(temporal_correlation_matrix)

        # Analyze spatial correlation
        user_results = [
        [sum([self.processes[i].Battery.getEnergy() for i in range(len(self.processes))]) for _ in range(amount_of_times)]
        for _ in range(len(self.processes))]
        spatial_correlation_matrix = self.calculate_spatial_correlation(user_results)
        self.plot_spatial_correlation(spatial_correlation_matrix)
# ...

lambda_rate = 0.2  # Poisson stream rate
gamma = 0.5  # Rescheduling rate
p = 0.2  # External energy purchase rate
e = 0.8  # Battery charging efficiency
B_values = []  # To store minimal battery capacities for different alpha values

# Initialize the RingGraphSimulator
alphas_ring = [0.0, 0.01, 0.05, 0.1]

user1 = Process(alphas=[0.0, 0.01, 0.05, 0.1], gamma=0.5, p=0.2, e=0.8, gp=Gillespie(0.5, 0.5))
user2 = Process(alphas=[0.0, 0.01, 0.05, 0.1], gamma=0.5, p=0.2, e=0.8, gp=Gillespie(0.5, 0.5))
user3 = Process(alphas=[0.0, 0.01, 0.05, 0.1], gamma=0.5, p=0.2, e=0.8, gp=Gillespie(0.5, 0.5))

# Initialize the RingGraphSimulator
ring_simulator = RingGraphSimulator(num_users=3, alphas=alphas_ring, max_lambda=1.0, lambda_step=0.01, ring_rate=0.1, gamma=0.5, p=0.2, e=0.8)

# Add User instances to the simulator
ring_simulator.addUser(user1)
ring_simulator.addUser(user2)
ring_simulator.addUser(user3)

# Simulate the ring graph scenario
ring_simulator.analyze_correlations(gp=ExclusiveGillespie(0.5, 0.5), amount_of_times=10)

# Print lambdas
ring_simulator.readLambdas()
