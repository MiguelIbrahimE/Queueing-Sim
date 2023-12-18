import numpy as np
import simpy
import matplotlib.pyplot as plt
import copy
import random
import threading

from concurrent.futures import ThreadPoolExecutor, as_completed

class Gillespie:

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
        # Debugging
        if (round(self.getEnergy(), 5) == round(self.getTotalAddition() + self.getTotalUsage(), 5)):
            s = 'True, final values are: Remaining Energy: {}, Total Balance: {}, Total Addition: {}, Total Usage:  {}'.format(round(self.getEnergy(), 5), round(self.getTotalAddition() + self.getTotalUsage(), 5), round(self.getTotalAddition(), 5), round(self.getTotalUsage(), 5))
            print(s)
            return 1
        else:
            s = 'False, final values are: Remaining Energy: {}, Total Balance: {}, Total Addition: {}, Total Usage:  {}'.format(round(self.getEnergy(), 5), round(self.getTotalAddition() + self.getTotalUsage(), 5), round(self.getTotalAddition(), 5), round(self.getTotalUsage(), 5))
            print(s)
            return 0
            
    def getTotalUsage(self):
        # Added definition of TotalUsage
        with self.lock:
            return self.total_usage
        
    def getTotalAddition(self):
        # Debugging
        with self.lock:
            return self.total_additions

class Simulator:
    def __init__(self, max_lambda=1.0, lambda_step=0.1):
        self.processes = []
        self.Battery = Energy_Collector()
        self.max_lambda = max_lambda
        self.lambda_step = lambda_step

    def addUser(self, process):
        process.setBattery(self.Battery)  # Pass the Battery object to the process
        self.processes.append(process)

    def deleteProcess(self, process):
        self.processes.remove(process)

    def simulate(self, max_capacity=2001,  min_capacity=0, amount_of_times=1, capacity_increment=10, time=20000):
        for process in self.processes:
            process.setBattery(self.Battery)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process.simulate, max_capacity, min_capacity, amount_of_times, capacity_increment, time) for process in self.processes]

            for future in as_completed(futures):
                future.result()

            self.Battery.Check()

    def simulate_all_same_on_off_process(self, gp=Gillespie(0.5, 0.5), max_capacity=2001, min_capacity=0, amount_of_times=1, capacity_increment=10, time=20000):
        for process in self.processes:
            process.setGP(gp)

        self.simulate(max_capacity, min_capacity, amount_of_times, capacity_increment, time)

    def find_max_lambda(self, max_capacity=2001, min_capacity=0, amount_of_times=1, capacity_increment=10, time=20000):
        max_lambda = self.max_lambda
        lambda_step = self.lambda_step
        while max_lambda > 0:
            for user in self.processes:
                user.gp.p01 = max_lambda
                user.gp.p10 = max_lambda
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

    def __init__(self, alphas, gamma, p, e, gp=Gillespie(0.5, 0.5)):
        self.alphas = alphas
        self.gamma = gamma
        self.p = p
        self.e = e
        self.gp = gp
        self.Battery = None
        self.results = None

    def getResults(self):
        if (self.results is None):
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

        while job_times:
            state = self.gp.Generate()
            if job_times[0] <= my_time:
                tau_k = np.random.exponential(lambda_rate)
                W_k_new = np.random.exponential(2000)  # Random wattage for the job
                Wh_new = tau_k * W_k_new 
                job_times.pop(0)
                total_energy_demand += Wh_new
                jobs.append(Wh_new)
            my_time += 10

            if jobs:
                if state == 1:
                    total_source = np.random.uniform(0, 10)
                    if jobs[0] - total_source < 0:
                        remaining_source = total_source - jobs[0]
                        jobs[0] = 0
                        jobs.pop(0)
                        if B + remaining_source > capacity:
                            self.Battery.addEnergy(remaining_source)
                            B = capacity
                        else:
                            B = B + e * remaining_source
                    else:
                        jobs[0] = jobs[0] - total_source
                        if B - jobs[0] > 0:
                            B = B - jobs[0]
                            jobs[0] = 0
                            jobs.pop(0)
                            yield env.timeout(1 / tau_k)
                        else:
                            jobs[0] = jobs[0] - B
                            B = 0
                        yield env.timeout(1 / tau_k)
                else:
                    if B - jobs[0] > 0:
                        B = B - jobs[0]
                        jobs[0] = 0
                        jobs.pop(0)
                    else:
                        jobs[0] = jobs[0] - B
                        B = 0
                        if (((jobs[0] + total_energy_bought) / total_energy_demand <= alpha) and (self.Battery.getEnergy() > jobs[0])):
                            cost = p * jobs[0]
                            total_energy_bought += jobs[0]
                            self.Battery.useEnergy(jobs[0])
                            jobs[0] = 0
                        else:
                            yield env.timeout(np.random.exponential(gamma))

    def simulate(self, max_capacity, min_capacity=0, amount_of_times=1, capacity_increment=10, time=20000):
        results = []
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
                    env.run(until=time)

                    if not job_times_copy:
                        results[self.alphas.index(alpha)] += capacity
                        break
                    else:
                        continue
        self.results = results

        print("Simulation Finished, amount of simulations ran: ", amount_of_times)
        for alpha in self.alphas:
            print(f"Alpha: {alpha}, Average Min Battery Capacity: {results[self.alphas.index(alpha)] / amount_of_times}")

lambda_rate = 0.2  # Poisson stream rate
gamma = 0.5  # Rescheduling rate
p = 0.2  # External energy purchase rate
e = 0.8  # Battery charging efficiency
B_values = []  # To store minimal battery capacities for different alpha values

# Create User instances
user1 = Process([0.0, 0.01, 0.05, 0.1], gamma, p, e)
user2 = Process([0.0, 0.01, 0.05, 0.1], gamma, p, e)
user3 = Process([0.0, 0.01, 0.05, 0.1], gamma, p, e)

# Initialize the Simulator
s = Simulator(max_lambda=1.0, lambda_step=0.01)

# Add User instances to the simulator
s.addUser(user1)
s.addUser(user2)
s.addUser(user3)


# Added code for simulating regeneration periods
regeneration_periods = []
B_min = 0.1
B_max= 1000
simulation_time= 2000
N = 3
E_s = 0.5  # Energy consumption rate for discharging the battery
E_b = 0.8  # Energy consumption rate for recharging the battery

def user_process(env, battery):
    while True:
        # Simulate energy demand arrivals using a Poisson process
        yield env.timeout(np.random.exponential(1 / lambda_rate))

        # Check battery level
        if battery.capacity <= B_min:
            # Battery needs to recharge
            recharge_time = (B_max - battery.capacity) / E_s
            yield env.timeout(recharge_time)
            battery.capacity = B_max  # Battery is fully charged

        elif battery.capacity >= B_max:
            # Battery needs to discharge
            discharge_time = (battery.capacity - B_min) / (E_s + E_b)
            yield env.timeout(discharge_time)
            battery.capacity = B_min  # Battery is discharged


def simulate_system(N, lambda_rate, B_max, B_min, E_s, E_b, simulation_time):
    # Create a simulation environment
    env = simpy.Environment()

    # Create a shared battery resource with capacity B_max
    battery = simpy.Container(env, capacity=B_max, init=B_max)

    # Create user processes
    for _ in range(N):
        env.process(user_process(env, battery))

    # Run the simulation
    env.run(until=simulation_time)

    return battery
# Perform multiple simulations to estimate the lower bound of expected regeneration period
num_simulations = 1000
for _ in range(num_simulations):
    battery = simulate_system(N, lambda_rate, B_max, B_min, E_s, E_b, simulation_time)
    regeneration_period = simulation_time / num_simulations
    regeneration_periods.append(regeneration_period)

# Calculate the lower bound of expected regeneration period
lower_bound = np.mean(regeneration_periods)

print(f"Lower Bound of Expected Regeneration Period for {N} Users: {lower_bound}")
