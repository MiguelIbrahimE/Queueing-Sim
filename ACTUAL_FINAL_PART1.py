import numpy as np
import simpy
import matplotlib.pyplot as plt
import copy
import random


class Gillespie:

    def __init__(self, p01, p10):
        self.p01 = p01
        self.p10 = p10

    def calculate_total_rate(self):
        return self.p01 + self.p10
    
    def Generate(self):
        rando = random.uniform(0, 1)

        if (rando < self.p01/self.calculate_total_rate()):
            return 1
        else:
            return 0


class Simulator:
    #for question 2

    def __init__(self):
        self.processes = []

    def addProcess(self, process):
        self.processes.append(process)

    def deleteProcess(self, process):
        self.processes.remove(process)

    def simulate(self, max_capacity=2001,  min_capacity = 0, amount_of_times = 1, capacity_increment = 10, time = 20000):
        for process in self.processes:
            process.simulate(max_capacity, min_capacity, amount_of_times, capacity_increment, time)


class Process:

    def __init__(self, alphas, gamma, p, e, gp = Gillespie(0.5,0.5)):
        self.alphas = alphas
        self.gamma = gamma
        self.p = p
        self.e = e
        self.gp = gp

    def energy_source_process(self, env, alpha, gamma, p, e, job_times, capacity):

        state = self.gp.Generate()
        total_energy_demand = 0
        B = 0.1
        total_energy_bought = 0
        #jobs queue
        jobs = []
        #time of the system currently
        my_time = 0

        #as long as there are still remaining jobs left to do
        while job_times != []:
            # Generate a new job
            state = self.gp.Generate()

            # check if the next job time stamp is within the current simulation time, if a new job has to be created
            if job_times[0] <= my_time:
                #generate tau, time needed to perform the job
                tau_k = np.random.exponential(lambda_rate)
                #generate the amount of total wattage needed for the job
                Watt_new = np.random.exponential(2000) 
                #Calculate Watthours for job, amount of wattage used over a time period
                Wh_new = tau_k*Watt_new  
                #Add up job Wh demand to total demand
                total_energy_demand += Wh_new

                #remove the job timestamp
                job_times.pop(0)
                #add new job to the queue 
                jobs.append(Wh_new)

            #every iteration increment the time with 10 units                
            my_time += 10

            # Check if there are any jobs to spend 
            if jobs == []:
                continue

            # If we can get energy from the source
            if state == 1:
                total_source = np.random.uniform(0, 10)
                # Check if source is enough to satisfy the job
                if jobs[0] - total_source < 0: 
                    #spend source to complete the job
                    remaining_source = total_source - jobs[0]
                    # the job now needs 0 remaining Wh and can be removed
                    jobs[0] = 0
                    jobs.pop(0)
                    # Add the excess remaining source to the battery
                    if B + e * remaining_source > capacity:
                        B = capacity
                    else:
                        B = B + e * remaining_source

                # Source is not enough so spend all of it, and then we check the battery        
                else: 
                    jobs[0] = jobs[0] - total_source
                    # Check if battery is enough
                    if B - jobs[0] > 0: 
                        B = B - jobs[0]
                        # the job now needs 0 remaining Wh and can be removed
                        jobs[0] = 0
                        jobs.pop(0)

                        # Spend tau amount of time on the job (needed for the simulation, together with the job_times variable)
                        yield env.timeout(1/tau_k)
                        
                    # Battery does not have enough
                    else: 
                        # Use all energy from battery for the job
                        jobs[0] = jobs[0] - B
                        B = 0

                    yield env.timeout(1/tau_k)
             
            
            # Else if the energy source is OFF
            else:
                # Determine whether to reschedule or buy external energy

                # If battery is enough
                if B - jobs[0] > 0: 
                        B = B - jobs[0]
                        jobs[0] = 0
                        jobs.pop(0)
                
                # Battery does not have enough
                else: 
                    # Use all energy from battery for the job
                    jobs[0] = jobs[0] - B
                    B = 0

                    # Check if we can buy enough external energy while staying under alpha
                    if (jobs[0] + total_energy_bought) / total_energy_demand <= alpha: 
                        # Calculate cost (for if we want to calculate total cost)
                        cost = p * jobs[0]

                        # Buy required energy and finish the job
                        total_energy_bought += jobs[0]
                        jobs[0] = 0
                    else:
                        # Reschedule
                        yield env.timeout(np.random.exponential(gamma))

    def simulate(self, max_capacity,  min_capacity = 0, amount_of_times = 1, capacity_increment = 10, time = 20000):

        results = []
        for alpha in self.alphas:
            results.append(0)

        # Run the simulations amount_of_times to get a more average result
        for i in range(1, amount_of_times+1):
            
            # For every alpha value do the simulation
            for alpha in self.alphas:

                my_time = 0
                job_times = []

                # Make the list of job_times
                while my_time < 20000:
                    new_job_time = np.random.poisson(lam= lambda_rate) * 100
                    my_time += new_job_time
                    if my_time < 20000:
                        job_times.append(my_time)
                
                # Find a capacity by iterating over possible values for it
                for capacity in list(range(min_capacity, max_capacity, capacity_increment)):

                    job_times_copy = copy.copy(job_times)
                    env = simpy.Environment()

                    # Run the simulation
                    energy_source = env.process(self.energy_source_process(env, alpha, self.gamma, self.p, self.e, job_times_copy, capacity))

                    # Adjust simulation time accordingly
                    env.run(until=20000)

                    # Make sure that all the jobs are done
                    if not job_times_copy:
                        # Add up the calulated capacity for this simulation for this alpha (Accumulator that is later used to calculate an average)
                        results[self.alphas.index(alpha)] += capacity
                        # Move on to the next alpha
                        break  
                    else:
                        continue

        # Calculate average of min battery of the simulations and print it out
        print("Simulation Finished, amount of simulations ran: " , amount_of_times)
        for alpha in self.alphas:
            print(f"Alpha: {alpha}, Average Min Battery Capacity: {results[self.alphas.index(alpha)]/amount_of_times}")


lambda_rate = 0.2  # Poisson stream rate
gamma = 0.5  # Rescheduling rate
p = 0.2  # External energy purchase rate
e = 0.8  # Battery charging efficiency
B_values = []  # To store minimal battery capacities for different alpha values


p = Process([0.0, 0.01, 0.05, 0.1], gamma, p, e)
p.simulate(max_capacity=2001, amount_of_times=25)
