import numpy as np
import simpy
import random
import threading

class MarkovOnOffProcess:
    def __init__(self, env, user_id, arrival_rate, off_rate, battery_capacity, users):
        self.env = env
        self.user_id = user_id
        self.arrival_rate = arrival_rate
        self.off_rate = off_rate
        self.battery_capacity = max(0, battery_capacity)
        self.remaining_battery = self.battery_capacity
        self.job_queue = simpy.Store(self.env)
        self.consecutive_zero_capacity_count = 0  # Initialize the counter
        self.max_lambda = 0.01  # Initial lambda value
        self.estimate = 0  # Variable for estimating expected value
        self.control_estimate = 0  # Variable for estimating control function
        self.control_variate = 0  # Control variate
        self.users = users  # List of all users

    def user_process(self):
        while True:
            # Check if there are jobs in the queue
            if len(self.job_queue.items) > 0:
                job = yield self.job_queue.get()
                energy_demand = job['energy_demand']
                energy_production = job['energy_production']
            else:
                # Generate tau_k (time taken for the job)
                tau_k = np.random.exponential(1.0 * self.max_lambda)

                # Calculate energy demand in watt-hours
                energy_demand = tau_k * 2000 * self.max_lambda  # Corrected lambda for demand

                # Simulate energy production (random generation)
                energy_production = tau_k * energy_demand  # Modify scale here

            # Check if battery capacity is sufficient
            if self.remaining_battery >= energy_demand:
                self.remaining_battery -= energy_demand
                self.consecutive_zero_capacity_count = 0  # Reset the counter
            else:
                # Not enough energy in the battery, add job to the queue
                yield self.job_queue.put({'energy_demand': energy_demand, 'energy_production': energy_production})

                # Reschedule the process
                yield self.env.timeout(np.random.exponential(scale=1.0 / self.off_rate))

            # Calculate the energy difference between production and demand
            energy_difference = energy_production - energy_demand

            # Update battery capacity based on energy difference
            if energy_difference > 0:
                # Supply exceeds demand, add charge to the battery (up to maximum capacity)
                self.remaining_battery = min(self.battery_capacity, self.remaining_battery + energy_difference)
            else:
                # Demand exceeds supply, remove the difference from the battery
                self.remaining_battery -= abs(energy_difference)

            # Ensure battery capacity doesn't exceed the maximum
            self.remaining_battery = max(0, min(self.battery_capacity, self.remaining_battery))

            # Increase lambda with each iteration
            self.max_lambda += 0.01

            # Update estimates for control variates
            self.estimate += energy_production
            self.control_estimate += energy_demand

            # Calculate the control variate
            self.control_variate = abs(self.estimate - self.control_estimate)

            # Print battery capacity at each instant
            print(f"User {self.user_id}: Battery Capacity: {self.remaining_battery} Lambda: {self.max_lambda} Control Variate: {self.control_variate}")

            # Check if battery capacity is consistently 0
            if self.remaining_battery == 0:
                self.consecutive_zero_capacity_count += 1
                if self.consecutive_zero_capacity_count >= 10:  # Set a threshold for exiting
                    print(f"User {self.user_id}: Battery capacity remained at 0 for too long. Exiting simulation.")
                    exit()

            # Wait for the next energy demand event
            yield self.env.timeout(np.random.exponential(scale=1.0 / self.max_lambda))

    def help_other_users(self, energy_demand):
        # Check if other users need assistance and share energy if possible
        for user in self.users:
            if user.user_id != self.user_id and user.remaining_battery < energy_demand:
                energy_to_share = min(self.remaining_battery, energy_demand - user.remaining_battery)
                user.remaining_battery += energy_to_share
                self.remaining_battery -= energy_to_share

def generate_user_values(user_id, alpha, off_rate, battery_capacity, users):
    env = simpy.Environment()
    process = MarkovOnOffProcess(env, user_id, alpha, off_rate, battery_capacity, users)
    env.process(process.user_process())
    env.run(until=10000)  # Adjust the simulation duration as needed
    return user_id, process.remaining_battery, process.control_variate

def find_max_lambda_multithreaded(alpha, num_users, battery_capacity, max_lambda_limit=2048):
    threads = []
    results = []
    users = []

    for i in range(num_users):
        thread = threading.Thread(target=lambda u=i: results.append(generate_user_values(u, alpha, 0.5, battery_capacity, users)))
        threads.append(thread)
        user = MarkovOnOffProcess(None, i, alpha, 0.5, battery_capacity, users)
        users.append(user)
        thread.start()

    for thread in threads:
        thread.join()

    # Process results as needed
    max_lambda_values = [0.001] * num_users  # Initialize lambda values for each user
    remaining_battery_values = [result[1] for result in results]
    control_variate_values = [result[2] for result in results]

    return max_lambda_values, remaining_battery_values, control_variate_values

if __name__ == '__main__':
    # Set parameters
    alpha = 0.0
    num_users = 2  # Number of users
    initial_battery_capacity = 750
    max_lambda_limit = 12048

    # Calculate max_lambda and other values for each user using multithreading
    max_lambda_values, remaining_battery_values, control_variate_values = find_max_lambda_multithreaded(
        alpha, num_users, initial_battery_capacity, max_lambda_limit)

    print(f"Maximal Arrival Rates (max lambdas) for 𝛼 = {alpha}: {max_lambda_values}")
    print(f"Remaining Battery Charges: {remaining_battery_values}")
    print(f"Control Variates: {control_variate_values}")
