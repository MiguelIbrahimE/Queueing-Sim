import numpy as np
import queue
# Function to find the minimal battery capacity needed to achieve a target alpha value
def find_minimal_battery_capacity(lmbda, T, gamma, c, p, epsilon, target_alpha, tolerance=0.01):
    low = 0
    high = 1000  # Set an initial upper bound for the battery capacity

    while high - low > tolerance:
        B = (low + high) / 2
        achieved_alpha = simulate_queued_system(lmbda, T, gamma, c, p, epsilon, B)

        # Adjust the search range based on the achieved alpha
        if achieved_alpha > target_alpha:
            high = B
        else:
            low = B

    return low

# Function to simulate an energy system with queuing
def simulate_queued_system(lmbda, T, gamma, c, p, epsilon, B):
    total_energy_purchased = 0
    total_energy_demands = 0
    battery_storage = 0  # Initial battery storage
    job_queue = queue.Queue()
    time = 0

    while time < T:
        # Poisson process for job arrivals
        time += np.random.exponential(1/lmbda)
        if time >= T:  # No more jobs after simulation time
            break

        energy_demand = np.random.uniform(0, 1)
        job_queue.put(energy_demand)

        while not job_queue.empty():
            current_job_energy = job_queue.get()
            intermittent_source_energy = np.random.uniform(0, 1)

            # Calculate energy available for storage or needed from storage
            energy_difference = intermittent_source_energy - current_job_energy

            if energy_difference > 0:
                # Store excess energy in the battery, considering its capacity
                energy_to_store = min(energy_difference * epsilon, B - battery_storage)
                battery_storage += energy_to_store
            else:
                # Use stored energy to meet the shortfall, if available
                energy_needed = abs(energy_difference)
                energy_from_battery = min(energy_needed, battery_storage)
                battery_storage -= energy_from_battery

                energy_shortfall = energy_needed - energy_from_battery

                # Purchase additional energy if there's still a shortfall
                if energy_shortfall > 0:
                    energy_to_purchase = energy_shortfall
                    total_energy_purchased += energy_to_purchase

            total_energy_demands += current_job_energy

    achieved_alpha = total_energy_purchased / total_energy_demands if total_energy_demands else 0
    return achieved_alpha

# Define simulation parameters for the queued system
lmbda = 0.1  # Rate of job arrivals
T = 10000    # Simulation time (in time units)
gamma = 0.2  # Rescheduling parameter
c = 1.0      # Rescheduling cost
p = 0.5      # External energy rate
epsilon = 0.8  # Efficiency of the battery
alpha_values = [0.0, 0.01, 0.05, 0.1]  # Set of alpha values to test

# Storing and displaying the results for the queued system
queued_results = {}

for target_alpha in alpha_values:
    minimal_capacity = find_minimal_battery_capacity(lmbda, T, gamma, c, p, epsilon, target_alpha)
    queued_results[target_alpha] = minimal_capacity

queued_results
# Print the results for each alpha value in the queued system
for target_alpha, minimal_capacity in queued_results.items():
    print(f"For target_alpha={target_alpha}, minimal_capacity={minimal_capacity}")


