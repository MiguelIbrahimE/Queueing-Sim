import numpy as np

# Function to simulate an energy system
def simulate_system(lmbda, T, gamma, c, p, epsilon, B):
    """
    Simulates an energy system over a given time period.

    Parameters:
    lmbda (float): Rate of job arrivals.
    T (int): Simulatiaon time.
    gamma (float): Rescheduling parameter.
    c (float): Rescheduling cost.
    p (float): External energy rate.
    epsilon (float): Efficiency of the battery.
    B (float): Battery capacity.

    Returns:
    float: The achieved alpha value representing the ratio of purchased energy to total energy demands.
    """
    total_energy_purchased = 0
    total_energy_demands = 0
    battery_storage = 0  # Initial battery storage
    for _ in range(T):

        energy_demand = np.random.uniform(0, 1)

        intermittent_source_energy = np.random.uniform(0, 1)

        # Calculate energy available for storage or needed from storage
        energy_difference = intermittent_source_energy - energy_demand

        if energy_difference > 0:
            # Store excess energy in the battery, considering its capacity
            energy_to_store = min(energy_difference, B - battery_storage)
            battery_storage += energy_to_store
        
        else:
            # Use stored energy to meet the shortfall, if available
            energy_needed = abs(energy_difference)
            
            energy_from_battery = min(energy_needed, battery_storage)
           
            battery_storage -= energy_from_battery

            energy_shortfall = energy_needed - energy_from_battery

            # Purchase additional energy if there's still a shortfall
            if energy_shortfall > 0:
                energy_to_purchase = energy_shortfall / p
                total_energy_purchased += energy_to_purchase

        total_energy_demands += energy_demand

    achieved_alpha = total_energy_purchased / total_energy_demands if total_energy_demands else 0
    return achieved_alpha

# Define simulation parameters
lmbda = 0.1  # Rate of job arrivals
T = 1000     # Simulation time (in time units)
gamma = 0.2  # Rescheduling parameter
c = 1.0      # Rescheduling cost
p = 0.5      # External energy rate
epsilon = 0.80  # Efficiency of the battery
alpha_values = [0.0, 0.01, 0.05, 0.1]  # Set of alpha values to test

# Function to find the minimal battery capacity needed to achieve a target alpha value
def find_minimal_battery_capacity(target_alpha, tolerance=0.01):
    """
    Performs a binary search to find the minimal battery capacity that achieves a target alpha value.
    The target_alpha represents the maximum fraction of energy that can be purchased externally
    compared to the total energy demands of the system.

    Parameters:
    target_alpha (float): The target alpha value to achieve (maximum fraction of energy to be purchased externally).
    tolerance (float): The tolerance level for the search (in Watt-hours).

    Returns:
    float: The minimal battery capacity needed (in Watt-hours).
    """
    low = 0
    high = 1000  # Set an initial upper bound for the battery capacity

    while high - low > tolerance:
        B = (low + high) / 2
        achieved_alpha = simulate_system(lmbda, T, gamma, c, p, epsilon, B)

        # Adjust the search range based on the achieved alpha
        if achieved_alpha < target_alpha:
            high = B
        else:
            low = B

    return low

# Storing and displaying the results
results = {}

for target_alpha in alpha_values:
    # Find minimal capacity for each target alpha value
    minimal_capacity = find_minimal_battery_capacity(target_alpha)
    results[target_alpha] = minimal_capacity

# Print the results for each alpha value
for target_alpha, minimal_capacity in results.items():
    print(f"For target_alpha={target_alpha}, minimal_capacity={minimal_capacity} Wh")
