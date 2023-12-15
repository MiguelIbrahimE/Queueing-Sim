import numpy as np

def simulate_system(lmbda, T, gamma, c, p, epsilon, B, on_probability):
    total_energy_purchased = 0
    total_energy_demands = 0
    battery_storage = 0

    for _ in range(T):
        # Determine if the energy source is ON or OFF using random choice
        is_source_on = np.random.rand() < on_probability

        # Sample job from the Poisson stream with higher energy demand
        if np.random.rand() < lmbda:
            energy_demand = np.random.uniform(0.5, 2)  # Increase minimum energy demand
            total_energy_demands += energy_demand

            # Generate intermittent source energy based on ON/OFF state
            intermittent_source_energy = np.random.uniform(0, 1) if is_source_on else 0

            # Calculate energy available for storage or needed from storage
            energy_difference = intermittent_source_energy - energy_demand

            if energy_difference > 0:
                energy_to_store = min(energy_difference * epsilon, B - battery_storage)
                battery_storage += energy_to_store
            else:
                energy_needed = abs(energy_difference)
                energy_from_battery = min(energy_needed, battery_storage)
                battery_storage -= energy_from_battery

                energy_shortfall = energy_needed - energy_from_battery
                if energy_shortfall > 0:
                    energy_to_purchase = energy_shortfall / p
                    total_energy_purchased += energy_to_purchase

    achieved_alpha = total_energy_purchased / total_energy_demands if total_energy_demands else 0
    return achieved_alpha

def find_minimal_battery_capacity(target_alpha, tolerance, on_probability):
    lmbda = 0.1  # Rate of job arrivals
    T = 1000     # Simulation time
    gamma = 0.2  # Rescheduling parameter
    c = 1.0      # Rescheduling cost
    p = 0.5      # External energy rate
    epsilon = 0.8  # Efficiency of the battery

    low = 0
    high = 1000  # Set an initial upper bound for the battery capacity

    while high - low > tolerance:
        B = (low + high) / 2
        achieved_alpha = simulate_system(lmbda, T, gamma, c, p, epsilon, B, on_probability)

        if achieved_alpha < target_alpha:
            high = B
        else:
            low = B

    return low

# Main execution
on_probability = 0.5  # Probability that the energy source is ON
alpha_values = [0.0, 0.01, 0.05, 0.1]  # Set of alpha values to test

# Storing and displaying the results
results = {}
for target_alpha in alpha_values:
    minimal_capacity = find_minimal_battery_capacity(target_alpha, 0.01, on_probability)
    results[target_alpha] = minimal_capacity

for target_alpha, minimal_capacity in results.items():
    print(f"For target_alpha={target_alpha}, minimal_capacity={minimal_capacity} Wh")
