import numpy as np

def create_birth_death_matrix(M, p_minus, p_plus):
    Q = np.zeros((M, M))  # Initialize a matrix of zeros

    # Setting the transition probabilities
    for i in range(M):
        if i > 0:
            Q[i, i - 1] = p_minus  # Death probability
        if i < M - 1:
            Q[i, i + 1] = p_plus  # Birth probability

        # Stay probability
        if i == 0:
            Q[i, i] = 1 - p_plus
        elif i == M - 1:
            Q[i, i] = 1 - p_minus
        else:
            Q[i, i] = 1 - p_minus - p_plus

    return Q

        
class EnergySystemSimulation:
    def __init__(self, M, Q, lambda_rate, C, B, e, alpha, time_step, total_time, eta_charge, eta_discharge):
        self.M = M
        self.Q = Q
        self.lambda_rate = lambda_rate
        self.C = C
        self.B = B
        self.e = e
        self.alpha = alpha
        self.time_step = time_step
        self.total_time = total_time
        self.current_state = np.random.choice(M)
        self.battery_level = 0
        self.peak_battery_needed = 0
        self.energy_bought = 0
        self.total_energy_required = 0
        self.time = 0
        self.eta_charge = eta_charge
        self.eta_discharge = eta_discharge
        self.energy_generated_total = 0

    def next_state(self):
        rates = -np.diag(self.Q)
        total_rate = rates[self.current_state]
        if total_rate <= 0:
            return self.current_state, self.time_step

        time_to_next_state = np.random.exponential(1 / total_rate)
        next_state_prob = self.Q[self.current_state] / total_rate
        next_state = np.random.choice(self.M, p=next_state_prob)
        return next_state, time_to_next_state

    def generate_job_energy_requirement(self):
        # Adjust the range of energy requirement to be more aligned with generation capacity
        return np.random.uniform(0, 2 * self.C)  # Example adjustment

    def simulate_step(self):
        next_state, time_to_next_state = self.next_state()
        self.time += min(time_to_next_state, self.time_step)
        self.current_state = next_state

        energy_generated = (self.current_state / (self.M - 1)) * self.C  # Proportional to the state
        energy_required = self.generate_job_energy_requirement()
        self.total_energy_required += energy_required

        
        if energy_generated < energy_required:
            deficit = energy_required - energy_generated
            effective_deficit = deficit / self.eta_discharge  # Adjust for discharge efficiency
            if self.battery_level >= effective_deficit:
                self.battery_level -= effective_deficit
            else:
                if self.alpha > 0:
                    max_external_energy = self.alpha * self.total_energy_required
                    energy_to_buy = min(deficit - self.battery_level, max_external_energy - self.energy_bought)
                    self.energy_bought += energy_to_buy
                    effective_deficit -= energy_to_buy
                self.peak_battery_needed = max(self.peak_battery_needed, effective_deficit)
                self.battery_level = 0
        else:
            chargeable_energy = min(self.B - self.battery_level, 
                                    self.eta_charge * (energy_generated - energy_required))
            self.battery_level += chargeable_energy
    def run_simulation(self):
        while self.time < self.total_time:
            self.simulate_step()

        return self.peak_battery_needed
    def calculate_achieved_alpha(self):
        if self.total_energy_required > 0:
            achieved_alpha = min(self.energy_bought / self.total_energy_required, self.alpha)
            return max(0, achieved_alpha)  # Ensuring alpha is within [0, alpha]
        else:
            return 0


def main():
    eta_charge = 0.85   # Example charge efficiency
    eta_discharge = 0.75 # Example discharge efficiency
    p_minus = 0.2
    p_plus = 0.3
    C = 9.999999  # Maximum capacity
    M = 5     # Number of states in the Markov chain
    B = 100   # Starting with a maximum battery capacity
    e = 0.8   # Efficiency of battery storage
    lambda_rate = 0.2
    time_step = 1
    total_time = 2000
    alphas = [0, 0.01, 0.05, 0.1]

    # Create the birth-death matrix
    Q = create_birth_death_matrix(M, p_minus, p_plus)

    for alpha in alphas:
        simulation = EnergySystemSimulation(M, Q, lambda_rate, C, B, e, alpha, time_step, total_time, eta_charge, eta_discharge)
        min_battery_capacity = simulation.run_simulation()
        achieved_alpha = simulation.calculate_achieved_alpha()
        print(f"Minimum Capacity for a={alpha}: {min_battery_capacity} Wh, Achieved alpha: {achieved_alpha:.2f}")

# ... [rest of the code] ...
if __name__ == "__main__":
    main()

