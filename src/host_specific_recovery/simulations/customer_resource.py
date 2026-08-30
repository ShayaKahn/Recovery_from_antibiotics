import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed


class ConsumerResource:

    """
    Consumer-resource model with separate consumption and growth matrices:

    dR/dt = rho - R * (C @ S)

    dS/dt = S * (G.T @ R - mu)

    where:
        C : consumption matrix, shape (n_resources, n_species)
        G : growth matrix, shape (n_resources, n_species)
        rho : resource influx rates
        mu : species mortality rates
    """

    def __init__(self, n_samples, n_resources, n_species, delta, rho, mu,
                 consumption_matrix, growth_matrix, initial_resources,
                 initial_species, final_time, max_step,
                 normalize_species=True, method='RK45',
                 multiprocess=True, n_jobs=4, filter_by_event=True):

        (self.smp, self.n_resources, self.n_species, self.delta,
         self.rho, self.mu, self.C, self.G,
         self.R0, self.S0, self.final_time, self.max_step,
         self.normalize_species, self.method,
         self.multiprocess, self.n_jobs, self.filter_by_event) = self._validate_input(
            n_samples, n_resources, n_species, delta, rho, mu,
            consumption_matrix, growth_matrix, initial_resources,
            initial_species, final_time, max_step, normalize_species,
            method, multiprocess, n_jobs, filter_by_event
        )

    @staticmethod
    def _validate_input(n_samples, n_resources, n_species, delta, rho, mu,
                        consumption_matrix, growth_matrix, initial_resources,
                        initial_species, final_time, max_step, normalize_species,
                        method, multiprocess, n_jobs, filter_by_event):

        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")
        if not isinstance(n_resources, int) or n_resources <= 0:
            raise ValueError("n_resources must be a positive integer.")
        if not isinstance(n_species, int) or n_species <= 0:
            raise ValueError("n_species must be a positive integer.")
        if not (0 < delta < 1):
            raise ValueError("delta must be between 0 and 1.")
        if not isinstance(rho, np.ndarray) or rho.shape != (n_resources,):
            raise ValueError("rho must have shape (n_resources,)")
        if not isinstance(mu, np.ndarray) or mu.shape != (n_species,):
            raise ValueError("mu must have shape (n_species,)")

        if not isinstance(consumption_matrix, np.ndarray):
            raise ValueError("consumption_matrix must be numpy array.")
        if consumption_matrix.shape != (n_resources, n_species):
            raise ValueError("consumption_matrix must have shape (n_resources, n_species)")
        if np.any(consumption_matrix < 0):
            raise ValueError("consumption_matrix must be non-negative.")

        if not isinstance(growth_matrix, np.ndarray):
            raise ValueError("growth_matrix must be numpy array.")
        if growth_matrix.shape != (n_resources, n_species):
            raise ValueError("growth_matrix must have shape (n_resources, n_species)")
        if np.any(growth_matrix < 0):
            raise ValueError("growth_matrix must be non-negative.")

        if not isinstance(initial_resources, np.ndarray):
            raise ValueError("initial_resources must be numpy array.")
        if initial_resources.shape != (n_samples, n_resources):
            raise ValueError("initial_resources must have shape (n_samples, n_resources)")

        if not isinstance(initial_species, np.ndarray):
            raise ValueError("initial_species must be numpy array.")
        if initial_species.shape != (n_samples, n_species):
            raise ValueError("initial_species must have shape (n_samples, n_species)")

        if np.any(initial_resources < 0):
            raise ValueError("initial_resources must be non-negative.")
        if np.any(initial_species < 0):
            raise ValueError("initial_species must be non-negative.")

        if not isinstance(final_time, (int, float)) or final_time <= 0:
            raise ValueError("final_time must be positive.")
        if not isinstance(max_step, (int, float)) or not (0 < max_step < final_time):
            raise ValueError("max_step must be positive and smaller than final_time.")
        if not isinstance(normalize_species, bool):
            raise ValueError("normalize_species must be bool.")
        if not isinstance(multiprocess, bool):
            raise ValueError("multiprocess must be bool.")
        if not (isinstance(n_jobs, int) or n_jobs is None):
            raise ValueError("n_jobs must be int or None.")
        if not isinstance(filter_by_event, bool):
            raise ValueError("filter_by_event must be bool.")

        return (n_samples, n_resources, n_species, delta, rho, mu,
                consumption_matrix, growth_matrix, initial_resources,
                initial_species, final_time, max_step, normalize_species,
                method, multiprocess, n_jobs, filter_by_event)

    def rhs(self, t, y):

        R = y[:self.n_resources]
        S = y[self.n_resources:]

        dRdt = self.rho - R * (self.C @ S)
        dSdt = S * (self.G.T @ R - self.mu)

        return np.concatenate([dRdt, dSdt])

    #def event(self, t, y):

    #    dydt = self.rhs(t, y)
    #    species_dydt = dydt[self.n_resources:]

    #    relative_error = np.max(np.abs(species_dydt))

    #    return relative_error - self.delta

    def event(self, t, y):

        dydt = self.rhs(t, y)

        species = y[self.n_resources:]
        species_dydt = dydt[self.n_resources:]

        relative_error = np.max(
            np.abs(species_dydt) / (np.abs(species) + 1e-12)
        )

        #if relative_error < self.delta:
        #    print(relative_error)

        return relative_error - self.delta

    def solve_for_m(self, rhs_with_params, event_with_params, m):

        y0 = np.concatenate([self.R0[m, :], self.S0[m, :]])

        if self.filter_by_event:
            sol = solve_ivp(rhs_with_params, (0, self.final_time), y0,
                            max_step=self.max_step, events=event_with_params,
                            method=self.method)
        else:
            sol = solve_ivp(rhs_with_params, (0, self.final_time), y0,
                            max_step=self.max_step, method=self.method)

        return sol

    def solve(self):

        rhs_with_params = lambda t, y: self.rhs(t, y)
        event_with_params = lambda t, y: self.event(t, y)
        event_with_params.terminal = True
        event_with_params.direction = -1

        event_not_satisfied_ind = []

        final_resources = np.zeros((self.smp, self.n_resources))
        final_species = np.zeros((self.smp, self.n_species))

        if self.multiprocess:
            solutions = Parallel(n_jobs=self.n_jobs)(
                delayed(self.solve_for_m)(rhs_with_params, event_with_params, m)
                for m in range(self.smp)
            )
            print("Finished solving the consumer-resource model in parallel...")
        else:
            solutions = []

            for m in range(self.smp):

                sol = self.solve_for_m(rhs_with_params, event_with_params, m)
                solutions.append(sol)

                if m % 10 == 0:
                    print(f"Finished solving sample {m}...")

        self.solutions = solutions

        for m, sol in enumerate(solutions):

            y_final = sol.y[:, -1]
            y_final[y_final < 0] = 0.0

            final_resources[m, :] = y_final[:self.n_resources]
            final_species[m, :] = y_final[self.n_resources:]

            if self.filter_by_event and self.event(sol.t[-1], sol.y[:, -1]) > 0:
                event_not_satisfied_ind.append(m)

        if self.normalize_species:
            final_species = self.normalize_cohort(final_species)

        return final_resources, final_species, event_not_satisfied_ind

    def plot_trajectories(self, sample_index=0, figsize=(12, 8), alpha=0.75,
                          linewidth=1.2, save_path=None, show=True):

        if not hasattr(self, "solutions"):
            raise RuntimeError("Call solve() before plot_trajectories().")
        if not isinstance(sample_index, int) or not (0 <= sample_index < self.smp):
            raise ValueError("sample_index must be an integer in [0, n_samples).")

        sol = self.solutions[sample_index]
        resources = np.maximum(sol.y[:self.n_resources, :], 0.0)
        species = np.maximum(sol.y[self.n_resources:, :], 0.0)

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        axes[0].plot(sol.t, resources.T, alpha=alpha, linewidth=linewidth)
        axes[0].set_ylabel("Resource abundance")
        axes[0].set_title(f"Resource trajectories - sample {sample_index}")

        axes[1].plot(sol.t, species.T, alpha=alpha, linewidth=linewidth)
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Species abundance")
        axes[1].set_title(f"Species trajectories - sample {sample_index}")

        for ax in axes:
            ax.grid(True, alpha=0.25)

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()

        return fig, axes

    @staticmethod
    def normalize_cohort(cohort):

        if cohort.ndim == 1:
            total = cohort.sum()
            if total > 0:
                return cohort / total
            return cohort

        row_sums = cohort.sum(axis=1, keepdims=True)

        return np.divide(cohort, row_sums, out=np.zeros_like(cohort), where=row_sums != 0)
