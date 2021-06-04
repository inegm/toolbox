from dataclasses import dataclass, field
from math import inf
from typing import Callable, List, Tuple

from numpy import array, zeros
from numpy.random import uniform
from scipy.spatial.distance import euclidean


@dataclass
class Particle:
    position: List[float]
    search_space: List[Tuple[float, float]] = field(repr=False)
    fitness_func: Callable = field(repr=False)
    inertia: float
    cognitive: float  # Cognitive constant
    social: float  # Social constant

    def __post_init__(self):
        if (self.inertia < 0) or (self.inertia > 1):
            raise ValueError(f"inertia must be within [0, 1]")
        if (self.cognitive <= 0) or (self.cognitive >= 2):
            raise ValueError(f"cognitive must be within ]0, 2[")
        if (self.social <= 0) or (self.social >= 2):
            raise ValueError(f"social must be within ]0, 2[")

        self.velocity = zeros(len(self.position))
        self.position_best = self.position
        self.fitness = inf
        self.fitness_best = self.fitness

    def evaluate_fitness(self) -> None:
        self.fitness = self.fitness_func(self.position)
        if self.fitness < self.fitness_best:
            self.fitness_best = self.fitness
            self.position_best = self.position

    def update(self, swarm_best_pos: array) -> None:
        for ix, d in enumerate(self.position):
            inertia = self.inertia * self.velocity[ix]
            cognitive_acc = self.cognitive * uniform(0, 1)
            cognitive_pos = cognitive_acc * (self.position_best[ix] - self.position[ix])
            social_acc = self.social * uniform(0, 1)
            social_pos = social_acc * (swarm_best_pos[ix] - self.position[ix])
            self.velocity[ix] = inertia + cognitive_pos + social_pos
            self.position[ix] = self.position[ix] + self.velocity[ix]


def generate_swarm(
    n_particles: int,
    search_space: List[Tuple[float, float]],
    fitness_func: Callable,
    inertia: float,
    cognitive: float,
    social: float,
) -> List[Particle]:
    positions = array(
        [uniform(lo, hi, n_particles) for lo, hi in search_space]
    ).transpose()
    particles = []
    for position in positions:
        particles.append(
            Particle(
                position=position,
                search_space=search_space,
                fitness_func=fitness_func,
                inertia=inertia,
                cognitive=cognitive,
                social=social,
            )
        )
    return particles


def evaluate_swarm(swarm: List[Particle]) -> Particle:
    best_particle = swarm[0]
    best_fitness = best_particle.fitness
    for particle in swarm:
        particle.evaluate_fitness()
        if particle.fitness < best_fitness:
            best_fitness = particle.fitness
            best_particle = particle
    return best_particle


def update_swarm(swarm: List[Particle]) -> Particle:
    best_particle = evaluate_swarm(swarm)
    for particle in swarm:
        particle.update(best_particle.position)
    return best_particle


def pso(
    fitness_func: Callable,
    n_particles: int,
    search_space: List[Tuple[float, float]],
    inertia: float,
    cognitive: float,
    social: float,
    max_steps: int,
    min_delta: float = 0,
    min_delta_repeats: int = 10,
    verbose: bool = True,
):
    if n_particles < 2:
        raise ValueError("the swarm must count at least two particles")
    if max_steps < 1:
        raise ValueError("there should be at least one step")
    if min_delta < 0:
        raise ValueError("absolute delta should be non-negative (0 to reach max_steps)")

    swarm = generate_swarm(
        n_particles=n_particles,
        search_space=search_space,
        fitness_func=fitness_func,
        inertia=inertia,
        cognitive=cognitive,
        social=social,
    )
    delta = 0
    delta_count = 0
    step = 1
    best_particle = update_swarm(swarm)
    best_position = best_particle.position
    while step < max_steps:
        best_particle = update_swarm(swarm)
        delta = euclidean(best_position, best_particle.position)
        best_position = best_particle.position
        if verbose:
            position_str = ", ".join([f"{d:.8f}" for d in best_position])
            print(f"{step}\t[{position_str}]\t{delta:.8f}")
        if delta < min_delta:
            delta_count += 1
            if delta_count >= min_delta_repeats:
                break
        else:
            delta_count = 0
        step += 1
    return best_position


if __name__ == "__main__":

    search_space = [(-10, 10), (-10, 10)]

    def fitness(position: List[float]) -> float:
        x, y = position[0], position[1]
        return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

    print("result:", pso(fitness, 500, search_space, 0.1, 0.35, 0.45, 1000, 1e-7))
