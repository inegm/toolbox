from dataclasses import dataclass, field
from math import inf
from typing import Callable, List, Tuple

from numpy import array, zeros
from numpy.random import uniform


def distance(a, b):
    """Simple distance between two 1D arrays"""
    return ((a - b) ** 2).sum() ** (0.5)


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

    def update(self, swarm_best_pos: List[float]) -> None:
        for ix, _d in enumerate(self.position):
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
    """Particle Swarm Optimization.

    Starts by initializing a number `n_particles` of particles uniformally
    randomly across the `search_space`. The `search_space` defines the range of
    possible values to search for in each of the dimensions of the space.

    Subsequently at every step :

    1. each particle measures it's fitness against the given fitness function
    `fitness_func`.
    2. the best position, the one for which `fitness_func` yields the minimal
    value, is found and broadcast to the entire swarm. Note that this is an
    implementation of the **Global topology**, in which each particle is able
    to communicate with all other particles of the swarm. There are other
    topologies which may be less prone to particles getting stuck in local
    minima.
    3. given the swarm's best position, each particle moves towards it by a
    degree dependent on the optimization parameters (described below), adjusted
    by random factor.

    The position update (step 3 above) is dependent on three parameters :

    - `inertia`, in the range [0, 1] which determines the proportion by which
    each step's velocity will affect the change in position. Low values will
    slow the progress made at each step, and will potentially require more
    steps, but it will encourage *exploitation*. Higher values encourage
    *exploration*.
    - `cognitive` constant, in the range ]0, 2[, represents an individual
    particle's sense of certainty about its personal best position. A swarm with
    a high cognitive constant will favour *exploitation*.
    - `social` constant, in the range ]0, 2[, represents an individual
    particle's sense of certainty about the swarm's best position. A swarm with
    a high cognitive constant will favour *exploration*.

    The three steps above are repeated until the stopping condition is met.
    Here, the stopping condition is met if either :

    - `max_steps` is reached, or
    - there is a `min_delta` or smaller change in the swarm's best position over
    `min_delta_repeats` steps.
    """
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
        delta = distance(best_position, best_particle.position)
        best_position = best_particle.position
        if verbose:
            position_str = ", ".join([f"{d:.8f}" for d in best_position])
            print(f"{step}\t[{position_str}]\t{delta:.8f}")
        if delta <= min_delta:
            delta_count += 1
            if delta_count >= min_delta_repeats:
                break
        else:
            delta_count = 0
        step += 1
    return best_position
