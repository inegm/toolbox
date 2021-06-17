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
    """
    Args:
        position (List[float]): The particle's current position within the
            search space.
        search_space (List[Tuple[float, float]]): see pso.
        inertia (float): see pso.
        cognitive (float): see pso.
    """

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
        """
        Particles evaluate their fitness simply by applying the fitness
        function to their current position. They keep track of their historical
        best position, which has an effect on their update velocity.
        """
        self.fitness = self.fitness_func(*self.position)
        if self.fitness < self.fitness_best:
            self.fitness_best = self.fitness
            self.position_best = self.position

    def update(self, swarm_best_pos: List[float]) -> None:
        """
        This is the core of the PSO algorithm. At each iteration (step), every
        particle updates its position in the search space according to the
        swarm's current best position, that is the position of the particle with
        the best fitness. The extent to which each particle moves towards the
        best position, its update velocity, is determined by three key constants,
        the definitions of which are worth repeating here :

        - **inertia** : A constant in the range $[0, 1]$ which determines the
            proportion by which each step's velocity will affect the change in
            position. Low values will slow the progress made at each step, and will
            potentially require more steps, but it will encourage *exploitation*.
            Higher values encourage *exploration*.
        - **cognitive** : A constant in the range $]0, 2[$, represents an individual
            particle's sense of certainty about its personal best position.
            A swarm with a high cognitive constant will favour *exploitation*.
        - **social** : A constant in the range $]0, 2[$, represents an individual
            particle's sense of certainty about the swarm's best position.
            A swarm with a high cognitive constant will favour *exploration*.


        Args:
            swarm_best_pos: The current position of the Particle with the best
                fitness.
        """
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
    """
    This is the initialization step of the PSO algorithm, it is called only once.

    Args:
        n_particles: More particles will slow the performance but will increase
            the likelihood of finding a global minimum
        search_space: A range, for each dimension, within which to search
        fitness_func: The function to minimize
        inertia: A constant in the range [0, 1] which determines the proportion
            by which each step's velocity will affect the change in position.
            Low values will slow the progress made at each step, and will
            potentially require more steps, but it will encourage *exploitation*.
            Higher values encourage *exploration*.
        cognitive: A constant in the range ]0, 2[, represents an individual
            particle's sense of certainty about its personal best position.
            A swarm with a high cognitive constant will favour *exploitation*.
        social: A constant in the range ]0, 2[, represents an individual
            particle's sense of certainty about the swarm's best position.
            A swarm with a high cognitive constant will favour *exploration*.

    Returns:
        A list of [Particles][toolbox.algorithms.optimization.swarm.Particle]
    """
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
    """
    At each iteration (step) of the PSO algorithm, the particles evaluate their
    fitness values by calling their [Particle.evaluate_fitness]
    [toolbox.algorithms.optimization.swarm.Particle.evaluate_fitness] methods.
    The particle with the best fitness determines the swarm's best position for
    that iteration.

    Args:
        A list of [Particles][toolbox.algorithms.optimization.swarm.Particle]
        belonging to the swarm.

    Returns:
        The particle with the best position.
    """
    best_particle = swarm[0]
    best_fitness = best_particle.fitness
    for particle in swarm:
        particle.evaluate_fitness()
        if particle.fitness < best_fitness:
            best_fitness = particle.fitness
            best_particle = particle
    return best_particle


def update_swarm(swarm: List[Particle]) -> Particle:
    """
    At each iteration (step) of the PSO algorithm, after discovering the swarm's
    best position with [evaluate_swarm]
    [toolbox.algorithms.optimization.swarm.evaluate_swarm], the particles update
    their positions by calling their [Particle.update]
    [toolbox.algorithms.optimization.swarm.Particle.evaluate_fitness] methods.

    Args:
        A list of [Particles][toolbox.algorithms.optimization.swarm.Particle]
        belonging to the swarm.

    Returns:
        The particle with the best position.
    """
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
) -> List[float]:
    """
    This implementation of particle-swarm optimization starts by initializing a
    number of particles uniformally randomly across the search space. The
    search space  defines the range of possible values to search for in each of
    the dimensions of the space. See [generate_swarm]
    [toolbox.algorithms.optimization.swarm.generate_swarm].

    Subsequently, at each iteration :

    1. every particle measures its fitness against the given fitness function.
    See [Particle.evaluate_fitness]
    [toolbox.algorithms.optimization.swarm.Particle.evaluate_fitness].
    2. the best position - the one for which the fitness function yields the minimal
    value - is found and broadcast to the entire swarm. Note that this is an
    implementation of the **Global topology**, in which each particle is able
    to communicate with all other particles of the swarm. There are other
    topologies which may be less prone to particles getting stuck in local
    minima. See [evaluate_swarm]
    [toolbox.algorithms.optimization.swarm.evaluate_swarm].
    3. given the swarm's best position, each particle moves towards it by a
    degree dependent on the optimization parameters (described below), adjusted
    by a random factor. See [update_swarm]
    [toolbox.algorithms.optimization.swarm.update_swarm].

    The position update (step 3 above) is dependent on the three parameters :

    - `inertia`
    - `cognitive`
    - `social`

    The three steps above are repeated until the stopping condition is met.
    Here, the stopping condition is met if either :

    - `max_steps` is reached, or
    - there is a `min_delta` or smaller change in the swarm's best position
    repeatedly for `min_delta_repeats` steps.

    Args:
        fitness_func: The function to minimize
        n_particles: More particles will slow the performance but will increase
            the likelihood of finding a global minimum
        search_space: A range, for each dimension, within which to search
        inertia: A constant in the range $[0, 1]$ which determines the proportion
            by which each step's velocity will affect the change in position.
            Low values will slow the progress made at each step, and will
            potentially require more steps, but it will encourage *exploitation*.
            Higher values encourage *exploration*.
        cognitive: A constant in the range $]0, 2[$, represents an individual
            particle's sense of certainty about its personal best position.
            A swarm with a high cognitive constant will favour *exploitation*.
        social: A constant in the range $]0, 2[$, represents an individual
            particle's sense of certainty about the swarm's best position.
            A swarm with a high cognitive constant will favour *exploration*.
        max_steps: A stopping condition.
        min_delta: In combination with `min_delta_repeats` defines a stopping
            condition based on the change in distance travelled across the search
            space between steps.
        min_delta_repeats: The number of times a value smaller than `min_delta`
            is observed before the optimization is stopped.
        verbose: Set to False to silence printing at each step.

    Returns:
        The final best position, and hopefully the global minimum.

    Examples:

        >>> def booth(x, y):
                return (x + 2*y - 7)**2 + (2*x + y - 5)**2
        >>> x, y = pso(
                fitness_func=booth,
                n_particles=1000,
                search_space=[(-10, 10), (-10, 10)],
                inertia=0.2,
                cognitive=0.4,
                social=0.6,
                max_steps=100,
                min_delta=0.00001,
                min_delta_repeats=10,
                verbose=True
            )

            1       [0.90367590, 3.07077955]        0.20555572
            2       [0.90864515, 3.07453398]        0.00000000
            3       [0.91783424, 3.14256790]        0.09377374
            4       [1.01895113, 2.98260133]        0.17020089
            5       [1.00240149, 3.05197469]        0.05203608
            6       [0.98202492, 2.98871176]        0.06888439
            7       [0.99610861, 2.99957312]        0.01907586
            8       [0.99225189, 3.00850750]        0.01124424
            9       [0.99971696, 3.00322148]        0.00939563
            10      [1.00077373, 3.00156704]        0.00204074
            11      [1.00038588, 2.99972079]        0.00187193
            12      [1.00008264, 2.99990549]        0.00026376
            13      [0.99999774, 2.99996748]        0.00009338
            14      [1.00004893, 2.99993361]        0.00004135
            15      [1.00008673, 2.99996720]        0.00006891
            16      [1.00000604, 2.99997651]        0.00006335
            17      [1.00002732, 2.99998583]        0.00002063
            18      [1.00000184, 3.00000745]        0.00002983
            19      [0.99999714, 2.99999854]        0.00000854
            20      [0.99999807, 2.99999881]        0.00000145
            21      [1.00000006, 2.99999973]        0.00000165
            22      [1.00000015, 3.00000011]        0.00000038
            23      [0.99999995, 2.99999995]        0.00000016
            24      [1.00000003, 3.00000010]        0.00000016
            25      [1.00000002, 3.00000002]        0.00000008
            26      [1.00000003, 2.99999998]        0.00000003
            27      [1.00000000, 3.00000000]        0.00000002
            28      [1.00000000, 3.00000000]        0.00000000

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
