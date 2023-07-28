import asyncio
import csv
import random
import logging

import numpy as np
from deap import base, creator, tools

from train import train
from models import Hyperparameter

# np.random.seed(12)
# random.seed(12)


def initialize():
    # Objective: maximize the accuracy
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    # Individual: a list of binaries
    # `fitness` will become an member of `Individual`
    creator.create("Individual", list, fitness=creator.FitnessMin)

    ind_length = [int(np.log2(design_space[key].shape[0])) for key in design_space]

    def encode_individual(paras: Hyperparameter) -> creator.Individual:
        """Encode a hyperparameter into a binary list
        args:
            paras: a Hyperparameter instance
        returns:
            a binary list, i.e. list[int]
        """
        ind = list()
        for i, key in enumerate(design_space):
            # idx = np.where(design_space[key] == getattr(paras, key))[0][0]
            idx = np.argmin(np.abs(design_space[key] - getattr(paras, key)))
            ind.append(list(map(int, np.binary_repr(idx, width=ind_length[i]))))
        return sum(ind, [])

    def decode_individual(ind: creator.Individual) -> Hyperparameter:
        """Decode a binary list into a hyperparameter
        args:
            ind: a binary list, i.e. list[int]
        returns:
            a Hyperparameter instance
        """
        assert sum(ind_length) == len(ind), "The length of ind is not correct"
        p = 0
        paras = dict()
        for i, x in enumerate(ind_length):
            idx = int("".join(map(str, ind[i : i + x])), 2)
            paras[list(design_space)[i]] = design_space[list(design_space)[i]][idx]
            p += x
        return Hyperparameter(**paras)

    def hash_individual(ind: creator.Individual) -> int:
        """Generate an unique hash value for an individual
        args:
            ind: a binary list, i.e. list[int]
        returns:
            an integer. In fact it's the decimal representation of the binary list
        """
        return int("".join(map(str, ind)), 2)

    def hash_decode_individual(key: int) -> creator.Individual:
        return creator.Individual(map(int, np.binary_repr(key, sum(ind_length))))

    def random_individual() -> creator.Individual:
        """Generate a random individual
        returns:
            a binary list, i.e. list[int]
        """
        return creator.Individual(
            [np.random.randint(0, 2) for _ in range(sum(ind_length))]
        )

    def random_population(n: int) -> list[creator.Individual]:
        """Generate a random population
        args:
            n: the size of population
        returns:
            a list of binary lists, i.e. list[list[int]]
        """
        res = list()
        while len(res) < n:
            ind = random_individual()
            if ind not in res:
                res.append(ind)
        return res

    def handle_offline_data(path: str) -> dict[int, tuple[float]]:
        headers = (*Hyperparameter.model_fields,) + ("accuracy", "time")
        with open(path, "r") as f:
            reader = csv.DictReader(f, headers)
            data = [*reader]
        res = dict()
        for row in data:
            parameters = Hyperparameter(**{k: float(row[k]) for k in headers[:-2]})
            key = hash_individual(encode_individual(parameters))
            res[key] = (float(row["accuracy"]), float(row["time"]))
        return res

    async def evaluate_individual(
        ind: creator.Individual,
        offline_data=handle_offline_data(OFFLINE_DATA),
    ) -> tuple[float]:
        """Evaluate the fitness of an individual
        args:
            ind: the individual
        returns:
            a tuple of fitness value (test accuracy)
        """
        # if offline data is provided
        # acc, _ = offline_data[hash_individual(ind)]
        res = await train(decode_individual(ind))
        return (res[0],)

    toolbox.register("random_individual", random_individual)
    toolbox.register("random_population", random_population)
    toolbox.register("encode_individual", encode_individual)
    toolbox.register("decode_individual", decode_individual)
    toolbox.register("hash_individual", hash_individual)
    toolbox.register("hash_decode_individual", hash_decode_individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("crossover", tools.cxUniform, indpb=CROSSOVER_PROB)
    toolbox.register("mutate", tools.mutFlipBit, indpb=MUTATION_PROB)


async def main() -> None:
    # We record all searched hyperparameter sets and
    # test whether there are duplicates when adding new offsprings,
    # which truncate repeated trials, saving the time as a result
    evaluated_history: dict[int, float] = dict()

    while len(evaluated_history) < 20:
        population = toolbox.random_population(POPULATION_SIZE)

        # This step takes a lot of time
        fitness = await asyncio.gather(*map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitness):
            evaluated_history[toolbox.hash_individual(ind)] = fit[0]
            ind.fitness.values = fit
        selected_ind = toolbox.select(population, SELECT_SIZE)
        temp_ind = [toolbox.clone(ind) for ind in selected_ind]
        temp_ind += [
            toolbox.random_individual() for _ in range(POPULATION_SIZE - SELECT_SIZE)
        ]
        for ind in temp_ind:
            toolbox.mutate(ind)

        offspring: list[creator.Individual] = []
        while len(offspring) < POPULATION_SIZE:
            parents = random.sample(temp_ind, 2)
            if np.random.rand() < MATE_PROB:
                toolbox.crossover(*parents)
            new_offspring = filter(
                lambda ind: toolbox.hash_individual(ind) not in evaluated_history
                and ind not in offspring,
                parents,
            )
            offspring += list(new_offspring)

        population[:] = offspring

    print(sorted(evaluated_history.values()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    toolbox = base.Toolbox()

    # Define the hyperparameter searching space, which need 8 bits to encode
    design_space = {
        "batch_size": np.array([16, 32, 64, 128]),
        "learning_rate": np.linspace(0.001, 0.1, 16),
        "momentum": np.linspace(0.6, 0.9, 4),
    }
    CROSSOVER_PROB = 0.3
    MUTATION_PROB = 0.1
    MATE_PROB = 0.5

    POPULATION_SIZE = 8
    SELECT_SIZE = 4
    MAX_GEN = 10
    OFFLINE_DATA = "./lenet-all.csv"

    initialize()

    asyncio.run(main())
