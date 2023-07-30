import argparse
import asyncio
import csv
import json
import logging
import random
from typing import Any

import numpy as np
from deap import base, creator, tools  # type: ignore

from models import Hyperparameter
from train import logger as train_logger
from train import train

with open("config.json", "r") as f:
    config: dict[str, Any] = json.load(f)

CROSSOVER_PROB: float = config["genetic.crossoverProb"]
MUTATION_PROB: float = config["genetic.mutationProb"]
MATE_PROB: float = config["genetic.mateProb"]
MAX_EVALUATED_INDIVIDUAL: int = config["genetic.maxEvaluatedIndividual"]
POPULATION_SIZE: int = config["genetic.population.size"]
SELECT_SIZE: int = config["genetic.population.selectNumber"]

logger = logging.getLogger(__name__)
logger.propagate = True  # default to be True in fact
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(
    logging.Formatter(
        "{asctime} | {name} | L{lineno} | {levelname}\n {message}\n",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{",
    )
)
logger.addHandler(stream_handler)

toolbox = base.Toolbox()
design_space = {
    "batch_size": np.array([16, 32, 64, 128]),
    "learning_rate": np.linspace(0.001, 0.1, 16),
    "momentum": np.linspace(0.6, 0.9, 4),
}
RANDOM_SEED = np.random.randint(0, 1000)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def initialize(*, offline_data=None):
    # Objective: maximize the accuracy
    creator.create(
        "FitnessMin",
        base.Fitness,
        weights=(-1.0, 0, 0) if offline_data is None else (-1.0, 0),
    )
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
        res = list()
        for i, key in enumerate(design_space):
            # idx = np.where(design_space[key] == getattr(paras, key))[0][0]
            idx = np.argmin(np.abs(design_space[key] - getattr(paras, key)))
            res.append(list(map(int, np.binary_repr(int(idx), width=ind_length[i]))))
        return creator.Individual(sum(res, []))

    def decode_individual(ind: creator.Individual) -> Hyperparameter:
        """Decode a binary list into a hyperparameter
        args:
            ind: a binary list, i.e. list[int]
        returns:
            a Hyperparameter instance
        """
        assert sum(ind_length) == len(ind), "The length of ind is not correct"
        p = 0
        params = dict()
        for i, x in enumerate(ind_length):
            idx = int("".join(map(str, ind[i : i + x])), 2)
            params[list(design_space)[i]] = design_space[list(design_space)[i]][idx]
            p += x
        return Hyperparameter(**params)

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
        res: list[creator.Individual] = list()
        while len(res) < n:
            ind = random_individual()
            if len(list(filter(lambda _ind: _ind == ind, res))) == 0:
                res.append(ind)
        return res

    def handle_offline_data(path: str) -> dict[int, tuple[float, float]]:
        headers = (*Hyperparameter.model_fields,) + ("accuracy", "time")
        with open(path, "r") as f:
            reader = csv.DictReader(f, headers)
            data = [*reader]
        res = dict()
        for row in data:
            parameters = Hyperparameter(
                batch_size=int(row["batch_size"]),
                learning_rate=float(row["learning_rate"]),
                momentum=float(row["momentum"]),
            )
            key = hash_individual(encode_individual(parameters))
            res[key] = (float(row["accuracy"]), float(row["time"]))
        return res

    if offline_data is None:
        # Calc online
        async def evaluate_individual(
            ind: creator.Individual,
            ind_idx: int,
        ) -> tuple[float, ...]:
            """Evaluate the fitness of an individual
            args:
                ind: the individual
            returns:
                a tuple of fitness value (test accuracy)
            """
            res = await train(decode_individual(ind), ind_idx)
            return (*res,)

    else:
        # Use offline data
        async def evaluate_individual(  # type: ignore
            ind: creator.Individual,
            *args,
            offline_data=handle_offline_data(offline_data),
        ) -> tuple[float, ...]:
            """Evaluate the fitness of an individual
            args:
                ind: the individual
            returns:
                a tuple of fitness value (test accuracy)
            """
            # if offline data is provided
            acc = offline_data[hash_individual(ind)]
            return (*acc,)

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
    evaluated_history: dict[int, tuple[float, ...]] = dict()

    population = toolbox.random_population(POPULATION_SIZE)
    while len(evaluated_history) < MAX_EVALUATED_INDIVIDUAL:
        # This step takes a lot of time
        fitness_task: list[asyncio.Task[tuple[float]]] = list()
        for i, ind in enumerate(population):
            fitness_task.append(
                asyncio.create_task(toolbox.evaluate(ind, i + len(evaluated_history)))
            )
        fitness: list[tuple[float, ...]] = await asyncio.gather(*fitness_task)
        for ind, fit in zip(population, fitness):
            if (
                previous_fit := evaluated_history.get(
                    hash_value := toolbox.hash_individual(ind)
                )
            ) is not None:
                logger.critical(
                    "Duplicate trial detected: %d(%s) with fitness %s",
                    toolbox.hash_individual(ind),
                    toolbox.decode_individual(ind),
                    previous_fit,
                )
            evaluated_history[hash_value] = fit
            ind.fitness.values = fit
            logger.debug("record: (%s), %.4f", toolbox.decode_individual(ind), fit)

        if len(evaluated_history) >= MAX_EVALUATED_INDIVIDUAL:
            break

        selected_ind = toolbox.select(population, SELECT_SIZE)
        temp_ind = [toolbox.clone(ind) for ind in selected_ind]
        # Add more individuals in order to boost the diversity of population
        temp_ind += toolbox.random_population(POPULATION_SIZE - SELECT_SIZE)
        for ind in temp_ind:
            del ind.fitness.values
            toolbox.mutate(ind)

        offspring: list[creator.Individual] = list()

        # tool function to check if an individual is duplicated
        def not_duplicated(
            ind: creator.Individual,
        ) -> bool:
            if toolbox.hash_individual(ind) in evaluated_history:
                return False
            for ind2 in offspring:
                if ind == ind2:
                    return False
            return True

        while (
            len(offspring) + len(evaluated_history) < MAX_EVALUATED_INDIVIDUAL
            # Possible: len(offspring) == POPULATION_SIZE + 1
            and len(offspring) < POPULATION_SIZE
        ):
            parents = random.sample(temp_ind, 2)
            parents = list(map(toolbox.clone, parents))
            if np.random.rand() < MATE_PROB:
                new_offspring = toolbox.crossover(*parents)
                offspring[:] = offspring + list(filter(not_duplicated, new_offspring))

        population[:] = offspring

    logger.info("evaluated individual number: %d", len(evaluated_history))
    logger.info(
        "evaluated history: %s",
        list(
            map(
                lambda record: (
                    toolbox.decode_individual(
                        toolbox.hash_decode_individual(record[0])
                    ),
                    record[1],
                ),
                sorted(
                    evaluated_history.items(), key=lambda kv: kv[-1][0], reverse=True
                ),
            )
        ),
    )

    if args.offline_data is None:
        total_time: float = 0
        total_cost: float = 0
        for _, t, cost in evaluated_history.values():
            total_time += t
            total_cost += cost

        logger.info("Total cost: %f; Total time: %f", total_cost, total_time)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    train_logger.setLevel(logging.INFO)
    logger.debug("Use random seed: %d", RANDOM_SEED)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--offline-data",
        type=str,
        default=None,
        help="The offline data path. If not specified, calculate online.",
    )
    args = parser.parse_args()

    # Define the hyperparameter searching space, which need 8 bits to encode

    initialize(offline_data=args.offline_data)

    asyncio.run(main())
