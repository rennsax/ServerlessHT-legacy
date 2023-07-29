import asyncio
import hashlib
import logging
import pathlib
import json
from typing import Any

from models import Hyperparameter

with open("config.json", "r") as f:
    config: dict[str, Any] = json.load(f)

EPOCH: int = config["train.epoch"]
TRAIN_DATA_PATH: str = config["train.trainSet"]
TEST_DATA_PATH: str = config["train.testSet"]
WORKER_NUM: int = config["train.workerNum"]
REINVOKE_TIME: int = config["train.restartTime"]
LAMBDA_NAME: str = config["train.LambdaName"]

logger = logging.getLogger(__name__)
logger.propagate = True  # default to be True in fact
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(
    logging.Formatter(
        "{asctime} | {name} | {levelname}\n {message}\n",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{",
    )
)
logger.addHandler(stream_handler)


def hash_hyperparameter(params: Hyperparameter) -> str:
    m = hashlib.sha1()
    m.update(str(params).encode())
    return m.hexdigest()


async def train(params: Hyperparameter, index: int) -> tuple[float, ...]:
    output_file = pathlib.Path("output/" + str(index) + ".txt")
    log_output = pathlib.Path("subprocess/" + str(index) + ".txt")
    command: list[str] = [
        "python3",
        "EC2.py",
        "--total-groups",
        "1",
        "--num-parts",
        str(WORKER_NUM),
        "--port",
        f"{9000+index}",
        "--bucket-name",
        "fychttptest1",
        "--batch-size",
        f"{params.batch_size:d}",
        "--lr",
        f"{params.learning_rate:f}",
        "--momentum",
        f"{params.momentum:f}",
        "--epoch",
        f"{EPOCH}",
        "--reinvoke-time",
        str(REINVOKE_TIME),
        "--data-s3path",
        TRAIN_DATA_PATH,
        "--test-data-path",
        TEST_DATA_PATH,
        "--output-file",
        output_file.as_posix(),
        "--limit-loss",
        "0.8",
        "--limit-epoch",
        "10",
        "--lambda-name",
        LAMBDA_NAME,
    ]
    log_file = open(log_output, "w")
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=log_file,
        stderr=log_file,
    )
    logger.info("Create process with command: %s", " ".join(command))
    await proc.wait()
    logger.info("Process %d finished", index)
    log_file.close()
    try:
        with open(output_file, "r") as f:
            lines = f.readlines()
        if len(lines) == 0:
            raise RuntimeError("no output")
        res = lines[0].strip().split(",")
        if len(res) != 3:
            raise RuntimeError("invalid output")
        logger.info("Train (%s) over with result %s", params, res)
    except Exception:
        logger.exception("Train (%s) failed", params)
        return (0.0, 0.0, 0.0)
    else:
        return (*map(float, res),)


async def main():
    logger.info(
        await train(
            Hyperparameter(
                batch_size=128,
                learning_rate=0.001,
                momentum=0.9,
            ),
            0,
        )
    )


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    res = asyncio.run(main())
