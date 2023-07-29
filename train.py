import asyncio
import hashlib
import logging
import pathlib

from models import Hyperparameter

EPOCH: int = 20
TRAIN_DATA_PATH: str = "mnist.tar.gz"
TEST_DATA_PATH: str = "mnist.tar.gz"
WORKER_NUM: int = 8
REINVOKE_TIME: int = 840  # 14 minutes
LAMBDA_NAME = "Hyperparameter_optimization_group_rfm"


def hash_hyperparameter(params: Hyperparameter) -> str:
    m = hashlib.sha1()
    m.update(str(params).encode())
    return m.hexdigest()


async def train(params: Hyperparameter, index: int) -> tuple[float]:
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
        "10.0",
        "--limit-epoch",
        "5",
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
        logger.info("%s: %s", params, res)
        # pathlib.Path.unlink(output_file)
    except Exception as e:
        logger.error("%s: %s (%s)", params, type(e), e)
        return (0, 0, 0)
    else:
        return (*map(float, res),)


logger = logging.getLogger(__name__)


async def main():
    logger.info(
        await train(
            Hyperparameter(batch_size=128, learning_rate=0.001, momentum=0.9), 0
        )
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = asyncio.run(main())
    logging.shutdown()
