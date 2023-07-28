import asyncio
import hashlib
import logging
import pathlib

from models import Hyperparameter

EPOCH: int = 20
TRAIN_DATA_PATH: str = "mnist.tar.gz"
TEST_DATA_PATH: str = "mnist.tar.gz"
WORKER_NUM: int = 4


def hash_hyperparameter(params: Hyperparameter) -> str:
    m = hashlib.sha1()
    m.update(str(params).encode())
    return m.hexdigest()


async def train(params: Hyperparameter, index: int) -> tuple[float]:
    output_file = pathlib.Path("output/" + str(index) + ".txt")
    log_output = pathlib.Path("subprocess/" + str(index) + ".txt")
    command = [
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
        "300",
        "--data-s3path",
        TRAIN_DATA_PATH,
        "--test-data-path",
        TEST_DATA_PATH,
        "--output-file",
        output_file,
        "--limit-loss",
        "10.0",
        "--limit-epoch",
        "5",
    ]
    log_file = open(log_output, "w")
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=log_file,
        stderr=log_file,
    )
    await proc.wait()
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
        pathlib.Path.unlink(output_file)
    except FileNotFoundError:
        logger.error("%s: output file not found", params)
    except RuntimeError as e:
        logger.error("%s: RuntimeError %s", params, e)
    except Exception as e:
        logger.debug("%s: Other exception %s", params, e)

    return (*map(float, res),)


logger = logging.getLogger(__name__)


async def main() -> int:
    with open(pathlib.Path("./lenet-all.csv"), "r") as f:
        print(f.readlines())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = asyncio.run(main())
