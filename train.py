import asyncio

from models import Hyperparameter


async def train(params: Hyperparameter) -> tuple[float]:
    command = ["sleep", "5"]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await process.wait()
    return (1.0,)


async def main() -> int:
    params = [
        Hyperparameter(batch_size=16, learning_rate=0.1, momentum=0.6) for _ in range(5)
    ]
    print(await asyncio.gather(*map(train, params)))
    return 1


if __name__ == "__main__":
    res = asyncio.run(main())
    print(res)
