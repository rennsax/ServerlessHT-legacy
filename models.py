from pydantic import BaseModel, PositiveFloat, PositiveInt


class Hyperparameter(BaseModel):
    batch_size: PositiveInt
    learning_rate: PositiveFloat
    momentum: PositiveFloat
