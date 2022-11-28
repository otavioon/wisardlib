from wisardlib.wisard import Discriminator, WiSARD


def build_symmetric_wisard(
    RAM_cls: type,
    number_of_rams_per_discriminator: int,
    number_of_discriminators: int,
    indices: list,
    tuple_size: int,
    RAM_creation_kwargs: dict = None,
    shuffle_indices: bool = True,
    count_responses: bool = True,
):
    RAM_creation_kwargs = RAM_creation_kwargs or dict()
    discriminators = []
    for i in range(number_of_discriminators):
        rams = [
            RAM_cls(**RAM_creation_kwargs)
            for j in range(number_of_rams_per_discriminator)
        ]
        discriminator = Discriminator(rams=rams, count_responses=count_responses)
        discriminators.append(discriminator)
    model = WiSARD(
        discriminators=discriminators,
        indices=indices,
        tuple_size=tuple_size,
        shuffle_indices=shuffle_indices,
    )
    return model
