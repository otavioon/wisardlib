from typing import Optional, List
from wisardlib.wisard import Discriminator, WiSARD
from wisardlib.bleaching.base import Bleaching, LinearBleaching

# def build_symmetric_wisard(
#     RAM_cls: type,
#     number_of_rams_per_discriminator: int,
#     number_of_discriminators: int,
#     indices: list,
#     tuple_size: int,
#     RAM_creation_kwargs: dict = None,
#     shuffle_indices: bool = True,
#     count_responses: bool = True,
# ):
#     RAM_creation_kwargs = RAM_creation_kwargs or dict()
#     discriminators = []
#     for i in range(number_of_discriminators):
#         rams = [
#             RAM_cls(**RAM_creation_kwargs)
#             for j in range(number_of_rams_per_discriminator)
#         ]
#         discriminator = Discriminator(rams=rams, count_responses=count_responses)
#         discriminators.append(discriminator)
#     model = WiSARD(
#         discriminators=discriminators,
#         indices=indices,
#         tuple_size=tuple_size,
#         shuffle_indices=shuffle_indices,
#     )
#     return model


def build_symmetric_wisard(
    RAM_cls: type,
    number_of_rams_per_discriminator: int,
    number_of_discriminators: int,
    tuple_size: int,
    mapping: Optional[List[int]] = None,
    bleaching_method: Bleaching = None,
    RAM_creation_kwargs: dict = None,
    use_tqdm: bool = False
):
    if bleaching_method is None:
        bleaching_method = LinearBleaching()
        
    RAM_creation_kwargs = RAM_creation_kwargs or dict()
    discriminators = []
    for i in range(number_of_discriminators):
        rams = [
            RAM_cls(**RAM_creation_kwargs)
            for j in range(number_of_rams_per_discriminator)
        ]
        discriminator = Discriminator(rams=rams)
        discriminators.append(discriminator)
        
    model = WiSARD(
        discriminators=discriminators,
        tuple_size=tuple_size,
        mapping=mapping,
        bleaching_method=bleaching_method or LinearBleaching(),
        use_tqdm=use_tqdm
    )
    return model