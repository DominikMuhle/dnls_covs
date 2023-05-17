from typing import Callable, Tuple

import torch

MatchingOutput = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
# MatchingFunction = (
#     Union[Callable[[torch.Tensor, torch.Tensor, torch.Tensor, str], MatchingOutput],
#     Callable[[torch.Tensor, torch.Tensor, torch.Tensor, str, torch.Tensor], MatchingOutput],]
# )
MatchingFunction = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.device], MatchingOutput]
    