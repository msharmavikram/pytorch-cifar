from collections import OrderedDict
from typing import Iterator, Tuple

from torch import nn


def flatten(module: nn.Sequential) -> nn.Sequential:
    """Flattens a nested sequential module."""
    if not isinstance(module, nn.Sequential):
        raise TypeError('not sequential')

    return nn.Sequential(OrderedDict(_flatten(module)))


def _flatten(module: nn.Sequential) -> Iterator[Tuple[str, nn.Module]]:
    for name, child in module.named_children():
        # Flatten child sequential layers only.
        if isinstance(child, nn.Sequential):
            for sub_name, sub_child in _flatten(child):
                yield ('%s_%s' % (name, sub_name), sub_child)
        else:
            yield (name, child)
