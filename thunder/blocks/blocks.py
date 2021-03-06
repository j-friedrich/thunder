from numpy import prod, rollaxis

from ..base import Base
import logging


class Blocks(Base):
    """
    Superclass for subdivisions of Images data.

    Subclasses of Blocks will be returned by an images.toBlocks() call.
    """
    _metadata = Base._metadata + ['blockshape']

    def __init__(self, values):
        super(Blocks, self).__init__(values)

    @property
    def _constructor(self):
        return Blocks

    @property
    def blockshape(self):
        return tuple(self.values.plan)

    def count(self):
        """
        Explicit count of the number of items.

        For lazy or distributed data, will force a computation.
        """
        if self.mode == 'spark':
            return self.tordd().count()

        if self.mode == 'local':
            return prod(self.values.values.shape)

    def collect_blocks(self):
        """
        Collect the blocks in a list
        """
        if self.mode == 'spark':
            return self.values.tordd().values().collect()

        if self.mode == 'local':
            return self.values.values.flatten().tolist()

    def map(self, func, dims=None, dtype=None):
        """
        Apply an array -> array function to each block
        """
        mapped = self.values.map(func, value_shape=dims, dtype=dtype)
        return self._constructor(mapped).__finalize__(self, noprop=('dtype',))

    def first(self):
        """
        Return the first element.
        """
        if self.mode == 'spark':
            return self.values.tordd().values().first()

        if self.mode == 'local':
            return self.values.first

    def toimages(self):
        """
        Convert blocks to images.
        """
        from thunder.images.images import Images

        if self.mode == 'spark':
            values = self.values.values_to_keys((0,)).unchunk()

        if self.mode == 'local':
            values = self.values.unblock()

        return Images(values)

    def toseries(self):
        """
        Converts blocks to series.
        """
        from thunder.series.series import Series

        if self.mode == 'spark':
            values = self.values.values_to_keys(tuple(range(1, len(self.shape)))).unchunk()

        if self.mode == 'local':
            values = self.values.unblock()
            values = rollaxis(values, 0, values.ndim)

        return Series(values)
