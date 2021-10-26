import enum


class EMatrixType(str,  enum.Enum):
    BLOCK = 'block_matrix'
    SCATTER = 'scatter_matrix'

    def __str__(self):
        return self.value
