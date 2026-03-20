class ReliableWriteupError(Exception):
    """Base exception for reliability/writeup enforcement errors."""


class InvalidFactKeyError(ReliableWriteupError):
    pass


class UnknownFactKeyError(ReliableWriteupError):
    pass


class InvalidParamKeyError(ReliableWriteupError):
    pass


class UnknownParamKeyError(ReliableWriteupError):
    pass


class InvalidFigurePathError(ReliableWriteupError):
    pass


class FactStoreFormatError(ReliableWriteupError):
    pass


class ParamStoreFormatError(ReliableWriteupError):
    pass


class SymbolicLatexError(ReliableWriteupError):
    pass
