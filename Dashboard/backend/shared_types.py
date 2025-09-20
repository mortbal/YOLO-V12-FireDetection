from enum import Enum

class UpdateType(Enum):
    STATUS = "status"
    DEBUG = "debug"
    VERBOSE = "verbose"
    ERROR = "error"
    WARNING = "warning"