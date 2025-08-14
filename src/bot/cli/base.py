"""
Base command class for CLI commands
"""

import logging
from abc import ABC, abstractmethod


class BaseCommand(ABC):
    """Base class for all CLI commands"""

    name = None
    help = None

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @classmethod
    @abstractmethod
    def add_parser(cls, subparsers):
        """Add command parser to subparsers"""
        pass

    @abstractmethod
    def execute(self, args):
        """Execute the command

        Args:
            args: Parsed command line arguments

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        pass
