from abc import ABC, abstractmethod
import pytest
import inspect


class Exercise(ABC):
    @abstractmethod
    def solve(self, *args, **kwargs):
        """
        The function that implements the solution for the exercise.
        Must be overridden.
        """
        pass

    def testmethods(self):
        """
        Returns all test methods defined on this Exercise.
        """
        return [
            method
            for name, method in inspect.getmembers(self, predicate=inspect.ismethod)
            if name.startswith("test")
        ]

    def validate(self):
        """
        Validates that the exercise has at least 5 test methods.
        """
        test_count = len(self.test_methods())
        if test_count < 5:
            raise AssertionError(
                f"Expected at least 5 test methods, found {test_count}."
            )
