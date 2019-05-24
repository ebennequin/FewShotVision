import sys

from src.steps.method_training import MethodTraining


if __name__ == '__main__':
    args = sys.argv[1:]

    MethodTraining().apply(args)
