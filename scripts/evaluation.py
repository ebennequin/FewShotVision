import sys

from src.steps.method_evaluation import MethodEvaluation

if __name__ == '__main__':
    args = sys.argv[1:]
    MethodEvaluation().apply(args)
