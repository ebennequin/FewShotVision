import sys

from src.steps.embedding import Embedding



if __name__ == '__main__':
    args = sys.argv[1:]
    Embedding().apply(args)
