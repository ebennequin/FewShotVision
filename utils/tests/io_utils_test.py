from utils.io_utils import set_and_print_random_seed

class TestIoUtils:

    class TestSetAndPrintRandomSeed:

        def test_function_returns_different_seed_at_each_call_with_none(self):
            first_output = set_and_print_random_seed(None)
            second_output = set_and_print_random_seed(None)

            assert first_output != second_output

        def test_function_returns_same_seed_when_arguments_are_the_same(self):
            first_output = set_and_print_random_seed(0)
            second_output = set_and_print_random_seed(0)

            assert first_output == second_output
