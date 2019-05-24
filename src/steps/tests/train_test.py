from src.steps.method_training import MethodTraining

class TestTrainingMethods:

    @staticmethod
    def test_step_does_not_return_error():
        args_without_method = [
            '--dataset',
            'omniglot',
            '--model',
            'Conv4',
            '--num_classes',
            '4412',
            '--stop_epoch',
            '1',
            '--shallow',
            'True',
            '--method',
        ]

        methods=[
            'baseline',
            'baseline++',
            'protonet',
            'matchingnet',
            'relationnet',
            'relationnet_softmax',
            # 'maml',
            # 'maml_approx',
        ]

        for method in methods:
            MethodTraining(args_without_method+[method]).apply()
