from src.steps.embedding import Embedding

class TestEmbedding:
    args_without_method = [
        '--dataset',
        'CUB',
        '--model',
        'Conv4',
        # '--shallow',
        # 'True',
        '--method',
    ]


    def test_baseline(self):

        Embedding().apply(self.args_without_method + ['baseline'])

    def test_baseline_p(self):

        Embedding().apply(self.args_without_method + ['baseline++'])

    def test_protonet(self):

        Embedding().apply(self.args_without_method + ['protonet'])

    def test_matchingnet(self):

        Embedding().apply(self.args_without_method + ['matchingnet'])

    def test_relationnet(self):

        Embedding().apply(self.args_without_method + ['relationnet'])

    def test_relationnet_softmax(self):

        Embedding().apply(self.args_without_method + ['relationnet_softmax'])
