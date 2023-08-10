GLUE = {
    'cola': {
        'features': ['sentence'],
        'metrics': 'matthews_correlation',
        'model_type': 'classification',
    },
    'sst2': {
        'features': ['sentence'],
        'metrics': 'accuracy',
        'model_type': 'classification',
    },
    'mrpc': {
        'features': ['sentence1', 'sentence2'],
        'metrics': 'f1',
        'model_type': 'classification',
    },
    'qqp': {
        'features': ['question1', 'question2'],
        'metrics': 'accuracy',
        'model_type': 'classification',
    },
    'stsb': {
        'features': ['sentence1', 'sentence2'],
        'metrics': 'pearsonr',
        'model_type': 'regression',
    },
    'mnli': {
        'features': ['premise', 'hypothesis'],
        'metrics': 'accuracy',
        'model_type': 'classification',
    },
    'ax': {
        'features': ['premise', 'hypothesis'],
        'metrics': 'accuracy',
        'model_type': 'classification',
    },
    'qnli': {
        'features': ['question', 'sentence'],
        'metrics': 'accuracy',
        'model_type': 'classification',
    },
    'rte': {
        'features': ['sentence1', 'sentence2'],
        'metrics': 'accuracy',
        'model_type': 'classification',
    },
    'wnli': {
        'features': ['sentence1', 'sentence2'],
        'metrics': 'accuracy',
        'model_type': 'classification',
    },
}
