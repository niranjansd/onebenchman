GLUE = {
    'cola': {
        'features': ['sentence'],
        'metrics': 'matthews_correlation',
        'model_type': 'classification',
        'eval_split': 'validation',
    },
    'sst2': {
        'features': ['sentence'],
        'metrics': 'accuracy',
        'model_type': 'classification',
        'eval_split': 'validation',
    },
    'mrpc': {
        'features': ['sentence1', 'sentence2'],
        'metrics': 'f1',
        'model_type': 'classification',
        'eval_split': 'validation',
    },
    'qqp': {
        'features': ['question1', 'question2'],
        'metrics': 'accuracy',
        'model_type': 'classification',
        'eval_split': 'validation',
    },
    'stsb': {
        'features': ['sentence1', 'sentence2'],
        'metrics': 'pearsonr',
        'model_type': 'regression',
        'eval_split': 'validation',
    },
    # 'mnli': {
    #     'features': ['premise', 'hypothesis'],
    #     'metrics': 'accuracy',
    #     'model_type': 'classification',
    #     'eval_split': 'validation_matched',
    # },
    # 'mnli(mm)': {
    #     'features': ['premise', 'hypothesis'],
    #     'metrics': 'accuracy',
    #     'model_type': 'classification',
    #     'eval_split': 'validation_mismatched',
    # },
    # 'ax': {
    #     'features': ['premise', 'hypothesis'],
    #     'metrics': 'accuracy',
    #     'model_type': 'classification',
    #     'eval_split': 'test',
    # },
    'qnli': {
        'features': ['question', 'sentence'],
        'metrics': 'accuracy',
        'model_type': 'classification',
        'eval_split': 'validation',
    },
    'rte': {
        'features': ['sentence1', 'sentence2'],
        'metrics': 'accuracy',
        'model_type': 'classification',
        'eval_split': 'validation',
    },
    'wnli': {
        'features': ['sentence1', 'sentence2'],
        'metrics': 'accuracy',
        'model_type': 'classification',
        'eval_split': 'validation',
    },
}
