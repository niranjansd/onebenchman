

# Define datasets and features

GLUE = {
    'features': {
    'cola': ['sentence'],
    'sst2': ['sentence'],
    'mrpc': ['sentence1', 'sentence2'],
    'qqp': ['question1', 'question2'],
    'stsb': ['sentence1', 'sentence2'],
    'mnli': ['premise', 'hypothesis'],
    'ax': ['premise', 'hypothesis'],
    'qnli': ['question', 'sentence'],
    'rte': ['sentence1', 'sentence2'],
    'wnli': ['sentence1', 'sentence2'],
    },
    'metrics': {
    'cola': 'matthews_correlation',
    'sst2': 'accuracy',
    'mrpc': 'f1',
    'qqp': 'accuracy',
    'stsb': 'pearsonr',
    'mnli': 'accuracy',
    'ax': 'accuracy',
    'qnli': 'accuracy',
    'rte': 'accuracy',
    'wnli': 'accuracy',
    },
}
AEGEON = {
    'features': {
    'ag_news': ['text'],
    'amazon_polarity': ['text'],
    'amazon_reviews_multi': ['text'],
    'dbpedia_14': ['text'],
    'yahoo_answers_topics': ['text'],
    'yelp_polarity': ['text'],
    'yelp_review_full': ['text'],
    },
    'metrics': {
    'ag_news': 'accuracy',
    'amazon_polarity': 'accuracy',
    'amazon_reviews_multi': 'accuracy',
    'dbpedia_14': 'accuracy',
    'yahoo_answers_topics': 'accuracy',
    'yelp_polarity': 'accuracy',
    'yelp_review_full': 'accuracy',
    },
}

# Define datasets and features
DATASET = {'glue': GLUE, 'ag_news': AEGEON}