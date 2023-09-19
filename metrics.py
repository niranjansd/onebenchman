from datasets import load_metric
import numpy as np
def compute_mattcorr(eval_pred):
    predictions, labels = eval_pred
    metric = load_metric("matthews_correlation")
    return metric.compute(predictions=predictions.argmax(-1),
                          references=labels)


def compute_acc(eval_pred):
    predictions, labels = eval_pred
    metric = load_metric("accuracy")
    return metric.compute(predictions=predictions.argmax(-1),
                          references=labels)

def compute_f1acc(eval_pred):
    predictions, labels = eval_pred
    metric = load_metric("f1")
    f1_score = metric.compute(predictions=predictions.argmax(-1),
                              references=labels)
    metric = load_metric("accuracy")
    acc = metric.compute(predictions=predictions.argmax(-1),
                          references=labels)
    return {"f1": f1_score['f1'], "accuracy": acc['accuracy']}

def compute_corr(eval_pred):
    predictions, labels = eval_pred
    metric_pearson = load_metric("pearsonr")
    metric_spearman = load_metric("spearmanr")
    predictions = predictions[0] if isinstance(predictions, tuple) else predictions
    predictions = np.squeeze(predictions)
    pearson_corr = metric_pearson.compute(predictions=predictions,
                                          references=labels)
    spearman_corr = metric_spearman.compute(predictions=predictions,
                                            references=labels)
    return {"pearson": pearson_corr['pearsonr'],
            "spearmanr": spearman_corr['spearmanr']}


ACC_TASKS = ["sst2", "qqp", "mnli", "qnli", "rte", "wnli"]
ACC_TASKS += ["ag_news", "amazon_polarity", "amazon_reviews_multi", "dbpedia_14", "yahoo_answers_topics", "yelp_polarity", "yelp_review_full"]
def compute_metrics_fn(task):
    if task == "cola":
      return compute_mattcorr
    elif task == "mrpc":
      return compute_f1acc
    elif task == "stsb":
      return compute_corr
    elif task in ACC_TASKS:
      return compute_acc
    else:
        raise ValueError("Unknown task")
