# FedAvg (an FL algorithm)
def average_trainable_weights_weighted(model_list, sample_counts):
    new_weights = []
    total_samples = sum(sample_counts)
    for weights in zip(*[model.trainable_weights for model in model_list]):
        weighted_sum = sum(w.numpy() * (count / total_samples)
                           for w, count in zip(weights, sample_counts))
        new_weights.append(weighted_sum)
    return new_weights


def set_trainable_weights(model, new_weights):
    for var, new in zip(model.trainable_weights, new_weights):
        var.assign(new)

global_accs = []
global_models = []
