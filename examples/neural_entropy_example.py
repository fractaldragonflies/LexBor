"""
Created on Fri May 15 19:03:57 2020

@author: johnmiller

Entropy distribution analysis example for neural language model.
"""

# Import Python standard libraries
import io
import statistics
from pathlib import Path
from matplotlib import pyplot as plt
from tabulate import tabulate
from unidecode import unidecode

# Build namespace
import lexbor.neural_entropy as ent
import lexbor.neural_lm as lm
import lexbor.neural_data as ndat
import lexbor.config as cfg
import lexbor.util as util
import lexbor.wold as wold
import lexbor.foreign_data as fdat

output_path = Path(cfg.BaseSettings().output_path).resolve()


def plot_sparse_crossentropy(
        measures=None,
        title=None,
        output_path=None,
):
    # From Keras example
    # Plot training & validation crossentropy values
    plt.plot(measures['metric_sparse_categorical_crossentropy'][1:])
    if 'val_metric_sparse_categorical_crossentropy' in measures.keys():
        plt.plot(measures['val_metric_sparse_categorical_crossentropy'][1:])
    if title:
        plt.title(title)
    plt.ylabel('Model Crossentropy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    if output_path:
        plt.savefig(output_path, dpi=600)

    plt.close()


def describe_entropies(data):
    n = len(data)
    avg = statistics.mean(data)
    std = statistics.stdev(data)
    min_ = min(data)
    max_ = max(data)
    return [n, avg, std, min_, max_]


def print_entropy_statistics(stats, title="", row_names=None):
    table = [[title, "n", "mean", "stdev", "min", "max"]]
    for name, row in zip(row_names, stats):
        table.append([name] + row)
    print(tabulate(table, tablefmt="pipe", headers="firstrow", floatfmt=".3f"))


def report_entropy_statistics(model, data):
    # Report descriptive statistics.
    stats = []
    fit_entropies = model.calculate_entropies(data.get_data_tokens_ids(data.fit))
    stats.append(describe_entropies(fit_entropies))
    if len(data.val) > 0:
        val_entropies = model.calculate_entropies(data.get_data_tokens_ids(data.val))
        stats.append(describe_entropies(val_entropies))
    if len(data.testing) > 0:
        test_entropies = model.calculate_entropies(data.get_data_tokens_ids(data.testing))
        stats.append(describe_entropies(test_entropies))
    return stats


# Inserted code from notebook used to run attention experiments.
def get_data_for_language(language=None,
                          form='Tokens',
                          test_split=0.15):
    wold_db = wold.get_wold_access()
    language_ = wold.check_wold_languages(wold_db, language)[0]

    print(f'Language {language_}.')
    table = wold_db.get_table(language=language_, form=form, classification='Borrowed')
    train, test = util.train_test_split(table, split=test_split)
    return train, test


def get_data_from_file(filename=None, test_split=0.15):
    # Form assumed to be Tokens.
    table = fdat.get_pybor_format_data(filename)
    train, test = util.train_test_split(table, test_split)
    # print(len(train), len(test), train[:5], test[:5])
    return train, test


def get_data_for_basis(language, basis, test_split, filename=None):
    if filename:
        train, test = get_data_from_file(filename, test_split=test_split)
    else:
        train, test = get_data_for_language(language, test_split=test_split)

    if basis == "native":
        train = [row for row in train if row[2] == 0]
        test = [row for row in test if row[2] == 0]
    elif basis == "loan":
        train = [row for row in train if row[2] == 1]
        test = [row for row in test if row[2] == 1]
    # Otherwise use full train and test.

    return train, test


def run_model(model, data, basis):
    results = model.train(data.trainer, data.validator)
    plot_sparse_crossentropy(results)

    model.show_test(data.tester)

    # Report descriptive statistics -- functions defined in example application above.
    stats = report_entropy_statistics(model, data)
    title = f"{basis} basis"
    row_names = ["Fit", "Validate", "Test"]
    print_entropy_statistics(stats, title, row_names)


def save_embeddings(data, model, filename):
    weights = model.get_embedding_weights()
    vocab = data.vocab.vocab
    # Write them out.
    out_emb = io.open(filename, 'w', encoding='utf-8')
    # print(f'Vocab {vocab}')
    for key, segment in vocab.items():
        if isinstance(key, int) and key > 0:
            vec = weights[key]
            out_emb.write(segment + "\t" + '\t'.join([str(x) for x in vec]) + "\n")

    out_emb.close()


def run_analysis_entropy(language=None,
                         basis="all",
                         model_type="recurrent",
                         test_split=0.15,
                         filename=None,
                         embeddings_filename=None,
                         settings=None):

    train, test = get_data_for_basis(language,
                                     basis=basis,
                                     test_split=test_split,
                                     filename=filename)

    # Experiment with 'backward' in LSTM models.
    data = ndat.NeuralData(train, test, settings=settings)
    # Define model based on model_type and data module
    model_ = None
    if model_type == "multiheadattention":
        model_ = ent.NeuralEntropyMultiheadAttention
    elif model_type == "recurrentattention":
        model_ = ent.NeuralEntropyRecurrentAttention
    elif model_type == "recurrent":
        model_ = ent.NeuralEntropyRecurrent
    elif model_type == "recurrent_lm_for":
        model_ = lm.NeuralEntropyRecurrentLMFor
    elif model_type == "recurrent_lm_for_back":
        model_ = lm.NeuralEntropyRecurrentLMForBack
    else:
        print(f"Unknown model type {model_type}.")
        exit(-99)

    language = unidecode(language)
    model = model_(
        data.vocab.size, language=language, basis=basis,
        series="Entropy-" + model_type, settings=settings)

    run_model(model, data, basis)

    if embeddings_filename:
        save_embeddings(data, model, embeddings_filename)


# Main
if __name__ == "__main__":

    # import os
    # Prevent use of GPU.
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # import tensorflow as tf
    # tf.config.set_visible_devices([], 'GPU')
    # visible_devices = tf.config.get_visible_devices()
    # for device in visible_devices:
    #     assert device.device_type != 'GPU'

    # tf.config.threading.set_intra_op_parallelism_threads(4)
    # tf.config.threading.set_inter_op_parallelism_threads(4)

    settings = cfg.RecurrentSettings(rnn_cell_type='LSTM',
                                     skip_step=3,
                                     output_encoding='sparse',
                                     verbose=1, tf_verbose=0,
                                     val_split=0.2,
                                     print_summary=True,
                                     basis='loan',
                                     # LMForBack settings
                                     epochs=200,
                                     learning_rate=0.0035,
                                     learning_rate_schedule='TR',
                                     embedding_dropout=0.5,
                                     decoding_dropout=0.5,
                                     logit_dropout=0.0,
                                     kernel_l2=0.01,
                                     dlm_is_plus=True,  # Experiment with this.
                                     attention_causal=True,

                                     # *** Multihead Attention settings
                                     # epochs=80,
                                     # learning_rate=0.0055,
                                     # learning_rate_schedule='TR',
                                     # embedding_dropout=0.3,
                                     # attention_dropout=0.2,
                                     # transformer_dropout=0.1,
                                     # *** Recurrent settings ...
                                     # learning_rate=0.01,
                                     # epochs=45,
                                     # learning_rate_schedule='SP',
                                     # merge_embedding_dropout=0.2,
                                     # rnn_output_dropout=0.2,
                                     # recurrent_l2=0.001,
                                     )

    run_analysis_entropy(language="English",
                         basis='native',
                         model_type='multiheadattention',  # 'recurrent_lm_for_back',
                         test_split=0.0,
                         # If filename given, then look to this file instead of in WOLD.
                         # filename="tables/SK-4vowels.3col.tsv",
                         # embeddings_filename="embedding/SK_4vowels_emb_F.tsv",  # to save embeddings.
                         settings=settings)
