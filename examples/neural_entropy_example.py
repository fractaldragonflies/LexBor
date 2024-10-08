#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 19:03:57 2020
Revised for lexbor on Sat Sep 28, 2024

@author: johnmiller

Entropy distribution analysis example for neural language model.
"""

# Import for development of model
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
tf.config.list_physical_devices("GPU")


# Import Python standard libraries
# import io
import statistics
from pathlib import Path
from matplotlib import pyplot
from tabulate import tabulate
# from unidecode import unidecode

# Build namespace
# import lexbor.neural_lm as lm
import lexbor.neural_data as neudat
import lexbor.transformer_decoder as trdec
import lexbor.config as cfg
import lexbor.util as util
import lexbor.wold as wold
import lexbor.foreign_data as fordat
import lexbor.plot as plt

output_path = Path(cfg.BaseSettings().output_path).resolve()


def plot_crossentropy(
        measures=None,
        title=None,
        output_path=None,
):
    # From Keras example
    # Plot training & validation crossentropy values
    pyplot.plot(measures['masked_crossentropy'][1:])
    if 'val_masked_crossentropy' in measures.keys():
        pyplot.plot(measures['val_masked_crossentropy'][1:])
    if title:
        pyplot.title(title)
    pyplot.ylabel('Model Crossentropy')
    pyplot.xlabel('Epoch')
    pyplot.legend(['Fit', 'Val'], loc='upper left')
    pyplot.show()

    if output_path:
        pyplot.savefig(output_path, dpi=600)
    pyplot.close()


def describe_values(values):
    # Values is list of float.
    n = len(values)
    avg = statistics.mean(values)
    std = statistics.stdev(values)
    min_ = min(values)
    max_ = max(values)
    return [n, avg, std, min_, max_]

def print_statistics(stats, title="", row_names=None):
    table = [[title, "n", "mean", "stdev", "min", "max"]]
    for name, row in zip(row_names, stats):
        table.append([name] + row)
    print(tabulate(table, tablefmt="pipe", headers="firstrow", floatfmt=".3f"))

def calculate_crossentropy_statistics(ce_results):
    # Calculate entropies, and report descriptive statistics.
    # Data is neural_data object with train, fit, val, test datasets.
    stats = []
    if "train_ce" in ce_results:
        stats.append(describe_values(ce_results["train_ce"]))
    if "fit_ce" in ce_results:
        stats.append(describe_values(ce_results["fit_ce"]))
    if "val_ce" in ce_results:
        stats.append(describe_values(ce_results["val_ce"]))
    if "test_ce" in ce_results:
        stats.append(describe_values(ce_results["test_ce"]))
    return stats

def calculate_crossentropies(neuent, data):
    results = {}
    results["fit_ce"] = neuent.calculate_entropies(data.fit_data, data.pkg)
    results["train_ce"] = results["fit_ce"][:]  # copy entropies.
    if len(data.val_data) > 0:
        results["val_ce"] = neuent.calculate_entropies(data.val_data, data.pkg)
        results["train_ce"] += results["val_ce"]
    if len(data.test_data) > 0:
        results["test_ce"] = neuent.calculate_entropies(data.test_data, data.pkg)
    # print("Count of entropies:", len(results["train_ce"]), len(results["fit_ce"]),
    #       len(results["val_ce"]), len(results["test_ce"]))
    return results

def get_data_for_language(language=None, form='Tokens'):
    wold_db = wold.get_wold_access()
    language_ = wold.check_wold_languages(wold_db, language)[0]

    table = wold_db.get_table(
        language=language_, form=form, classification='Borrowed')
    return table

def select_basis_for_data(basis, train, test):
    if basis == "Inherited":
        train = [row for row in train if row[2] == 0]
        test = [row for row in test if row[2] == 0]
    elif basis == "Borrowed":
        train = [row for row in train if row[2] == 1]
        test = [row for row in test if row[2] == 1]
    return train, test

def get_data_for_basis(language, basis, test_split=None, filename=None,
                       settings=cfg.NeuralSettings()):
    if filename:
        table = fordat.get_pybor_format_data(filename)
    else:
        table = get_data_for_language(language, form='Tokens')

    test_split = (test_split if test_split is not None
                  else settings.test_split)
    train, test = util.train_test_split(table, split=test_split)

    train, test = select_basis_for_data(basis, train, test)
    # Otherwise use full train and test as basis = All Combined

    return train, test

def analyze_both(neural_model):
    """
    Special case of both Inherited and Borrowed cases.
    Analyse the crossentropy results, especially the differences
    between crossentropies, for Inherited and Borrowed word models.

    Parameters
    ----------
    neural_model : {str: NeuralEntropy}
        Basis (Inherited or Borrwed) are keys for outside dictionary, and
        neural_model(s) are the fit language models for Inherited and Borrowed.

        # data stream of crossentropies (train_ce, fit_ce, val_ce, test_ce) are
        # contents of interior dictionaries, where keys are the datastream names.
        # Values of interior dictionary are crossentropy values corresponding to
        # words in fit, val, test datasets.  Note, the train crossentropies are
        # ordered as fit and val datasets concatenated.

    Returns
    -------
    None.

    """

    # Look at Train and Test separately.
    # Date differences between Inherited and Borrowed.
    # Report statistics on the differences.
    # Graph the distributions of differences.

def run_model(neuent, data,
              basis=None,
              settings=cfg.NeuralSettings()):
    """
    Fit and report on neural entropy model.

    Parameters
    ----------
    neuent : NeuralDecoder
         Neural entropy modeling for token representation of words.
    data : NeuralDataGen
        Manage data for use by NeuralDecoder.
    basis : str
        Whether All, Inherited, Borrowed, Both, Each.
        Basis for grouping of data from wordlists.
    Returns
    -------
    None.

    """
    measures = neuent.train(data.fit, data.val)
    neuent.show_test(data.test)

    # print("Results measures:", measures.keys())
    if settings.plot_crossentropy:
        print("Plot crossentropy measures from training.")
        plot_crossentropy(measures)

    if settings.process_crossentropies:
        # Report crossentropies.
        ce_results = calculate_crossentropies(neuent, data)
        stats = calculate_crossentropy_statistics(ce_results)
        title = f"{basis} Basis"
        row_names = ["Train", "Fit", "Val", "Test"]
        print_statistics(stats, title, row_names)

        # Plot distribution of crossentropies.
        path_out = output_path / basis
        plt.graph_distribution(ce_results["train_ce"], ce_results["test_ce"],
                               title=title, label1="Train", label2="Test",
                               output_path=path_out)
        return ce_results

    return None

# def save_embeddings(data, model, filename):
#     weights = model.get_embedding_weights()
#     vocab = data.vocab.vocab
#     # Write them out.
#     out_emb = io.open(filename, 'w', encoding='utf-8')
#     # print(f'Vocab {vocab}')
#     for key, segment in vocab.items():
#         if isinstance(key, int) and key > 0:
#             vec = weights[key]
#             out_emb.write(segment + "\t" + '\t'.join([str(x) for x in vec]) + "\n")

#     out_emb.close()

def run_single_analysis(train,
                        test=None,
                        basis=None,
                        model_type="Transformer_Decoder",
                        settings=cfg.NeuralSettings()):
    """
    Fit word model and perform crossentropy analysis.


    Parameters
    ----------
    train :
        DESCRIPTION.
    test : TYPE, optional
        DESCRIPTION. The default is None.
    basis : TYPE, optional
        DESCRIPTION. The default is None.
    model_type : TYPE, optional
        DESCRIPTION. The default is "Transformer_Decoder".
    settings : TYPE, optional
        DESCRIPTION. The default is cfg.NeuralSettings().

    Returns
    -------
    results_ce : TYPE
        DESCRIPTION.
    neuent : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.

    """

    print("** Model for basis:", basis, "**")

    # data = neudat.NeuralDataGen(train, test, settings=settings)
    data = neudat.NeuralData(train, test, settings=settings)

    neuent = trdec.NeuralDecoder(data.vocab.size, settings=settings)
    neuent.make_decoder_model()
    if settings.print_model_summary:
        print(neuent.model.summary())
    results_ce = run_model(neuent, data, basis, settings)

    return results_ce, neuent, data


def run_analysis_entropy(language=None,
                         basis="All",
                         model_type="Transformer_Decoder",
                         test_split=0.15,
                         filename=None,
                         # embeddings_filename=None,
                         settings=cfg.NeuralSettings()):

    # Only get data once. If basis is Both, then separate in analysis.
    # Train and Test remain the same for 'Both' analysis.
    train, test = get_data_for_basis(language,
                                     basis=basis,
                                     test_split=test_split,
                                     filename=filename)

    if basis in ["All", "Inherited", "Borrowed"]:
        run_single_analysis(train,
                            test=test,
                            basis=basis,
                            settings=settings)
    else:
        # Both - Much more complex.
        # Inherited only and Competing approaches.
        neural_models = {}
        data_models = {}
        data_sources = {}  # Inherited/borrowed and train/test organization.
        for basis_ in ["Inherited", "Borrowed"]:
            train_, test_ = select_basis_for_data(basis_, train, test)
            _, neuent, data = run_single_analysis(train_,
                                            test=test_,
                                            basis=basis_,
                                            settings=settings)
            neural_models[basis_] = neuent
            data_models[basis_] = data
            data_sources[basis_+"_train"] = train_
            data_sources[basis_+"_test"] = test_

        # Analyze inherited word model distributions.
        neuent = neural_models["Inherited"]
        data = data_models["Inherited"]
        # Calculate crossentropies for all training data with inherited model.
        cross_entropies = {}
        cross_entropies["Inherited_train"] = neuent.calculate_entropies(
            data_sources["Inherited_train"], data.pkg)
        cross_entropies["Borrowed_train"] = neuent.calculate_entropies(
            data_sources["Borrowed_train"], data.pkg)
        cross_entropies["Inherited_test"] = neuent.calculate_entropies(
            data_sources["Inherited_test"], data.pkg)
        cross_entropies["Borrowed_test"] = neuent.calculate_entropies(
            data_sources["Borrowed_test"], data.pkg)

        # Plot distribution of crossentropies for Inherited word model.
        path_out = output_path / "Inherited_distribution_train"
        title = "Inherited word model - Train data - Cross-entropy distributions"
        plt.graph_distribution(cross_entropies["Inherited_train"],
                               cross_entropies["Borrowed_train"],
                               title=title, label1="Inherited", label2="Borrowed",
                               output_path=path_out)

        path_out = output_path / "Inherited_distribution_test"
        title = "Inherited word model - Test data - Cross-entropy distributions"
        plt.graph_distribution(cross_entropies["Inherited_test"],
                               cross_entropies["Borrowed_test"],
                               title=title, label1="Inherited", label2="Borrowed",
                               output_path=path_out)


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

    # settings = cfg.RecurrentSettings(rnn_cell_type='LSTM',
    #                                  skip_step=3,
    #                                  output_encoding='sparse',
    #                                  verbose=1, tf_verbose=0,
    #                                  val_split=0.2,
    #                                  print_summary=True,
    #                                  basis='loan',
    #                                  # LMForBack settings
    #                                  epochs=200,
    #                                  learning_rate=0.0035,
    #                                  learning_rate_schedule='TR',
    #                                  embedding_dropout=0.5,
    #                                  decoding_dropout=0.5,
    #                                  logit_dropout=0.0,
    #                                  kernel_l2=0.01,
    #                                  dlm_is_plus=True,  # Experiment with this.
    #                                  attention_causal=True,

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
                                     # )

    run_analysis_entropy(language="English",
                         basis='Inherited',
                         model_type='Transformer_Decoder',
                         test_split=0.15)
                         # settings=settings)

    # Model Types previously used.
    #
    # Define model based on model_type and data module
    # model_ = None
    # if model_type == "multiheadattention":
    #     model_ = ent.NeuralEntropyMultiheadAttention
    # elif model_type == "recurrentattention":
    #     model_ = ent.NeuralEntropyRecurrentAttention
    # elif model_type == "recurrent":
    #     model_ = ent.NeuralEntropyRecurrent
    # elif model_type == "recurrent_lm_for":
    #     model_ = lm.NeuralEntropyRecurrentLMFor
    # elif model_type == "recurrent_lm_for_back":
    #     model_ = lm.NeuralEntropyRecurrentLMForBack
    # else:
    #     print(f"Unknown model type {model_type}.")
    #     exit(-99)

    # language = unidecode(language)
    # model = model_(
    #     data.vocab.size, language=language, basis=basis,
    #     series="Entropy-" + model_type, settings=settings)
