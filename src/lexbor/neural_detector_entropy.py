from pathlib import Path

import abc
import attr
from keras.backend import clear_session

import pybor2.neural_entropy as entr
import pybor2.neural_lm as lm
import pybor2.neural_data as dat
from pybor2 import util
import pybor2.config as cfg

output_path = Path(cfg.BaseSettings().output_path).resolve()
logger = util.get_logger(__name__)


def dispose():
    """Dispose of model to prevent memory leak.
    Problem is with tf not releasing memory."""
    clear_session()


@attr.s
class NeuralDetector:
    """
    Construct and train neural model of native word entropy distribution,
    and used the distribution to predict native versus loan words.

    Train neural model on native word data and on loan word data if dual.
    Determine cutpoint to detect loan words if native, or entropy bias if dual
    Predict words as native or loan.
    """

    __metaclass__ = abc.ABCMeta

    training = attr.ib()
    testing = attr.ib(default=[])
    val_split = attr.ib(default=None)
    model_type = attr.ib(default="")
    language = attr.ib(default="")
    series = attr.ib(default="")
    settings = attr.ib(default=cfg.RecurrentSettings())

    model = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.language = self.language or self.settings.language
        self.series = self.series or self.settings.series
        self.model_type = self.model_type or self.settings.model_type
#         if self.settings.model_type and self.model_type != self.settings.model_type:
#             raise ValueError("Conflicting self and settings model_types: "+
#                              self.model_type+' '+self.settings.model_type)

        all_tokens = [row[1] for row in self.training]
        all_tokens += [row[1] for row in self.testing] if self.testing else []
        self.vocab = dat.Vocab(data=all_tokens)

        # Set the neural entropy class.
        if self.model_type == "multiheadattention":
            self.model = entr.NeuralEntropyMultiheadAttention
        elif self.model_type == "recurrentattention":
            self.model = entr.NeuralEntropyRecurrentAttention
        elif self.model_type == "recurrent_lm_for":
            self.model = lm.NeuralEntropyRecurrentLMFor
        elif self.model_type == "recurrent_lm_for_back":
            self.model = lm.NeuralEntropyRecurrentLMForBack
        elif self.model_type == "recurrent":
            self.model = entr.NeuralEntropyRecurrent
        else:
            raise ValueError("Unknown entropy model "+self.model_type)

    @abc.abstractmethod
    def train(self):
        """Train method implemented by native and dual subclasses"""
        return

    @abc.abstractmethod
    def evaluate_train(self):
        """Model evaluate_train method implemented by native and dual subclasses"""
        return

    @abc.abstractmethod
    def evaluate_test(self):
        """Model evaluate_test method implemented by native and dual subclasses"""

    @abc.abstractmethod
    def predict_tokens(self, tokens):
        """Predict tokens method implemented by native and dual subclasses"""
        return

    def predict_data(self, data):
        if not data or len(data) == 0:
            logger.warning("No data to use in prediction.")
            return []

        ids = [row[0] for row in data]
        tokens = [row[1] for row in data]
        predictions = self.predict_tokens(tokens)
        return list(zip(ids, tokens, predictions))

    def predict(self, token):
        return self.predict_data([["", token]])[0][2]


@attr.s
class NeuralDetectorEntropies(NeuralDetector):

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

        native_training = [row for row in self.training if row[2] == 0]
        loan_training = [row for row in self.training if row[2] == 1]

        # if self.model_type == "recurrent_lm_for_back":
        #     data_model = dat.NeuralDataForBack
        # else:
        #     data_model = dat.NeuralData

        self.native_data = dat.NeuralData(
            training=native_training,
            testing=[row for row in self.testing if row[2] == 0]
            if self.testing
            else [],
            vocab=self.vocab,
            val_split=self.val_split,
            settings=self.settings,
        )

        # Convenient to separate out loan data always.
        self.loan_data = dat.NeuralData(
            training=loan_training,
            testing=[row for row in self.testing if row[2] == 1]
            if self.testing
            else [],
            vocab=self.vocab,
            val_split=self.val_split,
            settings=self.settings,
        )


@attr.s
class NeuralDetectorNative(NeuralDetectorEntropies):
    """
    Construct neural model for native words.
    Train neural model on native word data.
    Determine cutpoint to detect loan words.
    Predict words as native if less than empirical entropy cut-point and loan otherwise.
    """

    fraction = attr.ib(default=False)
    cut_point = attr.ib(init=False)
    native_history = attr.ib(init=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

        self.fraction = self.fraction or self.settings.fraction
        self.cut_point = None

        self.native_model = self.model(
            vocab_len=self.vocab.size,
            language=self.language,
            basis="native",
            series=self.series,
            settings=self.settings,
        )

    # Train only native model if detect type is native.
    def train(self, epochs=None):
        logger.debug("training native model")
        self.native_history = self.native_model.train(
            train_gen=self.native_data.trainer,
            val_gen=self.native_data.validator,
            epochs=epochs,
        )

    def evaluate_train(self):
        logger.debug("evaluating train on native model")
        self.native_model.evaluate_quality_measures(self.native_history)

    def evaluate_test(self):
        logger.debug("evaluating test on native model")
        self.native_model.evaluate_test(self.native_data.tester)

    def reset_cut_point(self, fraction=None):
        # Allows to set new fraction for prediction without having to start over.
        self.fraction = fraction if fraction is not None else self.settings.fraction
        self.calculate_cut_point()

    def calculate_cut_point(self):
        # Base cut-point on loan data fraction.
        train_tokens_ids = self.loan_data.get_data_tokens_ids(
            self.loan_data.training)
        entropies = self.native_model.calculate_entropies(train_tokens_ids)
        self.cut_point = util.find_frac_cut_point(
            entropies=entropies, fraction=1.0-self.fraction
        )

    def predict_tokens(self, tokens):
        # Convert to tokens_ids and then calculate entropies.
        # If necessary form empirical distribution of training entropies and determine cutpoint.
        # All tokens with entropy > cut_point are loanwords.

        if self.cut_point is None:
            self.calculate_cut_point()

        tokens_ids = [self.vocab.translate(t) for t in tokens]
        native_entropies = self.native_model.calculate_entropies(tokens_ids)
        return [int(entropy > self.cut_point) for entropy in native_entropies]


@attr.s
class NeuralDetectorDual(NeuralDetectorNative):
    """
    Construct neural models for native and loan words.
    Train corresponding neural models on native and loan word data.
    Determine entropy bias between native and loan entropy distributions.
    Predict words as native or loan based on whether native or loan calculates lesser entropy.
    """

    loan_settings = attr.ib(default=None)
    loan_history = attr.ib(init=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.loan_settings = self.loan_settings or self.settings
        self.cut_point = None

        self.loan_model = self.model(
            vocab_len=self.vocab.size,
            language=self.language,
            basis="loan",
            series=self.series,
            settings=self.loan_settings,
        )

    def train(self, epochs=None):
        epochs_n = epochs_l = epochs
        # epochs_n = epochs_l = epochs or self.settings.epochs
        # if self.settings.use_equal_steps:
        #     ratio = pow(max(0.2, len(self.loan_data.trainer.data) /
        #                     len(self.native_data.trainer.data)), 1.0/1.5)
        #     if ratio < 1.0:
        #         epochs_l = int(epochs_l/ratio)
        #     else:
        #         epochs_n = int(epochs_n*ratio)

        super().train(epochs_n)

        # Experiment with reducing learning_rate for loan model.
        # if epochs_l > epochs_n:  # Adjusted epochs for lesser loan words.
            
        logger.debug("training loan model")
        self.loan_history = self.loan_model.train(
            train_gen=self.loan_data.trainer,
            val_gen=self.loan_data.validator,
            epochs=epochs_l,
        )

    def evaluate_train(self):
        super().evaluate_train()

        logger.debug("evaluating train on loan model")
        self.loan_model.evaluate_quality_measures(self.loan_history)

    def evaluate_test(self):
        super().evaluate_test()

        logger.debug("evaluating test on loan model")
        self.loan_model.evaluate_test(self.loan_data.tester)

    def calculate_delta_entropies(self, tokens_ids):
        native_entropies = self.native_model.calculate_entropies(tokens_ids)
        loan_entropies = self.loan_model.calculate_entropies(tokens_ids)
        deltas = [
            native - loan for native, loan in zip(native_entropies, loan_entropies)
        ]

        return deltas

    def calculate_cut_point(self):
        min_val_len = 10
        if (self.settings.prediction_policy == "zero" or
                len(self.native_data.val) < min_val_len or
                len(self.loan_data.val) < min_val_len):
            self.cut_point = 0
        else:
            # Use validation data for calculation as most likely to be representative.
            data = self.loan_data.val
            tokens_ids = self.loan_data.get_data_tokens_ids(data)
            loan_deltas = self.calculate_delta_entropies(tokens_ids)

            data = self.native_data.val
            tokens_ids = self.native_data.get_data_tokens_ids(data)
            native_deltas = self.calculate_delta_entropies(tokens_ids)

            if self.settings.prediction_policy == "accuracy":
                self.cut_point = util.find_acc_cut_point_deltas(
                    native=native_deltas, loan=loan_deltas
                )
                logger.info(
                    f"Accuracy optimized prediction: cut_point ={self.cut_point:.4f}."
                )
            else:  # fscore.
                self.cut_point = util.find_fscore_cut_point_deltas(
                    native=native_deltas,
                    loan=loan_deltas,
                    beta=self.settings.fscore_beta,
                )
                logger.info(
                    f"F score optimized prediction: cut_point ={self.cut_point:.4f}."
                )

    def predict_tokens(self, tokens):
        # Convert to tokens_ids and then calculate entropies.
        # Entropy calculation and predictions depend on whether native or dual.
        if self.cut_point is None:
            self.calculate_cut_point()

        tokens_ids = [self.vocab.translate(t) for t in tokens]
        deltas = self.calculate_delta_entropies(tokens_ids)
        return [int(delta > self.cut_point) for delta in deltas]
