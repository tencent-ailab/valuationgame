import logging
import numpy as np
from shap import links
from shap.explainers._explainer import Explainer
from shap.models import Model
from shap.utils import MaskedModel

from utils.utils_fea import compute_true_vals_impl

log = logging.getLogger('shap')


class ExactNew(Explainer):
    """ Computes SHAP values via an optimized exact enumeration.

    This works well for standard Shapley value maskers for models with less than ~15 features that vary
    from the background per sample. It also works well for Owen values from hclustering structured
    maskers when there are less than ~100 features that vary from the background per sample. This
    explainer minmizes the number of function evaluations needed by ordering the masking sets to
    minimize sequential differences. This is done using gray codes for standard Shapley values
    and a greedly sorting method for hclustering structured maskers.
    """

    def __init__(self, model, masker, link=links.identity, feature_names=None, type="shap", tempe=1):
        """ Build an explainers.Exact object for the given model using the given masker object.

        Parameters
        ----------
        model : function
            A callable python object that executes the model given a set of input data samples.

        masker : function or numpy.array or pandas.DataFrame
            A callable python object used to "mask" out hidden features of the form `masker(mask, *fargs)`.
            It takes a single a binary mask and an input sample and returns a matrix of masked samples. These
            masked samples are evaluated using the model function and the outputs are then averaged.
            As a shortcut for the standard masking used by SHAP you can pass a background data matrix
            instead of a function and that matrix will be used for masking. To use a clustering
            game structure you can pass a shap.maskers.TabularPartitions(data) object.

        link : function
            The link function used to map between the output units of the model and the SHAP value units. By
            default it is shap.links.identity, but shap.links.logit can be useful so that expectations are
            computed in probability units while explanations remain in the (more naturally additive) log-odds
            units. For more details on how link functions work see any overview of link functions for generalized
            linear models.
        """
        super(ExactNew, self).__init__(model, masker, link=link, feature_names=feature_names)

        # here the output model is y
        self.model = Model(model)
        self.type = type
        self.tempe = tempe

        self._gray_code_cache = {}  # used to avoid regenerating the same gray code patterns

    def __call__(self, *args, max_evals=100000, main_effects=False, error_bounds=False, batch_size="auto",
                 interactions=1, silent=False):
        """ Explains the output of model(*args), where args represents one or more parallel iterators.
        """

        # we entirely rely on the general call implementation, we override just to remove **kwargs
        # from the function signature
        return super(ExactNew, self).__call__(
            *args, max_evals=max_evals, main_effects=main_effects, error_bounds=error_bounds,
            batch_size=batch_size, interactions=interactions, silent=silent
        )

    def _cached_gray_codes(self, n):
        if n not in self._gray_code_cache:
            self._gray_code_cache[n] = gray_code_indexes(n)
        return self._gray_code_cache[n]

    def explain_row(self, *row_args, max_evals, main_effects, error_bounds, batch_size, outputs, interactions, silent):
        """ Explains a single row and returns the tuple (row_values, row_expected_values, row_mask_shapes).
        :return
        {
            "values": (num_fea, num_y),
            "expected_values":  (num_y, 1),
            "mask_shapes": (num_fea, ),
            "main_effects": (num_fea, num_y) or None,
            "clustering": getattr(self.masker, "clustering", None)
        }
        """

        _, fm, inds, outputs = self.calculate_mask_output(batch_size, max_evals, outputs, row_args)
        bin_index = gray_to_index(len(inds))

        outputs_binorder = np.zeros(outputs.shape)
        # gray_deci2bin_deci = {}
        for i, j in enumerate(bin_index):
            # gray_deci2bin_deci[j] = i
            outputs_binorder[j] = outputs[i]

        # outputs_binorder = [outputs[x] for x in bin_index]
        # outputs_binorder = np.array(outputs_binorder)
        # outputs_binorder = outputs
        self.all_outputs = outputs_binorder
        n_class = len(outputs_binorder[0])
        row_values = []
        errors = []
        for i in range(n_class):
            res, error = compute_true_vals_impl(len(inds), outputs_binorder[..., i], type=self.type, tempe=self.tempe)
            row_values.append(res)
            errors.append(error)
        # n_players * n_class
        row_values = np.array(row_values).T

        return {
            "values": row_values,
            "expected_values": outputs[0],
            "mask_shapes": fm.mask_shapes,
            "main_effects": None,
            "clustering": None,
            "errors": errors
        }

    def calculate_mask_output(self, batch_size, max_evals, outputs, row_args):
        # build a masked version of the model for the current input sample
        fm = MaskedModel(self.model, self.masker, self.link, *row_args)
        # do the standard Shapley values
        inds = None
        # see which elements we actually need to perturb
        inds = fm.varying_inputs()
        # make sure we have enough evals
        if max_evals is not None and max_evals != "auto" and max_evals < 2 ** len(inds):
            raise Exception(
                f"It takes {2 ** len(inds)} masked evaluations to run the Exact explainer on this instance, but max_evals={max_evals}!"
            )
        # generate the masks in gray code order (so that we change the inputs as little
        # as possible while we iterate to minimize the need to re-eval when the inputs
        # don't vary from the background)
        delta_indexes = self._cached_gray_codes(len(inds))
        # map to a larger mask that includes the invarient entries
        extended_delta_indexes = np.zeros(2 ** len(inds), dtype=np.int)
        for i in range(2 ** len(inds)):
            if delta_indexes[i] == MaskedModel.delta_mask_noop_value:
                extended_delta_indexes[i] = delta_indexes[i]
            else:
                extended_delta_indexes[i] = inds[delta_indexes[i]]
        # run the model
        # The output y 对于一个样本
        outputs = fm(extended_delta_indexes, batch_size=batch_size)

        return extended_delta_indexes, fm, inds, outputs


def gray_code_masks(nbits):
    """ Produces an array of all binary patterns of size nbits in gray code order.

    This is based on code from: http://code.activestate.com/recipes/576592-gray-code-generatoriterator/
    """
    out = np.zeros((2 ** nbits, nbits), dtype=np.bool)
    li = np.zeros(nbits, dtype=np.bool)

    for term in range(2, (1 << nbits) + 1):
        if term % 2 == 1:  # odd
            for i in range(-1, -nbits, -1):
                if li[i] == 1:
                    li[i - 1] = li[i - 1] ^ 1
                    break
        else:  # even
            li[-1] = li[-1] ^ 1

        out[term - 1, :] = li
    return out


def gray_code_indexes(nbits):
    """ Produces an array of which bits flip at which position.

    We assume the masks start at all zero and -1 means don't do a flip.
    This is a more efficient represenation of the gray_code_masks version.
    """
    out = np.ones(2 ** nbits, dtype=np.int) * MaskedModel.delta_mask_noop_value
    li = np.zeros(nbits, dtype=np.bool)
    for term in range((1 << nbits) - 1):
        if term % 2 == 1:  # odd
            for i in range(-1, -nbits, -1):
                if li[i] == 1:
                    li[i - 1] = li[i - 1] ^ 1
                    out[term + 1] = nbits + (i - 1)
                    break
        else:  # even
            li[-1] = li[-1] ^ 1
            out[term + 1] = nbits - 1
    return out


def bool2int(x):
    y = 0
    for i, j in enumerate(x):
        y += j << i
    return y


def gray_to_index(n):
    gray_array = gray_code_masks(n)
    gray_array = gray_array.astype(int)
    # index = [bool2int(x[::-1]) for x in gray_array]
    index = [bool2int(x) for x in gray_array]
    return index
