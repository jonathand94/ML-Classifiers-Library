class ModelError(Exception):

    """
        Handle errors with respect to the model.
    """

    pass


class DimensionalityError(Exception):

    """
        Handle errors with respect to the model.
    """

    pass


class LabelsError(Exception):

    """
        Handle errors with respect to the labels associated to the classes.
    """

    pass


class MissingDataError(Exception):

    """
        Handle errors with respect to missing data in the model.
    """

    pass


class MetricError(Exception):

    """
        Handle errors with respect to the metrics to evaluate the model performance.
    """
    pass

class HyperParameterError(Exception):

    """
        Handle errors with respect to hyper parameters specified by the model.
    """
    pass