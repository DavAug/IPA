from typing import List

import numpy as np
import pints

from IPA.model import model as m


class SingleOutputInverseProblem(object):
    """The single output inverse problem class is based on pints.
    SingleOutputProblem https://pints.readthedocs.io/. Default error
    function is pints.SumOfSquaresError and default optimiser is
    pints.CMAES. If multiple models are provided, they are optimised
    simultaneously (naive pooled).
    """
    def __init__(self,
                 models: List[m.SingleOutputModel],
                 times: List[np.ndarray],
                 values: List[np.ndarray]
                 ):
        """Initialises a single output inverse problem with default
        error function pints.SumOfSquaresError and default optimiser
        pints.CMAES. Standard deviation in initial starting point of
        optimisation as well as restricted domain of support for
        inferred parameters is disabled by default.

        Arguments:
            models {List[m.SingleOutputModel]} -- List of naive pooled models,
            which parameters are to be inferred.
            times {List[np.ndarray]} -- Times of data points for the different
            models.
            values {List[np.ndarray]} -- State values of data points for the
            different models.

        Return:
            None
        """
        # initialise problem container
        self.problem_container = []
        for model_id, model in enumerate(models):
            self.problem_container.append(
                pints.SingleOutputProblem(model,
                                          times[model_id],
                                          values[model_id]
                                          )
                )

        # initialise error function container
        self.error_function_container = []
        for problem in self.problem_container:
            self.error_function_container.append(
                pints.SumOfSquaresError(problem)
                )

        # initialise optimiser
        self.optimiser = pints.CMAES

        # initialise fluctiations around starting point of optimisation
        self.initial_parameter_uncertainty = None

        # initialise parameter constraints
        self.parameter_boundaries = None

        # initialise outputs
        self.estimated_parameters = None
        self.objective_score = None

    def optimise(self,
                 initial_parameter: np.ndarray,
                 number_of_iterations: int = 1
                 ) -> None:
        """Find point in parameter space that optimises the error function,
        i.e. find the set of parameters that minimises the distance of the
        model to the data with respect to the error function. Optimisation
        is run number_of_iterations times and result with minimal score is
        returned. TODO: Changes this to majority vote in some meaningful way.

        Arguments:
            initial_parameter {np.ndarray} -- Starting point in parameter
            space of the optimisation algorithm. TODO: Change this, such that
            only boundaries can be provided and no initial points.
            number_of_iterations {int} -- Number of times optimisation is run.
            Default: 5 (arbitrary). TODO: Change this to correct number.

        Return:
            None
        """
        # create sum of errors measure
        error_measure = pints.SumOfErrors(self.error_function_container)

        # initialise optimisation
        optimisation = pints.OptimisationController(
            function=error_measure,
            x0=initial_parameter,
            sigma0=self.initial_parameter_uncertainty,
            boundaries=self.parameter_boundaries,
            method=self.optimiser
            )

        # run optimisation 'number_of_iterations' times
        estimate_container = []
        score_container = []
        for _ in range(number_of_iterations):
            estimates, score = optimisation.run()
            estimate_container.append(estimates)
            score_container.append(score)

        # return parameters with minimal score TODO: Change this accordingly
        min_score_id = np.argmin(score_container)
        self.estimated_parameters, self.objective_score = [
            estimate_container[min_score_id],
            score_container[min_score_id]
            ]

    def set_error_function(self, error_function: pints.ErrorMeasure) -> None:
        """Sets the objective function which is minimised to find the optimal
        parameter set. For multiple problems, all error functions are updated
        to the selected function.

        Arguments:
            error_function {pints.ErrorMeasure} -- Valid error functions are
            [MeanSquaredError, RootMeanSquaredError, SumOfSquaresError] in
            pints.
        """
        # List of valid error functions
        valid_err_func = [pints.MeanSquaredError,
                          pints.RootMeanSquaredError,
                          pints.SumOfSquaresError
                          ]

        # check of validity of selected error function
        if error_function not in valid_err_func:
            raise ValueError('Objective function is not supported.')

        # update error function
        for problem_id, problem in enumerate(self.problem_container):
            self.error_function_container[problem_id] = error_function(problem)

    def set_optimiser(self, optimiser: pints.Optimiser) -> None:
        """Sets the optimiser used for the inverse problem.

        Arguments:
            optimiser {pints.Optimiser} -- Valid optimisers are [CMAES,
            NelderMead, PSO, SNES, XNES] in pints.
        """
        valid_optimisers = [pints.CMAES,
                            pints.NelderMead,
                            pints.PSO,
                            pints.SNES,
                            pints.XNES
                            ]

        if optimiser not in valid_optimisers:
            raise ValueError('Method is not supported.')

        self.optimiser = optimiser

    def set_parameter_boundaries(self, boundaries: List):
        """Sets the parameter boundaries for inference.

        Arguments:
            boundaries {List} -- List of two lists. [min values, max values]
        """
        if boundaries is None:
            self.parameter_boundaries = None
        else:
            min_values, max_values = boundaries[0], boundaries[1]
            self.parameter_boundaries = pints.RectangularBoundaries(min_values,
                                                                    max_values
                                                                    )


class MultiOutputInverseProblem(object):
    """The multi output inverse problem is based on pints.MultiOutputProblem
    https://pints.readthedocs.io/. Default error function is pints.
    SumOfSquaresError and default optimiser is pints.CMAES.
    """
    def __init__(self,
                 models: List[m.MultiOutputModel],
                 times: List[np.ndarray],
                 values: List[np.ndarray]
                 ):
        """Initialises a multi output inverse problem with default error
        function pints.SumOfSquaresError and default optimiser pints.CMAES.
        Standard deviation in initial starting point of optimisation as well
        as restricted domain of support for inferred parameters is disabled by
        default.

        Arguments:
            models {List[m.MultiOutputModel]} -- Models, which parameters are
            to be inferred.
            times {List[np.ndarray]} -- Times of data points for the different
            models.
            values {List[np.ndarray]} -- State values of data points for the
            different models.

        Return:
            None
        """
        # initialise problem container
        self.problem_container = []
        for model_id, model in enumerate(models):
            self.problem_container.append(
                pints.MultiOutputProblem(model,
                                         times[model_id],
                                         values[model_id]
                                         )
                )

        # initialise error function container
        self.error_function_container = []
        for problem in self.problem_container:
            self.error_function_container.append(
                pints.SumOfSquaresError(problem)
                )

        # initialise optimiser
        self.optimiser = pints.CMAES

        # initialise fluctiations around starting point of optimisation
        self.initial_parameter_uncertainty = None

        # initialise parameter constraints
        self.parameter_boundaries = None

        # initialise outputs
        self.estimated_parameters = None
        self.objective_score = None

    def find_optimal_parameter(self,
                               initial_parameter: np.ndarray,
                               number_of_iterations: int = 1
                               ) -> None:
        """Find point in parameter space that optimises the objective function,
        i.e. find the set of parameters that minimises the distance of the
        model to the data with respect to the objective function.

        Arguments:
            initial_parameter {np.ndarray} -- Starting point in parameter
            space of the optimisation algorithm.

        Return:
            None
        """
        # create sum of errors measure
        error_measure = pints.SumOfErrors(self.error_function_container)

        # initialise optimisation
        optimisation = pints.OptimisationController(
            function=error_measure,
            x0=initial_parameter,
            sigma0=self.initial_parameter_uncertainty,
            boundaries=self.parameter_boundaries,
            method=self.optimiser
            )

        # run optimisation 'number_of_iterations' times
        estimate_container = []
        score_container = []
        for _ in range(number_of_iterations):
            estimates, score = optimisation.run()
            estimate_container.append(estimates)
            score_container.append(score)

        # return parameters with minimal score
        min_score_id = np.argmin(score_container)
        self.estimated_parameters, self.objective_score = [
            estimate_container[min_score_id],
            score_container[min_score_id]
            ]

    def set_error_function(self, error_function: pints.ErrorMeasure) -> None:
        """Sets the objective function which is minimised to find the optimal
        parameter set. For multiple problems, all error functions are updated
        to the selected function.

        Arguments:
            error_function {pints.ErrorMeasure} -- Valid error functions are
            [MeanSquaredError, RootMeanSquaredError, SumOfSquaresError] in
            pints.
        """
        # List of valid error functions
        valid_err_func = [pints.MeanSquaredError, pints.SumOfSquaresError]

        # check of validity of selected error function
        if error_function not in valid_err_func:
            raise ValueError('Objective function is not supported.')

        # update error function
        for problem_id, problem in enumerate(self.problem_container):
            self.error_function_container[problem_id] = error_function(problem)

    def set_optimiser(self, optimiser: pints.Optimiser) -> None:
        """Sets the optimiser to find the "global" minimum of the objective function.

        Arguments:
            optimiser {pints.Optimiser} -- Valid optimisers are [CMAES,
            NelderMead, PSO, SNES, XNES] in pints.
        """
        valid_optimisers = [pints.CMAES,
                            pints.NelderMead,
                            pints.PSO,
                            pints.SNES,
                            pints.XNES
                            ]

        if optimiser not in valid_optimisers:
            raise ValueError('Method is not supported.')

        self.optimiser = optimiser

    def set_parameter_boundaries(self, boundaries: List):
        """Sets the parameter boundaries for inference.

        Arguments:
            boundaries {List} -- List of two lists. [min values, max values]
        """
        min_values, max_values = boundaries[0], boundaries[1]
        self.parameter_boundaries = pints.RectangularBoundaries(min_values, max_values)
