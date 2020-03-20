import unittest

import myokit
import pints
import pytest
import numpy as np

from IPA.model import model as m
from IPA.inference import inference as inf


class TestSingleOutputProblem(unittest.TestCase):
    """Testing the methods of the SingleOutputinverseProblem class.
    """
    # Test Case I: Linear Growth Model (no dosing)
    # instantiate model
    file_linear_model = 'IPA/modelRepository/linear_growth_model.mmt'
    linear_model = m.SingleOutputModel(file_linear_model)
    linear_model_true_params = [0, 1]  # [init drug, lambda]

    # generate data (without noise)
    times = np.linspace(0.0, 24.0, 100)
    linear_model_data = linear_model.simulate(linear_model_true_params,
                                              times
                                              )

    # Test Case II: One Compartment Model (with dosing)
    # instantiate model
    file_one_comp_model = 'IPA/modelRepository/1_bolus_linear.mmt'
    one_comp_model = m.SingleOutputModel(file_one_comp_model)
    one_comp_model_true_params = [0, 1, 4]  # [init drug, CL, V]

    # create protocol object
    protocol = myokit.Protocol()

    # schedule dosing event
    protocol.schedule(level=5, start=4.0, duration=4)
    one_comp_model.simulation.set_protocol(protocol=protocol)

    # generate data
    times = np.linspace(0.0, 24.0, 100)
    one_comp_model_data = one_comp_model.simulate(
        one_comp_model_true_params,
        times
        )

    def test_find_optimal_parameter(self):
        """Test whether the find_optimal_parameter method works as expected.
        """
        # Test Case I: Linear Growth Model
        # instantiate inverse problem
        problem = inf.SingleOutputInverseProblem(
            models=[self.linear_model],
            times=[self.times],
            values=[self.linear_model_data]
            )

        # start somewhere in parameter space (close to the solution for ease)
        initial_parameters = np.array([0.1, 1.1])

        # solve inverse problem
        problem.optimise(initial_parameter=initial_parameters,
                         number_of_iterations=1)
        estimated_paramters = problem.estimated_parameters

        # assert aggreement of estimates with true paramters
        for param_id, true_value in enumerate(self.linear_model_true_params):
            estimated_value = estimated_paramters[param_id]
            assert true_value == pytest.approx(estimated_value, rel=0.05)

    def test_set_error_function(self):
        """Test whether the set_error_function method works as expected.
        """
        # Test Case I: Linear Growth Model
        problem = inf.SingleOutputInverseProblem(
            models=[self.linear_model],
            times=[self.times],
            values=[self.linear_model_data]
            )

        # iterate through valid error measures
        valid_err_func = [pints.MeanSquaredError,
                          pints.RootMeanSquaredError,
                          pints.SumOfSquaresError
                          ]
        for err_func in valid_err_func:
            problem.set_error_function(error_function=err_func)

            # assert that error measure is set as expected
            assert isinstance(problem.error_function_container[0],
                              type(err_func(
                                  problem.problem_container[0]
                                  ))
                              )

    def test_set_optimiser(self):
        """Test whether the set_optimiser method works as expected. The
        estimated values are not of interest but rather whether the
        optimisers are properly embedded.
        """
        # Test Case I: Linear Growth Model
        problem = inf.SingleOutputInverseProblem(
            models=[self.linear_model],
            times=[self.times],
            values=[self.linear_model_data]
            )

        # iterate through valid optimisers
        valid_optimisers = [pints.CMAES,
                            pints.NelderMead,
                            pints.PSO,
                            pints.SNES,
                            pints.XNES
                            ]
        for opt in valid_optimisers:
            problem.set_optimiser(optimiser=opt)

            assert opt == problem.optimiser


# class TestMultiOutputProblem(unittest.TestCase):
#     """Testing the methods of MultiOutputInverseProblem class.
#     """
#     ## Test case: Linear Two Compartment Model with extravascular Dosing
#     # generating data
#     file_name = 'PKPD/modelRepository/2_bolus_linear.mmt'
#     two_comp_model = m.MultiOutputModel(file_name)

#     # create protocol object
#     protocol = myokit.Protocol()

#     # schedule dosing event
#     protocol.schedule(level=1.0, start=4.0, duration=1.0)
#     protocol.schedule(level=1.0, start=10.0, duration=1.0)

#     # set dose schedule
#     two_comp_model.simulation.set_protocol(protocol)

#     # set dimensionality of data
#     two_comp_model.set_output_dimension(2)

#     # List of parameters: ['central_compartment.drug',
#       'dose_compartment.drug', 'peripheral_compartment.drug',
#       'central_compartment.CL',
#     # 'central_compartment.Kcp', 'central_compartment.V',
#        'dose_compartment.Ka', 'peripheral_compartment.Kpc',
#        'peripheral_compartment.V']
#     true_parameters = [1, 1, 1, 3, 5, 2, 2]

#     times = np.linspace(0.0, 24.0, 100)
#     model_result = two_comp_model.simulate(true_parameters, times)

#     # noise free data to check that inf works
#     data_two_comp_model = model_result


#     def test_find_optimal_parameter(self):
#         """Test whether the find_optimal_parameter method works as expected.
#         """
#         problem = inf.MultiOutputInverseProblem(models=[self.two_comp_model],
#                                                       times=[self.times],
#                                                       values=[self.data_two_comp_model]
#                                                       )

#         # start somewhere in parameter space (close to the solution for ease)
#         initial_parameters = np.array([1, 1, 1, 3, 5, 2, 2])

#         # solve inverse problem
#         problem.find_optimal_parameter(initial_parameter=initial_parameters)
#         estimated_parameters = problem.estimated_parameters

#         # assert aggreement of estimates with true paramters
#         for parameter_id, true_value in enumerate(self.true_parameters):
#             estimated_value = estimated_parameters[parameter_id]

#             assert true_value == pytest.approx(estimated_value, rel=0.5)


#     def test_set_error_function(self):
#         """Test whether the set_error_function method works as expected.
#         """
#         problem = inf.MultiOutputInverseProblem(models=[self.two_comp_model],
#                                                        times=[self.times],
#                                                        values=[self.data_two_comp_model]
#         )

#         # iterate through valid error measures
#         valid_err_func = [pints.MeanSquaredError, pints.SumOfSquaresError]
#         for err_func in valid_err_func:
#             problem.set_error_function(error_function=err_func)

#             # assert that error measure is set as expected
#             assert type(err_func(problem.problem_container[0])) == type(
#                   problem.error_function_container[0]
#                   )


#     def test_set_optimiser(self):
#         """Test whether the set_optimiser method works as expected. The
#               estimated values
#         are not of interest but rather whether the optimiser are
#           properly embedded.
#         """
#         problem = inf.MultiOutputInverseProblem(models=[self.two_comp_model],
#                                                        times=[self.times],
#                                                        values=[self.data_two_comp_model]
#                                                        )

#         # iterate through valid optimisers
#         valid_optimisers = [pints.CMAES, pints.NelderMead, pints.PSO,
#                               pints.SNES, pints.XNES]
#         for opt in valid_optimisers:
#             problem.set_optimiser(optimiser=opt)

#             assert opt == problem.optimiser
