import unittest

# import myokit
# import numpy as np

from IPA.model import model as m


class TestSingleOutputModel(unittest.TestCase):
    """Tests the functionality of all methods of the SingleOutputModel class.
    """
    # Test Case I: Linear Growth Model
    file_linear_growth_model = 'IPA/modelRepository/linear_growth_model.mmt'
    linear_model = m.SingleOutputModel(file_linear_growth_model)

    # Test Case II: One Copmartment Model
    file_one_comp_model = 'IPA/modelRepository/1_bolus_linear.mmt'
    one_comp_model = m.SingleOutputModel(file_one_comp_model)

    def test_init(self):
        """Tests whether the Model class initialises as expected.
        """
        # Test Case I: Linear Growth Model (only one possible output)
        # expected:
        state_names = ['central_compartment.drug']
        output_name = 'central_compartment.drug'
        parameter_names = ['central_compartment.lambda']
        number_fit_params = 2

        # assert initilised values coincide
        assert state_names == self.linear_model.state_names
        assert output_name == self.linear_model.output_name
        for param_id, param in enumerate(parameter_names):
            assert self.linear_model.parameter_names[param_id] == param
        assert number_fit_params == self.linear_model.number_parameters_to_fit

        # Test Case II: One Compartment Model (multiple possible outputs)
        # expected:
        state_names = ['central_compartment.drug']
        output_name = 'central_compartment.drug_concentration'
        parameter_names = ['central_compartment.CL', 'central_compartment.V']
        num_fit_params = 3

        # assert initilised values coincide
        assert state_names == self.one_comp_model.state_names
        assert output_name == self.one_comp_model.output_name
        for param_id, param in enumerate(parameter_names):
            assert self.one_comp_model.parameter_names[param_id] == param
        assert num_fit_params == self.one_comp_model.number_parameters_to_fit

    def test_n_parameters(self):
        """Tests whether the n_parameter method returns the correct number
        of fit parameters.
        """
        # Test Case I: Linear Growth Model
        # expected
        n_parameters = 2

        # assert correct number of parameters is returned.
        assert n_parameters == self.linear_model.n_parameters()

        # Test Case II: One Compartment Model
        # expected
        n_parameters = 3

        # assert correct number of parameters is returned.
        assert n_parameters == self.one_comp_model.n_parameters()


#     def test_n_outputs(self):
#         """Tests whether the n_outputs method returns the correct number of outputs.
#         """
#         # Test case I: 1-compartment model
#         ## expected
#         n_outputs = 1

#         ## assert correct number of outputs.
#         assert n_outputs == self.one_comp_model.n_outputs()


#     def test_simulate(self):
#         """Tests whether the simulate method works as expected. Tests implicitly also whether
#         the _set_parameters method works properly.
#         """
#         # Test case I: 1-compartment model
#         parameters = [0, 2, 4] # different from initialsed parameters
#         times = np.arange(25)

#         ## expected
#         model, protocol, _ = myokit.load(self.file_name)
#         model.set_state([parameters[0]])
#         model.set_value('central_compartment.CL', parameters[1])
#         model.set_value('central_compartment.V', parameters[2])
#         simulation = myokit.Simulation(model, protocol)
#         myokit_result = simulation.run(duration=times[-1]+1, log=['central_compartment.drug_concentration'], log_times = times)
#         expected_result = myokit_result.get('central_compartment.drug_concentration')

#         ## assert that Model.simulate returns the same result.
#         model_result = self.one_comp_model.simulate(parameters, times)

#         assert np.array_equal(expected_result, model_result)


# class TestMultiOutputModel(unittest.TestCase):
#     """Tests the functionality of all methods of the MultiOutputModel class.
#     """
#     # Test case I: 1-compartment model
#     file_name = 'PKPD/modelRepository/2_bolus_linear.mmt'
#     two_comp_model = m.MultiOutputModel(file_name)

#     # set dimensionality
#     output_dimension = 2
#     two_comp_model.set_output_dimension(output_dimension)


#     def test_init(self):
#         """Tests whether the Model class initialises as expected.
#         """
#         # Test case I: 1-compartment model
#         ## expected:
#         state_names = ['central_compartment.drug', 'peripheral_compartment.drug']
#         parameter_names = ['central_compartment.CL',
#                            'central_compartment.Kcp',
#                            'central_compartment.V',
#                            'peripheral_compartment.Kpc',
#                            'peripheral_compartment.V'
#                            ]

#         ## assert initilised values coincide
#         assert state_names == self.two_comp_model.state_names
#         assert parameter_names == self.two_comp_model.parameter_names


#     def test_n_parameters(self):
#         """Tests whether the n_parameter method returns the correct number of fit parameters.
#         """
#         # Test case I: 1-compartment model
#         ## expected
#         n_parameters = 7

#         ## assert correct number of parameters is returned.
#         assert n_parameters == self.two_comp_model.n_parameters()


#     def test_n_outputs(self):
#         """Tests whether the n_outputs method returns the correct number of outputs.
#         """
#         # Test case I: 1-compartment model
#         ## expected
#         n_outputs = 2

#         ## assert correct number of outputs.
#         assert n_outputs == self.two_comp_model.n_outputs()


#     def test_simulate(self):
#         """Tests whether the simulate method works as expected. Tests implicitly also whether
#         the _set_parameters method works properly.
#         """
#         output_names = ['central_compartment.drug_concentration',
#                         'peripheral_compartment.drug_concentration']
#         state_dimension = 2
#         parameters = [0, 0, 1, 3, 5, 2, 2] # states + parameters
#         parameter_names = ['central_compartment.CL',
#                            'central_compartment.Kcp',
#                            'central_compartment.V',
#                            'peripheral_compartment.Kpc',
#                            'peripheral_compartment.V'
#                            ]
#         times = np.arange(100)

#         ## expected
#         # initialise model
#         model, protocol, _ = myokit.load(self.file_name)

#         # set initial conditions and parameter values
#         model.set_state(parameters[:state_dimension])
#         for parameter_id, name in enumerate(parameter_names):
#             model.set_value(name, parameters[state_dimension + parameter_id])

#         # solve model
#         simulation = myokit.Simulation(model, protocol)
#         myokit_result = simulation.run(duration=times[-1]+1, log=output_names, log_times = times)

#         # get expected result
#         expected_result = []
#         for name in output_names:
#             expected_result.append(myokit_result.get(name))
#         np_expected_result = np.array(expected_result)

#         ## assert that Model.simulate returns the same result.
#         model_result = self.two_comp_model.simulate(parameters, times).transpose()

#         assert np.allclose(np_expected_result, model_result)