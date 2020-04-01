import unittest

import myokit
import numpy as np

from IPA.model import model as m


class TestSingleOutputModel(unittest.TestCase):
    """Tests the functionality of all methods of the SingleOutputModel class.
    """
    # Test Case I: Linear Growth Model
    # instantiate model
    file_linear_growth_model = 'IPA/modelRepository/LGM_plus_protocol.mmt'
    linear_model = m.SingleOutputModel(file_linear_growth_model)

    # Test Case II: One Copmartment Model
    # instantiate model
    file_one_comp_model = 'IPA/modelRepository/1_bolus_linear.mmt'
    one_comp_model = m.SingleOutputModel(file_one_comp_model)

    def test_init(self):
        """Tests whether the Model class initialises as expected.
        """
        # Test Case I: Linear Growth Model + protocol
        # expected:
        state_names = np.array(['central_compartment.drug'])
        output_name = 'central_compartment.drug'
        parameter_names = np.array(['central_compartment.lambda'])
        number_fit_params = 2

        # get models state and parameter names
        model_state_names = self.linear_model.get_state_names()
        model_param_names = self.linear_model.get_param_names()

        # assert initilised values coincide
        assert state_names == model_state_names
        assert output_name == self.linear_model.output_name
        for param_id, param in enumerate(parameter_names):
            assert model_param_names[param_id] == param
        assert number_fit_params == self.linear_model.number_fit_params

        # Test Case II: One Compartment Model (multiple possible outputs)
        # expected:
        state_names = np.array(['central_compartment.drug'])
        output_name = 'central_compartment.drug_concentration'
        parameter_names = np.array(['central_compartment.CL',
                                    'central_compartment.V'
                                    ]
                                   )
        num_fit_params = 3

        # get models state and parameter names
        model_state_names = self.one_comp_model.get_state_names()
        model_param_names = self.one_comp_model.get_param_names()

        # assert initilised values coincide
        assert state_names == model_state_names
        assert output_name == self.one_comp_model.output_name
        for param_id, param in enumerate(parameter_names):
            assert model_param_names[param_id] == param
        assert num_fit_params == self.one_comp_model.number_fit_params

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

    def test_n_outputs(self):
        """Tests whether the n_outputs method returns the correct number of
        outputs. One test case is sufficient as returned value of n_outputs()
        is hard coded.
        """
        # Test Case I: Linear Growth Model
        # expected
        n_outputs = 1

        # assert correct number of outputs.
        assert n_outputs == self.linear_model.n_outputs()

    def test_set_output(self):
        """Tests whether the set_output method sets the output as expected.
        """
        # Test Case I: Linear Growth Model
        # expected
        output_name = 'central_compartment.drug'

        # set output
        self.linear_model.set_output(output_name)

        # assert output is set correctly.
        assert output_name == self.linear_model.output_name

    def test_simulate(self):
        """Tests whether the simulate method works as expected. Tests
        implicitly also whether the _set_parameters method works properly.
        """
        # Test Case I: Linear Growth Model
        # define parameters and times
        parameters = [1, 2]
        times = np.arange(25)

        # expected
        # init model in myokit
        model, protocol, _ = myokit.load(self.file_linear_growth_model)

        # set inital state of model
        model.set_state([parameters[0]])

        # set linear growth constant
        model.set_value('central_compartment.lambda', parameters[1])

        # instantiate and run simulation
        simulation = myokit.Simulation(model, protocol)
        myokit_result = simulation.run(duration=times[-1]+1,
                                       log=['central_compartment.drug'],
                                       log_times=times
                                       )
        expected_result = myokit_result.get('central_compartment.drug')

        # assert that Model.simulate returns the same result.
        model_result = self.linear_model.simulate(parameters, times)

        assert np.array_equal(expected_result, model_result)

        # Test case II: One Compartment Model (checks whether access of
        # correct output works)
        # define parameters and times
        parameters = [0, 2, 4]
        times = np.arange(25)

        # expected
        # init model in myokit
        model, protocol, _ = myokit.load(self.file_one_comp_model)

        # set inital state of model
        model.set_state([parameters[0]])

        # set clearance and volume
        model.set_value('central_compartment.CL', parameters[1])
        model.set_value('central_compartment.V', parameters[2])

        # instantiate and run simulation
        simulation = myokit.Simulation(model, protocol)
        state_name = 'central_compartment.drug_concentration'
        myokit_result = simulation.run(duration=times[-1]+1,
                                       log=[state_name],
                                       log_times=times
                                       )
        expected_result = myokit_result.get(state_name)

        # assert that Model.simulate returns the same result.
        model_result = self.one_comp_model.simulate(parameters, times)

        assert np.array_equal(expected_result, model_result)


class TestMultiOutputModel(unittest.TestCase):
    """Tests the functionality of all methods of the MultiOutputModel class.
    """
    # Test Case I: Two Uncoupled Linear Growth Models (ULG models)
    # instantiate model
    file_ULG_model = 'IPA/modelRepository/ULGM_plus_protocol.mmt'
    ULG_model = m.MultiOutputModel(file_ULG_model)

    # set dimensionality
    output_dimension = 2
    ULG_model.set_output_dimension(output_dimension)

    # Test case II: Two Compartment Model
    file_two_comp_model = 'IPA/modelRepository/2_bolus_linear.mmt'
    two_comp_model = m.MultiOutputModel(file_two_comp_model)

    # set dimensionality
    output_dimension = 2
    two_comp_model.set_output_dimension(output_dimension)

    def test_init(self):
        """Tests whether the Model class initialises as expected.
        """
        # Test Case I: Two Uncoupled Linear Growth Models (ULG models)
        # expected
        state_names = ['central_compartment.drug',
                       'peripheral_compartment.drug'
                       ]
        parameter_names = ['central_compartment.lambda',
                           'peripheral_compartment.lambda',
                           ]

        # assert initialised values coincide
        assert state_names == self.ULG_model.state_names
        assert parameter_names == self.ULG_model.parameter_names

        # Test Case II: Two Compartment Model
        # expected:
        state_names = ['central_compartment.drug',
                       'peripheral_compartment.drug'
                       ]
        parameter_names = ['central_compartment.CL',
                           'central_compartment.Kcp',
                           'central_compartment.V',
                           'peripheral_compartment.Kpc',
                           'peripheral_compartment.V'
                           ]

        # assert initialised values coincide
        assert state_names == self.two_comp_model.state_names
        assert parameter_names == self.two_comp_model.parameter_names

    def test_n_parameters(self):
        """Tests whether the n_parameter method returns the correct number
        of fit parameters.
        """
        # Test Case I: Two Uncoupled Linear Growth Models (ULG models) (tests
        #  whether filter for non-bound variables works)
        # expected
        n_parameters = 4

        # assert correct number of parameters is returned.
        assert n_parameters == self.ULG_model.n_parameters()

        # Test Case II: Two Compartment Model (tests whether filter for non-
        # inter variables works)
        # expected
        n_parameters = 7

        # assert correct number of parameters is returned.
        assert n_parameters == self.two_comp_model.n_parameters()

    def test_n_outputs(self):
        """Tests whether the n_outputs method returns the correct number of
        outputs.
        """
        # Test Case I: Two Uncoupled Linear Growth Models (ULG models)
        # expected
        n_outputs = 2

        # assert that the number of outputs coincide
        assert n_outputs == self.ULG_model.n_outputs()

    def test_set_output(self):
        """Tests whether the set_output method sets the output as expected.
        """
        # Test Case I: Two Uncoupled Linear Growth Models (ULG models)
        # expected
        output_names = ['central_compartment.drug',
                        'peripheral_compartment.drug'
                        ]
        output_dimension = 2

        # set output
        self.ULG_model.set_output(output_names)

        # assert output is set correctly.
        assert output_dimension == self.ULG_model.output_dimension
        assert output_names == self.ULG_model.output_names

    def test_simulate(self):
        """Tests whether the simulate method works as expected. Tests
        implicitly also whether the _set_parameters method works properly.
        """
        # Test Case I: Two Uncoupled Linear Growth Models (ULG models)
        # define model
        output_names = ['central_compartment.drug',
                        'peripheral_compartment.drug'
                        ]
        state_dimension = 2
        parameters = [0, 0, 1, 2]  # states + parameters
        parameter_names = ['central_compartment.lambda',
                           'peripheral_compartment.lambda'
                           ]
        times = np.arange(100)

        # expected
        # initialise model
        model, protocol, _ = myokit.load(self.file_ULG_model)

        # set initial conditions and parameter values
        model.set_state(parameters[:state_dimension])
        for parameter_id, name in enumerate(parameter_names):
            model.set_value(name, parameters[state_dimension + parameter_id])

        # solve model
        simulation = myokit.Simulation(model, protocol)
        myokit_result = simulation.run(duration=times[-1]+1,
                                       log=output_names,
                                       log_times=times)

        # get expected result
        expected_result = []
        for name in output_names:
            expected_result.append(myokit_result.get(name))
        np_expected_result = np.array(expected_result)

        # simulate model with Model.simulate
        model_result = self.ULG_model.simulate(parameters, times)

        # make output compatible with myokit result
        model_result = model_result.transpose()

        # assert that simulation results are as expected
        assert np.allclose(np_expected_result, model_result)

        # Test Case II: Two Compartment Model
        # define model
        output_names = ['central_compartment.drug_concentration',
                        'peripheral_compartment.drug_concentration']
        state_dimension = 2
        parameters = [0, 0, 1, 3, 5, 2, 2]  # states + parameters
        parameter_names = ['central_compartment.CL',
                           'central_compartment.Kcp',
                           'central_compartment.V',
                           'peripheral_compartment.Kpc',
                           'peripheral_compartment.V'
                           ]
        times = np.arange(100)

        # expected
        # initialise model
        model, protocol, _ = myokit.load(self.file_two_comp_model)

        # set initial conditions and parameter values
        model.set_state(parameters[:state_dimension])
        for parameter_id, name in enumerate(parameter_names):
            model.set_value(name, parameters[state_dimension + parameter_id])

        # solve model
        simulation = myokit.Simulation(model, protocol)
        myokit_result = simulation.run(duration=times[-1]+1,
                                       log=output_names,
                                       log_times=times)

        # get expected result
        expected_result = []
        for name in output_names:
            expected_result.append(myokit_result.get(name))
        np_expected_result = np.array(expected_result)

        # simulate model with Model.simulate
        model_result = self.two_comp_model.simulate(parameters, times)

        # make output compatible with myokit result
        model_result = model_result.transpose()

        # assert that simulation results are as expected
        assert np.allclose(np_expected_result, model_result)
