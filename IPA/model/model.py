from array import array
from typing import List

import myokit
import numpy as np
import pints


class SingleOutputModel(pints.ForwardModel):
    """Model class inheriting from pints.ForwardModel. To solve the forward
    problem, methods from the myokit package are employed. The sole difference
    to the MultiOutputProblem is that the simulate methodreturns a 1d array
    instead of a 2d array.
    """
    def __init__(self, mmtfile: str) -> None:
        """Initialises the model class.

        Arguments:
            mmtfile {str} -- Path to the mmtfile defining the model and the
            protocol.
        """
        # load model and protocol
        self.model, protocol, _ = myokit.load(mmtfile)

        # get dose events from protocol
        self.mmt_dose_events = self._get_mmt_dose_events(protocol)

        # get state, parameter and output names
        self.state_dim = self.model.count_states()
        self.mdof_names = self._get_model_dof_names()
        self.output_name = self._get_default_output_name()

        # create mask for which mdof can be fitted (default: all)
        self.number_fit_params = len(self.mdof_names)
        self.fit_mask = np.ones(self.number_fit_params, dtype=bool)

        # initialise container for model dof values
        self.mdof_values = np.empty(self.number_fit_params)

        # instantiate the simulation
        self.simulation = myokit.Simulation(self.model, protocol)

    def _get_mmt_dose_events(self, protocol: myokit.Protocol) -> np.ndarray:
        """Get a list of dose events from the protocol provided in the model mmt file.

        Arguments:
            protocol {myokit.Protocol} -- Dosing protocol in mmt file.

        Returns:
            {np.ndarray} -- 2dim array with dose event details.
        """
        # if no protocol is provided in mmt file, set mmt_dose_shedule to None
        if protocol is None:
            return None

        # if protocol is provided, loop through events and safe level, start,
        # duration, period, multiplier
        else:
            # get dose events from protocol
            dose_events = protocol.events()

            # get number of dose events
            number_of_dose_events = len(dose_events)

            # initialise dose container
            dose_event_container = np.empty(shape=(number_of_dose_events, 5))

            # loop thorugh events
            for event_id, event in enumerate(dose_events):
                # safe event in container
                dose_event_container[event_id, :] = [event.level(),
                                                     event.start(),
                                                     event.duration(),
                                                     event.period(),
                                                     event.multiplier()
                                                     ]

            return dose_event_container

    def _get_model_dof_names(self) -> np.ndarray:
        """Returns a numpy array with the names of model degrees of freedom (dof).
        States first and parameters last.

        Returns:
            {np.ndarray} -- Array with model degrees of freedom.
        """
        # get state names
        state_names = [state.qname() for state in self.model.states()]

        # get parameter names
        parameter_names = self._get_parameter_names()

        # create model dof container [states, params]
        mdof_names = np.array(state_names + parameter_names)

        # return mdof container
        return mdof_names

    def _get_default_output_name(self):
        """Returns 'central_compartment.drug_concentration' as output_name by
        default. If variable does not exist in model, first state variable name
        is returned.

        Arguments:
            model {myokit.Model} -- A myokit model.

        Returns:
            str -- Output name of model.
        """
        default_output_name = 'central_compartment.drug_concentration'
        if self.model.has_variable(default_output_name):
            return default_output_name
        else:
            # if default output name does not exist, output first state
            # variable
            first_state_name = self.mdof_names[0]
            return first_state_name

    def _get_parameter_names(self):
        """Gets parameter names of the ODE model, i.e. initial conditions are excluded.

        Arguments:
            model {myokit.Model} -- A myokit model.

        Returns:
            List -- List of parameter names.
        """
        # initialise container for parameter names
        parameter_names = []

        # collect parameter names
        for component in self.model.components(sort=True):
            parameter_names += [var.qname() for var in
                                component.variables(state=False,
                                                    inter=False,
                                                    bound=False,
                                                    sort=True
                                                    )
                                ]

        # return parameter names
        return parameter_names

    def get_state_names(self) -> np.ndarray:
        """Returns state names.

        Returns:
            {np.ndarray} -- Returns array with state names.
        """
        return self.mdof_names[:self.state_dim]

    def get_param_names(self) -> np.ndarray:
        """Returns parameter names.

        Returns:
            {np.ndarray} -- Returns array with parameter names.
        """
        return self.mdof_names[self.state_dim:]

    def get_fixed_mdof(self) -> List[np.ndarray]:
        # get fixed model dof names
        names = self.mdof_names[~self.fit_mask]

        # get values of fixed mdof
        values = self.mdof_values[~self.fit_mask]

        return [names, values]

    def n_parameters(self) -> int:
        """Returns the number of parameters of the model, i.e. initial
        conditions and model parameters.

        Returns:
            int -- Number of parameters.
        """
        return self.number_fit_params

    def n_outputs(self) -> None:
        """Returns the dimension of the state variable.

        Returns:
            int -- Dimensionality of the output.
        """
        return 1

    def set_output(self, output_name: str):
        """Sets the output of the model.

        Arguments:
            output_name {str} -- Name of state that should be returned from
            the simulation.
        """
        self.output_name = output_name

    def fix_model_dof(self, names: List, values: List) -> None:
        """Fixes given model degrees of freedom, such that will no longer be
        inferred, but held constant.

        Arguments:
            names {List} -- List of model dof names.
            values {List} -- List of model dof values.
        """
        # if names or values is None, enable fitting for all mdof
        if names is None or values is None:
            self.number_fit_params = len(self.mdof_names)
            self.fit_mask = np.ones(self.number_fit_params, dtype=bool)

        else:
            # loop through parameter names and update value and mask
            for name_id, name in enumerate(names):
                # find location in mdof container
                mask = self.mdof_names == name

                # update mask
                self.fit_mask[mask] = False

                # update value
                self.mdof_values[mask] = values[name_id]

                # if name does not exist, return warning
                if not np.any(mask):
                    print('WARNING: the parameter %s does not seem to' % name
                          + ' exist in the model')

            # update the new number of fit parameters
            self.number_fit_params = int(np.sum(self.fit_mask))

    def simulate(self, parameters: np.ndarray, times: np.ndarray) -> array:
        """Solves the forward problem and returns the state values evaluated
        at the times provided.

        Arguments:
            parameters {np.ndarray} -- Parameters of the model. By convention
                                       [initial conditions, model parameters].
            times {np.ndarray} -- Times at which states will be evaluated.

        Returns:
            [array] -- State values evaluated at provided times.
        """
        self.simulation.reset()
        self._set_parameters(parameters)

        # duration is the last time point plus an increment to iclude the last
        # time step.
        result = self.simulation.run(duration=times[-1]+1,
                                     log=[self.output_name],
                                     log_times=times
                                     )

        return result[self.output_name]

    def _set_parameters(self, parameters: np.ndarray) -> None:
        """Internal helper method to set the parameters of the forward model.

        Arguments:
            parameters {np.ndarray} -- Parameters of the model. By convention
                                       [initial condition, model parameters].
        """
        # update model dof
        self.mdof_values[self.fit_mask] = parameters

        # get param names
        param_names = self.mdof_names[self.state_dim:]

        self.simulation.set_state(self.mdof_values[:self.state_dim])
        for param_id, value in enumerate(self.mdof_values[self.state_dim:]):
            self.simulation.set_constant(param_names[param_id], value)


class MultiOutputModel(pints.ForwardModel):
    """Model class inheriting from pints.ForwardModel. To solve the forward
    problem, methods from the myokit package are employed. The sole difference
    to the SingleOutputProblem is that the simulate method returns a 2d array
    instead of a 1d array.
    """
    def __init__(self, mmtfile: str) -> None:
        """Initialises the model class.

        Arguments:
            mmtfile {str} -- Path to the mmtfile defining the model and the
            protocol.
        """
        # load model and protocol
        model, protocol, _ = myokit.load(mmtfile)

        # get dose events from protocol
        self.mmt_dose_events = self._get_mmt_dose_events(protocol)

        # get state, parameter and output names
        self.state_names = [state.qname() for state in model.states()]
        self.state_dim = model.count_states()
        self.output_names = []
        self.output_dimension = None
        self.parameter_names = self._get_parameter_names(model)
        self.number_parameters_to_fit = model.count_variables(inter=False,
                                                              bound=False
                                                              )

        # instantiate the simulation
        self.simulation = myokit.Simulation(model, protocol)
        self.model = model

    def _get_mmt_dose_events(self, protocol: myokit.Protocol) -> np.ndarray:
        """Get a list of dose events from the protocol provided in the model mmt file.

        Arguments:
            protocol {myokit.Protocol} -- Dosing protocol in mmt file.

        Returns:
            {np.ndarray} -- 2dim array with dose event details.
        """
        # if no protocol is provided in mmt file, set mmt_dose_shedule to None
        if protocol is None:
            return None

        # if protocol is provided, loop through events and safe level, start,
        # duration, period, multiplier
        else:
            # get dose events from protocol
            dose_events = protocol.events()

            # get number of dose events
            number_of_dose_events = len(dose_events)

            # initialise dose container
            dose_event_container = np.empty(shape=(number_of_dose_events, 5))

            # loop thorugh events
            for event_id, event in enumerate(dose_events):
                # safe event in container
                dose_event_container[event_id, :] = [event.level(),
                                                     event.start(),
                                                     event.duration(),
                                                     event.period(),
                                                     event.multiplier()
                                                     ]

            return dose_event_container

    def _get_parameter_names(self, model: myokit.Model):
        """Gets parameter names of the ODE model, i.e. initial conditions are excluded.

        Arguments:
            model {myokit.Model} -- A myokit model.

        Returns:
            List -- List of parameter names.
        """
        parameter_names = []
        for component in model.components(sort=True):
            parameter_names += [var.qname() for var in
                                component.variables(state=False,
                                                    inter=False,
                                                    bound=False,
                                                    sort=True)
                                ]

        return parameter_names

    def n_parameters(self) -> int:
        """Returns the number of parameters of the model, i.e. initial
        conditions and model parameters.

        Returns:
            int -- Number of parameters.
        """
        return self.number_parameters_to_fit

    def n_outputs(self) -> None:
        """Returns the dimension of the state variable.

        Returns:
            int -- Dimensionality of the output.
        """
        return self.output_dimension

    def simulate(self, parameters: np.ndarray, times: np.ndarray):
        """Solves the forward problem and returns the state values evaluated
        at the times provided.

        Arguments:
            parameters {np.ndarray} -- Parameters of the model. By convention
                                       [initial conditions, model parameters].
            times {np.ndarray} -- Times at which states will be evaluated.

        Returns:
            [np.ndarray] -- State values evaluated at provided times.
        """
        self.simulation.reset()
        self._set_parameters(parameters)

        # duration is the last time point plus an increment to iclude the last
        # time step.
        output = self.simulation.run(duration=times[-1]+1,
                                     log=self.output_names,
                                     log_times=times
                                     )

        result = []
        for name in self.output_names:
            result.append(output[name])

        return np.array(result).transpose()

    def _set_parameters(self, parameters: np.ndarray) -> None:
        """Internal helper method to set the parameters of the forward model.

        Arguments:
            parameters {np.ndarray} -- Parameters of the model. By convention
                                       [initial condition, model parameters].
        """
        self.simulation.set_state(parameters[:self.state_dim])
        for param_id, value in enumerate(parameters[self.state_dim:]):
            self.simulation.set_constant(self.parameter_names[param_id], value)

    def set_output_dimension(self, data_dimension: int):
        """Set output dimension to data dimension, so optimisation/inference
        can be performed. Output state will be set to default output names.

        Arguments:
            data_dimension {int} -- Dimensionality of input data.
        """
        # set output dimension
        self.output_dimension = data_dimension

        # if dimension of outputs does not match, fill with default outputs
        if len(self.output_names) != self.output_dimension:
            self._set_default_output_names()

    def _set_default_output_names(self):
        """Returns 'central_compartment.drug_concentration' as output_name by
        default. If variable does not exist in model, first state variable name
        is returned.

        Arguments:
            model {myokit.Model} -- A myokit model.

        Returns:
            str -- Output names of model.
        """
        default_output_names = []
        default_output = 'drug_concentration'
        model = self.simulation._model

        # iterate through components and fill with default variables
        model_components = model.components(sort=True)
        for component in model_components:
            if component.has_variable(default_output):
                variable_name = component.name() + '.' + default_output
                default_output_names.append(variable_name)

        # check dimensional compatibility
        if len(default_output_names) >= self.output_dimension:
            self.output_names = default_output_names[:self.output_dimension]
        elif self.state_dim >= self.output_dimension:
            self.output_names = self.state_names[:self.output_dimension]

    def set_output(self, output_names: List):
        """Set output of the model.

        Arguments:
            output_names {List} -- List of (state) variable names in the model.
        """
        self.output_dimension = len(output_names)
        self.output_names = output_names
