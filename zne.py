#
# (C) Copyright Kathrin F. König, Finn Reinecke 2022.
#

from qiskit import QuantumCircuit, transpile
from qiskit.utils.mitigation import (tensored_meas_cal,
                                                 TensoredMeasFitter)
import mthree

import mapomatic as mm
import json
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit, Polynomial
from scipy.optimize import curve_fit
from time import strftime, localtime
from datetime import datetime
from os.path import isfile, join
from pathlib import Path


class ZNE():
    def __init__(self):
        self.date = strftime("%a %d.%m.%y", localtime())
        self.time = strftime("%H:%M:%S", localtime())
        self.backend = None
        self.per_n_val = 1
        self.mitigator = None
        self.mitigator_available = False
        self.x_0 = None
        self.memory = None
        self.mitigated_counts_per_n = None
        self.mitigated_counts = None
        self.mit_expect_vals = None

    def transpile(self, qc, backend, initial_layout=None,
                  optimization_level=3):
        """
        Transpiles the circuit qc for the given backend. If no
        initial_layout is given the module mapomatic is used to find
        the best layout based on the noise model of the backend.
        Arguments:
            qc (QuantumCircuit): circuit that will be transpiled
            backend (backend): backend for that qc will be transpiled
            initial_layout (list): initial position of virtual qubits
                                   on physical qubits.
            optimization_level (int): How much optimization to perform
                                      on the circuits.
        Returns:
            qc_t (QuantumCircuit): transpiled circuit with the lowest
                                   number of CNOT gates.
        """
        self.qc = qc
        if initial_layout is None:
            qc_t = self.get_min_gates(self.qc, backend,
                                      optimization_level=optimization_level)
            small_qc = mm.deflate_circuit(qc_t)
            initial_layout = mm.best_overall_layout(small_qc, backend)[0]
            qc_t = transpile(small_qc, backend, initial_layout=initial_layout,
                             optimization_level=optimization_level)
        else:
            qc_t = self.get_min_gates(self.qc, backend,
                                      initial_layout=initial_layout,
                                      optimization_level=optimization_level)
        self.initial_layout = initial_layout
        self.backend = backend
        return qc_t

    def get_min_gates(self, qc, backend, initial_layout=None,
                      optimization_level=3):
        """
        Transpiles qc for the given backend and the initial_layout.
        Transpiles 20 times and takes the circuit with the lowest
        number of CNOT gates.
        Arguments:
            qc (QuantumCircuit): circuit that will be transpiled
            backend (backend): backend for that qc will be transpiled
            initial_layout (list): layout thats best for the used
                                   backend
            optimization_level (int): How much optimization to perform
                                      on the circuits.
        Returns:
            qc_t (QuantumCircuit): transpiled circuit with lowest number
                            of CNOT-gates
        """
        cx = False
        trans_factor = 1
        if "cx" in qc.count_ops():
            cx = True
            trans_factor = 20
        if initial_layout is None:
            qcs = transpile([qc]*trans_factor, backend=backend,
                            optimization_level=optimization_level)
        else:
            qcs = transpile([qc]*trans_factor, backend=backend,
                            optimization_level=optimization_level,
                            initial_layout=initial_layout)
        if cx:
            cx_counts = np.array([circ.count_ops()["cx"] for circ in qcs])
            qc_t = qcs[np.argmin(cx_counts)]
        else:
            qc_t = qcs[0]
        return qc_t

    def fold_circs(self, qc_t, a=1, n_vals=None,  backend=None, **kwargs):
        """
        Folds the circuits / Generates circuits with added CNOTs out of
        qc_t. If n_vals is given, there will be one base circuit for
        each n_val. If no n_vals are given there will automatically be
        chosen n_vals based on the noise model of the backend (has to be
        specified) and the parameter a.
        Arguments:
            qc_t (QuantumCircuit): Transpiled circuit.
            a (float): parameter that defines the highest n_val
            n_vals (list): list of n_values to multiply the cx_i with.
                           Used instead of the method with parameter a.
            backend (backend): The backend to get the noise model from.
                               Needed if a is given.
        Returns:
            qc_list (list of QuantumCircuits): list of folded circuits
        """
        self.cx_gate_count_t = qc_t.count_ops()["cx"]
        if n_vals is None:
            self.x_0 = self.get_cx_error(qc_t, backend)
            n_vals = self.get_n_vals(self.x_0, a=a, **kwargs)
        qc_list, real_n_vals, n_dic =\
            self.get_qc_list(qc_t, n_vals, cx_count=self.cx_gate_count_t,
                             **kwargs)
        self.n_vals_given = n_vals
        self.real_n_vals = real_n_vals
        self.n_dic = n_dic
        if len(qc_list) == 1:
            return qc_list[0]
        return qc_list

    def get_cx_error(self, qc, backend=None):
        """
        Estimates the total error of the CNOT gates of a given circuit.
        The error rates are taken from the backend properties.
        Arguments:
            qc (QuantumCiruit): QuantumCircuit to estimate the CNOT-
                                error from.
            backend (backend): Backend whose noise_model will be used
        Returns:
            x_0 (float): total error of CNOT gates within one execution
        """
        if backend is None:
            backend = self.backend
        # CX erorrates of backend are saved in dict
        cx_errors = {}
        for gate in backend.properties().gates:
            if gate.name.startswith("cx"):
                cx_errors[str(gate.qubits)] = gate.parameters[0].value
        cx_gates = []
        # Searching for CX and save control target pairs
        for gate in qc:
            if gate[0].name == "cx":
                control, target = [gate[1][0].index, gate[1][1].index]
                cx_gates += [[control, target]]
        self.cx_gates = cx_gates
        # Counting same CX-gates in result
        cx_counts = dict((str(i), cx_gates.count(i)) for i in cx_gates)
        x_0 = 0
        for k in cx_counts:
            x_0 += cx_counts[k]*cx_errors[k]
        return np.round(x_0, 5)

    def get_n_vals(self, x_0, a=0.5, n_0=0, data_points=5, **kwargs):
        """
        Calculates the n-values for n € [0, a/(2 * x_0)] or [x_0,x_0+a].
        The values will be equally spaced.
        Args:
            x_0 (float): total CNOT-Errorrate
            a (int): defines highest n_value
            n_0 (float): first n-value
            data_points (int): number of n-values
        Returns:
            n_vals (list of length 'data_points'): n-values
        """
        if x_0 == 0:
            x_0 = 0.5  # very high default value if something fails
        self.a = a
        nmax = a / (2 * x_0)
        n_vals = np.linspace(n_0, nmax, data_points)
        return np.round(n_vals, 5)

    def get_qc_list(self, qc, n_vals, cx_count=None, per_n_val=None, **kwargs):
        """
        Creates the list of QuantumCiruits based on given n_vals.
        Arguments:
            qc (QuantumCircuit): transpiled circuit
            n_vals (list): list of n_values to multiply the cx_i with
            cx_count (int): number of CNOTs in the circuit. Not needed
                            only as microoptimization.
            per_n_val (int): number of circuits for one n-value. Used
                            to average about the randomness implied by
                            the random sampling.
        Returns:
            qc_list (list): list of circuits to be executed
        """
        if per_n_val is None:
            per_n_val = self.per_n_val
        else:
            self.per_n_val = per_n_val
        if cx_count is None:
            cx_count = qc.count_ops()["cx"]
        qc_list = []
        real_n_vals = []
        n_dic = {}
        for n in n_vals:
            real_decimals = np.round((n-int(n))*cx_count)/cx_count
            real_n = np.round(real_decimals + int(n), 5)
            real_n_vals += [real_n]
            n_dic[real_n] = {"n_1": int(n), "n_2": []}
            for _ in range(per_n_val):
                new_qc, n_1, n_2 = self.fold_qc(qc, n, cx_count)
                # n_2_real = n_2 - 0.5 !
                n_dic[real_n]["n_2"] += [n_2]  # no real n_2 representation!
                qc_list += [new_qc]

        return qc_list, real_n_vals, n_dic

    def fold_qc(self, qc, n, cx_count=None):
        """
        Folds the a circuit corresponding to the given n_value
        (extending the transpiled circuit by the extra CNOT gates).
        Arguments:
            qc_t (QuantumCircuit): Circuit to be folded.
            n (float): n value to multiply the CNOT of the value with.
            cx_count (int): Number of CNOTs in the circuit. Not needed
                            only as microoptimization.
        Returns:
            qc_list (list): list of folded circuits.
        """
        # n_2 is not a real n_val representation!
        # n_2_real = n_2 - 0.5 !!!!!!!
        if cx_count is None:
            cx_count = qc.count_ops()["cx"]
        n_1 = int(n)
        decimals = n-n_1
        extr_cx = np.round(cx_count*decimals)
        if 10**(-6) > decimals:
            n_2 = [0] * cx_count
        else:
            n_2 = self.random_sample(extr_cx, cx_count).tolist()

        qregs = qc.qregs
        cregs = qc.cregs
        new_qc = QuantumCircuit(*qregs, *cregs)
        for gate in qc:
            name = gate[0].name
            if name == "cx":
                for i in range(2*n_1+1):
                    if n != -1/2:
                        new_qc.data.append(gate)
                        new_qc.barrier()
                for i in range(int(2*n_2[-cx_count])):
                    new_qc.data.append(gate)
                    new_qc.barrier()
                cx_count -= 1
            else:
                new_qc.data.append(gate)
        return new_qc, n_1, n_2

    def random_sample(self, k, output_size=10):
        """
        Generates a random array of size output_size with k entries
        marked with 1 and rest is 0. (Corresponding CNOT gates to the
        marked entries will be folded one additional time)
        Arguments:
            k (int): Number of entries to mark.
            output_size (int): Size of returned array.
        Returns:
            result (array): Array with k marked (1) entries of size
                            output_size.
        """
        numbers = [i for i in range(output_size)]
        samples = np.random.choice(numbers, size=int(k), replace=False)
        result = np.zeros(np.shape(numbers))
        for number in samples:
            result[number] = 1
        return result

    def get_expectations(self, qc_list, backend=None,
                         eval_counts=None, shots=10000,
                         mitigation=None, memory=False):
        """
        Gets expectation values and counts for the circuits in qc_list.
        Uses the evaluation function eval_counts that takes the counts
        of a measurement and returns a value or array of values.
        Arguments:
            qc_list (list of QuantumCircuits): Circuits that will run,
                                               of which the expectation
                                               value is wanted.
            backend (backend): Backend on which the circuits will run.
            eval_counts (function): Function that calculates a value or
                                    array of values from the counts.
            shots (int): Number of shots per circuit.
            mitigation (str): Measurement mitigation method: Either
                - 'tensored': Tensored Measurement mitigation provided
                    by qiskit.
                - 'mthree': Measurement mitigation as provided by the
                    mthree package.
                    More info: 
                    https://qiskit.org/documentation/partners/mthree/
        Returns:
            expect_vals (list): Expectation values of the circuits.
            qc_counts (list of dict): A list of counts for the circuits
                                      in qc_list.
        """
        if backend is None:
            backend = self.backend
        if len(qc_list) == 0:
            return None
        qc_counts = self.run_circs(qc_list, backend, mitigation=mitigation,
                                   shots=int(shots/self.per_n_val),
                                   memory=memory)
        self.counts_per_n = qc_counts  # before averaging!
        qc_counts = self.avg_over_per_n_vals(qc_counts, self.per_n_val)
        self.raw_counts = qc_counts
        self.raw_expect_vals = self.apply_eval_counts(eval_counts, qc_counts)

        if mitigation is not None:
            qc_counts = self.avg_over_per_n_vals(
                self.mitigated_counts_per_n, self.per_n_val)
            self.mitigated_counts = qc_counts
            self.mit_expect_vals = self.apply_eval_counts(eval_counts,
                                                          qc_counts)

        if eval_counts is not None:
            if mitigation is not None:
                self.expect_vals = self.mit_expect_vals
            else:
                self.expect_vals = self.raw_expect_vals
            return self.expect_vals, qc_counts
        else:
            return qc_counts

    def apply_eval_counts(self, eval_counts, qc_counts):
        """
        Applies the eval_counts function to a list of counts.
        """
        if eval_counts is None:
            return None

        expect_vals = []
        for counts in qc_counts:
            expect_vals += [eval_counts(counts)]
        return expect_vals

    def run_circs(self, qc_list, backend, shots=10000, mitigation=None,
                  memory=False):
        """
        Runs the given circuits qc_list and applies measurement
        mitigation if specified.
        Arguments:
            qc_list (list of QuantumCircuits):
            backend (backend): Backend to be used
            shots (int): number of shots per circuit
            mitgation (str): Mitigation method to be applied
            memory (bool): If True, per-shot measurement bitstrings
                            are returned
        Returns:
            qc_counts (list of dict): list of counts corresponding to
                                      the given circuits qc_list.
        """
        if type(qc_list) == QuantumCircuit:
            qc_list = [qc_list]
        if mitigation is not None and not self.mitigator_available:
            self.mitigation_method = mitigation
            self.mitigator = MeasurementMitigation(qc_list[0], backend,
                                                   method=mitigation)
        job = backend.run(qc_list, shots=shots, memory=memory)
        if mitigation is not None and not self.mitigator_available:
            self.mitigator.run_mitigation_circs()
            self.mitigator_available = True
        result = job.result()
        self.result = result
        if memory:
            self.memory = []
            for qc in qc_list:
                self.memory += [result.get_memory(qc)]

        qc_counts = []
        for qc in qc_list:
            qc_counts += [result.get_counts(qc)]
        if mitigation is not None:
            self.mitigated_counts_per_n = self.mitigator.get_mitigated_counts(
                result, qc_list)
        return qc_counts

    def avg_over_per_n_vals(self, qc_counts, per_n_val):
        """
        Averages over the counts of the circuits that correspond to
        the same n-value. There are always per_n_val many circuits
        corresponding to one n.
        """
        avg_counts = []
        for i in range(int(len(qc_counts)/per_n_val)):
            avg_counts += [{}]
            for to_avg in range(per_n_val):
                considered_counts = qc_counts[i*per_n_val + to_avg]
                for counts in considered_counts:
                    if counts in avg_counts[-1]:
                        avg_counts[-1][counts] += considered_counts[counts]
                    else:
                        avg_counts[-1][counts] = considered_counts[counts]
        return avg_counts

    def fit(self, x=None, y=None, fit_method="pol", order=2):
        """
        Fits the data x, y with the specified method/fit and order and
        evaluates at n = - 1 / 2. Mainly calls the function poly_fit and
        exp_fit, while saving the specifications for possible later use.
        Arguments:
            x (list/array): list of x-values, noise scaling factors
            y (list/array): list of y-values, expectation values
            fit_method (str): 'pol' or 'exp'
            order (int): Order of the fit (for 'exp' order of exponent in
                         the exponential function).
        Returns:
            result (float or list of floats): Extrapolated value
            model (array): Parameters of the fit.
            residuals (array): Residuals of the fitted function to the
                               given (x, y) pairs.
        """
        if x is None:
            x = self.real_n_vals
        if y is None:
            y = self.expect_vals
        if fit_method == "pol":
            result, model, residuals = self.poly_fit(x, y, order=order)
        elif fit_method == "exp":
            y = np.transpose(y)
            result, model, residuals = self.exp_fit(x, y, order=order)
        self.fit_result = result
        self.fit_params = model
        self.fit_residuals = residuals
        self.fit_method = fit_method
        self.fit_order = order
        return result, model, residuals

    def poly_fit(self, x, y, order=2, out_val=-1/2):
        """
        Polynomial fit of specified order with evaluation
        at n = out_val.
        """
        assert len(x) >= order + 1,\
            "The number of datapoints is to low" + \
            " for the fit: {} for order {}".format(len(x), order)
        out_vect = np.array([(out_val)**i for i in range(order+1)])
        model, rest = polyfit(x, y, order, full=True)
        model = np.transpose(model)
        result = np.matmul(model, out_vect)
        return result, model, rest[0]

    def exp_fit(self, x, y, order=1, out_val=-1/2):
        """
        Exponential fit of specified order with evaluation
        at n = out_val. Fits via an optimizer to:
            f(x) = a + exp(-g(x))
        Where g(x) is a polynom of specified order.
        """
        if not isinstance(y[0], np.ndarray):
            y = np.array([y])
        result, model, residuals = [[], [], []]
        for y_i in y:
            try:
                model_i, pcov = curve_fit(self.apply_exponential, x, y_i,
                                          p0=[1.0 for i in range(order+2)])
                y_fit = self.apply_exponential(x, *model_i)
                residuals_i = np.array(np.sum((y_fit - y_i)**2))
                result_i = self.apply_exponential(np.array(out_val), *model_i)
            except RuntimeError as RE:
                result_i, model_i, residuals_i = [np.nan, np.nan, np.nan]
            result += [result_i]
            model += [model_i]
            residuals += [residuals_i]
        return result, np.array(model), np.array(residuals)

    def apply_exponential(self, x, a, *args):
        """
        Returns:
            f(x) = a + exp(-g(x))
        Where g(x) is a polynom in the order of len(args).
        """
        if type(x) == list:
            x = np.array(x)
        exponent = np.zeros(np.shape(x))
        for (power, i) in enumerate(args):
            exponent = exponent + i*x**(power)
        return a + np.exp(-exponent, dtype=np.float64)

    def to_dict(self):
        """
        Turns the attributes of this class instance (self) into a dict.
        """
        dic = self.__dict__
        new_dic = {}
        for key in dic:
            if "qc" in key or key == "noise_model" or key == "eval_counts"\
                    or key == "coupling_map" or key == "cx_gates"\
                    or key == "result" or key == "mitigator":
                pass
            elif "backend" in key:
                new_dic[key] = str(dic[key])
            else:
                new_dic[key] = dic[key]
        return new_dic

    def save_json(self, file=None, overwrite=False, dirpath="ZNEdata",
                  indent=1, **kwargs):
        """
        Saves the attribute of this class instance (self) into a json
        file.
        Arguments:
            file (str): filename
            overwrite (bool): whether or not an already existing file
                              will be overwritten (else consecutive
                              number is added)
            dirpath (str): (Dir)-Path to the folder where the file will
                            be saved in.
            indent (int): Indentation of the json file.
        Returns: --
            -
        """
        Path(dirpath).mkdir(parents=True, exist_ok=True)
        if file is None:
            file = "data_" +\
                   strftime("%Y-%m-%d_%H-%M-%S", localtime()) + ".json"
        file_new = file
        count = 1
        while isfile(join(dirpath, file_new)) is True and overwrite is False:
            file_new = file[:-5] + '__' + str(count) + '.json'
            count += 1
        file = file_new
        dic = self.to_dict()
        with open(join(dirpath, file), 'w') as wfile:
            json.dump(dic, wfile, indent=indent, cls=NpEncoder)

    def from_json(self, file):
        """Sets attributes of a saved json file to a ZNE object"""
        with open(file, 'r') as rfile:
            dic = json.load(rfile)
            for key in dic:
                setattr(self, key, dic[key])

    def plot(self, mit=True, label=""):
        """
        Plots the result of a ZNE.
        NOT FINAL (Multivalue results not yet tested.)
        """
        x = self.real_n_vals
        method = self.fit_method
        if mit and self.mit_expect_vals is not None:
            y = self.mit_expect_vals
            result = self.fit_result
        elif not mit and self.mit_expect_vals is not None:
            y = self.raw_expect_vals
            result, _, _ = self.fit(x, y, fit_method=method,
                                    order=self.fit_order)
        else:
            y = self.raw_expect_vals
            result = self.fit_result

        if method == "pol":
            new_x, new_y = self.get_polxy(self.fit_params, x[-1])
        elif method == "exp":
            new_x, new_y = self.get_expxy(self.fit_params, x[-1])
        r = 2*np.array(x)+1
        new_r = 2*np.array(new_x)+1
        ax = plt.gca()
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(r, y, "o-", label="Expec.val "+label)
        plt.plot(new_r, new_y, "--", color=color, label="Fit "+label)
        plt.plot(0, result, "o", color=color, label="ZNE "+label)
        plt.xlabel("noise scaling factor")
        plt.ylabel("expectation values")
        plt.legend(loc="best", prop={"size": 10})
        plt.text(0, result, format(result, ".5g"))

    def get_polxy(self, params, end):
        """Helper function that returns x,y pairs of a pol. fit."""
        pol = Polynomial(params)
        new_x, new_y = pol.linspace(100, [-1/2, end])
        return new_x, new_y

    def get_expxy(self, params, end):
        """Helper function that returns x,y pairs of a exp. fit."""
        new_x = np.linspace(-1/2, end, 100)
        new_y = self.apply_exponential(new_x, *params)
        return new_x, new_y


class MeasurementMitigation():
    """
    Not final yet. Implements the measurement mitigation of mthree
    and the tensored method of qiskit in a basic way.
    """
    def __init__(self, qc, backend, method="mthree"):
        self.qc = qc
        self.final_map = self.final_mapping(qc)
        self.method = method
        self.backend = backend
        if method == "mthree":
            self.mit = mthree.M3Mitigation(backend)
        elif method == "tensored":
            self.mit_pattern = [[i] for i in self.final_map]
            meas_calibs, state_labels = tensored_meas_cal(
                    mit_pattern=self.mit_pattern, qr=None, circlabel='mcal')
            self.qc_mit = transpile(meas_calibs, backend, optimization_level=2)

    def run_mitigation_circs(self, shots=10000):
        if self.method == 'mthree':
            self.mit.cals_from_system(self.final_map, method='independent',
                                      shots=shots)
        elif self.method == 'tensored':
            self.mit_job = self.backend.run(self.qc_mit, shots=shots)

    def get_mitigated_counts(self, result, qc_list, **kwargs):
        if type(qc_list) == QuantumCircuit:
            qc_list = [qc_list]

        qc_counts = []

        if self.method == "tensored":
            result = self.apply_tensored(result, **kwargs)
        for qc in qc_list:
            counts = result.get_counts(qc)
            if self.method == "mthree":
                counts = self.apply_mthree(counts)
            qc_counts += [counts]

        if len(qc_counts) == 1:
            qc_counts = qc_counts[0]
        return qc_counts

    def apply_tensored(self, result, meas_filter=None, **kwargs):
        if meas_filter is None:
            meas_filter = self.get_meas_filter(**kwargs)
        self.mitigated_result = meas_filter.apply(result,
                                                  method='least_squares')
        return self.mitigated_result

    def get_meas_filter(self, mit_circ_result=None, mit_pattern=None,
                        **kwargs):
        if mit_circ_result is None:
            mit_circ_result = self.mit_job.result()
        if mit_pattern is None:
            mit_pattern = self.mit_pattern
        meas_fitter = TensoredMeasFitter(mit_circ_result, mit_pattern,
                                         circlabel='mcal')
        self.meas_filter = meas_fitter.filter
        return self.meas_filter

    def apply_mthree(self, counts):
        return self.mit.apply_correction(counts, self.final_map)

    def final_mapping(self, qc):
        return mthree.utils.final_measurement_mapping(qc)


class NpEncoder(json.JSONEncoder):
    """
    Numpy encoder for json to save date and time in json-files.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif np.iscomplexobj(obj):
            obj = [obj.real, obj.imag]
            return obj
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            print(obj)
            return super(NpEncoder, self).default(obj)
