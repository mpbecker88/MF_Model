from abc import ABC, abstractmethod
from enum import Enum, auto
from sqlite3 import DatabaseError
from typing import Callable, List, Union

import sympy as sp

import data_parameters as dp


class ModelName(Enum):
	COBAIF = auto()
	NMDA = auto()
	INTERPOLATED = auto()
	EXP_INTERPOLATED = auto()


class ModelType(ABC):
	def __init__(self, Vth_val: float, Vr_val: float, tauL_val: float, tauR_val: float):
		self.model_name: ModelName
		self.Vth: sp.Symbol = sp.symbols('V_th', constant=True, positive=False)
		self.Vr: sp.Symbol = sp.symbols('V_r', constant=True, positive=False)
		self.tauL: sp.Symbol = sp.symbols('tau_L', constant=True, positive=True)
		self.tauR: sp.Symbol = sp.symbols('tau_R', constant=True, positive=True)
		self.V: sp.Symbol = sp.symbols('V')

		self.nonlinear_term: Union[sp.Symbol, float]
		self.vec_nu = sp.Matrix
		self.vec_w: sp.Matrix
		self.vec_tau: sp.Matrix
		self.vec_E: sp.Matrix
		self.vec_K: sp.Matrix
		self.modulating_functions: sp.Matrix

		self.data_parameters: dp.DataParameters = dp.DataParameters()

		self.data_parameters.add_variable(self.V)
		self.data_parameters.add_value(self.Vth, Vth_val)
		self.data_parameters.add_value(self.Vr, Vr_val)
		self.data_parameters.add_value(self.tauL, tauL_val)
		self.data_parameters.add_value(self.tauR, tauR_val)

	@abstractmethod
	def _define_modulating_functions(self) -> sp.Matrix:
		pass

	@abstractmethod
	def generate_lambda_functions(self, expressions_list: List[sp.core.expr.Expr]) -> List[Callable]:
		pass


class MFCoBaIF(ModelType):
	def __init__(self, nuE_val: float,
					nuI_val: float,
					wE_val: float,
					wI_val: float,
					tauE_val: float,
					tauI_val: float,
					KE_val: float,
					KI_val: float,
					EL_val: float = -60,
					EE_val: float = 0,
					EI_val: float = -80,
					Vth_val: float = -50,
					Vr_val: float = -60,
					tauL_val: float = 20,
					tauR_val: float = 2
				):

		super().__init__(Vth_val, Vr_val, tauL_val, tauR_val)

		EL = sp.symbols('E_L', constant=True)
		EE = sp.symbols('E_E', constant=True)
		EI = sp.symbols('E_I', constant=True)
		nuE = sp.symbols('nu_E', constant=True, positive=True)
		nuI = sp.symbols('nu_I', constant=True, positive=True)
		wE = sp.symbols('w_E', constant=True, positive=True)
		wI = sp.symbols('w_I', constant=True, positive=True)
		tauE = sp.symbols('tau_E', constant=True, positive=True)
		tauI = sp.symbols('tau_I', constant=True, positive=True)
		KE = sp.symbols('K_E', constant=True, positive=True)
		KI = sp.symbols('K_I', constant=True, positive=True)

		self.model_name = ModelName.COBAIF
		self.nonlinear_term = 0
		self.vec_nu = sp.Matrix([1., nuE, nuI])
		self.vec_w = sp.Matrix([1., wE, wI])
		self.vec_tau = sp.Matrix([1., tauE, tauI])
		self.vec_E = sp.Matrix([EL, EE, EI])
		self.vec_K = sp.Matrix([1., KE, KI])

		self.data_parameters.add_value(nuE, nuE_val)
		self.data_parameters.add_value(nuI, nuI_val)
		self.data_parameters.add_value(wE, wE_val)
		self.data_parameters.add_value(wI, wI_val)
		self.data_parameters.add_value(tauE, tauE_val)
		self.data_parameters.add_value(tauI, tauI_val)
		self.data_parameters.add_value(EL, EL_val)
		self.data_parameters.add_value(EE, EE_val)
		self.data_parameters.add_value(EI, EI_val)
		self.data_parameters.add_value(KE, KE_val)
		self.data_parameters.add_value(KI, KI_val)

		self.modulating_functions = self._define_modulating_functions()

	def _define_modulating_functions(self) -> sp.Matrix:
		return sp.Matrix([1., 1., 1.])

	def generate_lambda_functions(self, expressions_list: List[sp.core.expr.Expr]) -> List[Callable]:
		return self.data_parameters.transform_expressions_to_functions(expressions_list)


class MFInterpolatedCoBaIF(ModelType):
	def __init__(self, alpha_val: float,
					nuF_val: float,
					nuS_val: float,
					nuI_val: float,
					wF_val: float,
					wS_val: float,
					wI_val: float,
					tauF_val: float,
					tauS_val: float,
					tauI_val: float,
					KF_val: float,
					KS_val: float, 
					KI_val: float,
					EL_val: float = -60, 
					EF_val: float = 0,
					ES_val: float = 0,
					EI_val: float = -80,
					Vth_val: float = -50,
					Vr_val: float = -60,
					tauL_val: float = 20,
					tauR_val: float = 2
				):

		super().__init__(Vth_val, Vr_val, tauL_val, tauR_val)

		alpha = sp.symbols('alpha', positive = True)
		EL = sp.symbols('E_L', constant=True)
		EF = sp.symbols('E_F', constant=True)
		ES = sp.symbols('E_S', constant=True)
		EI = sp.symbols('E_I', constant=True)
		nuF = sp.symbols('nu_F', constant=True, positive=True)
		nuS = sp.symbols('nu_S', constant=True, positive=True)
		nuI = sp.symbols('nu_I', constant=True, positive=True)
		wF = sp.symbols('w_F', constant=True, positive=True)
		wS = sp.symbols('w_S', constant=True, positive=True)
		wI = sp.symbols('w_I', constant=True, positive=True)
		tauF = sp.symbols('tau_F', constant=True, positive=True)
		tauS = sp.symbols('tau_S', constant=True, positive=True)
		tauI = sp.symbols('tau_I', constant=True, positive=True)
		KF = sp.symbols('K_F', constant=True, positive=True)
		KS = sp.symbols('K_S', constant=True, positive=True)
		KI = sp.symbols('K_I', constant=True, positive=True)

		self.model_name = ModelName.INTERPOLATED
		self.alpha = alpha
		self.nonlinear_term = 0
		self.vec_nu = sp.Matrix([1., nuF, nuS, nuI])
		self.vec_w = sp.Matrix([1., wF, wS, wI])
		self.vec_tau = sp.Matrix([1., tauF, tauS, tauI])
		self.vec_E = sp.Matrix([EL, EF, ES, EI])
		self.vec_K = sp.Matrix([1., KF, KS, KI])

		self.data_parameters.add_value(alpha, alpha_val)
		self.data_parameters.add_value(nuF, nuF_val)
		self.data_parameters.add_value(nuS, nuS_val)
		self.data_parameters.add_value(nuI, nuI_val)
		self.data_parameters.add_value(wF, wF_val)
		self.data_parameters.add_value(wS, wS_val)
		self.data_parameters.add_value(wI, wI_val)
		self.data_parameters.add_value(tauF, tauF_val)
		self.data_parameters.add_value(tauS, tauS_val)
		self.data_parameters.add_value(tauI, tauI_val)
		self.data_parameters.add_value(EL, EL_val)
		self.data_parameters.add_value(EF, EF_val)
		self.data_parameters.add_value(ES, ES_val)
		self.data_parameters.add_value(EI, EI_val)
		self.data_parameters.add_value(KF, KF_val)
		self.data_parameters.add_value(KS, KS_val)
		self.data_parameters.add_value(KI, KI_val)

		self.modulating_functions = self._define_modulating_functions()

	def _define_modulating_functions(self) -> sp.Matrix:
		return sp.Matrix([1., (1 - self.alpha), self.alpha, 1.])

	def generate_lambda_functions(self, expressions_list: List[sp.core.expr.Expr]) -> List[Callable]:
		return self.data_parameters.transform_expressions_to_functions(expressions_list)


class MFExpInterpolatedCoBaIF(ModelType):
	def __init__(self, alpha_val: float,
					nuF_val: float,
					nuS_val: float,
					nuI_val: float,
					wF_val: float,
					wS_val: float,
					wI_val: float,
					tauF_val: float,
					tauS_val: float,
					tauI_val: float,
					KF_val: float,
					KS_val: float, 
					KI_val: float,
					EL_val: float = -60,
					ET_val: float = -43,
					EF_val: float = 0,
					ES_val: float = 0,
					EI_val: float = -80,
					deltaT_val: float = 1,
					Vth_val: float = 0,
					Vr_val: float = -60,
					tauL_val: float = 20,
					tauR_val: float = 2
				):

		super().__init__(Vth_val, Vr_val, tauL_val, tauR_val)

		alpha = sp.symbols('alpha', positive = True)
		EL = sp.symbols('E_L', constant=True)
		ET = sp.symbols('E_T', constant=True)
		EF = sp.symbols('E_F', constant=True)
		ES = sp.symbols('E_S', constant=True)
		EI = sp.symbols('E_I', constant=True)
		nuF = sp.symbols('nu_F', constant=True, positive=True)
		nuS = sp.symbols('nu_S', constant=True, positive=True)
		nuI = sp.symbols('nu_I', constant=True, positive=True)
		wF = sp.symbols('w_F', constant=True, positive=True)
		wS = sp.symbols('w_S', constant=True, positive=True)
		wI = sp.symbols('w_I', constant=True, positive=True)
		tauF = sp.symbols('tau_F', constant=True, positive=True)
		tauS = sp.symbols('tau_S', constant=True, positive=True)
		tauI = sp.symbols('tau_I', constant=True, positive=True)
		KF = sp.symbols('K_F', constant=True, positive=True)
		KS = sp.symbols('K_S', constant=True, positive=True)
		KI = sp.symbols('K_I', constant=True, positive=True)
		deltaT = sp.symbols('Delta_T', constant=True, positive=True)

		self.model_name = ModelName.EXP_INTERPOLATED
		self.alpha = alpha
		self.nonlinear_term = deltaT*sp.exp((self.V - ET)/deltaT)
		self.vec_nu = sp.Matrix([1., nuF, nuS, nuI])
		self.vec_w = sp.Matrix([1., wF, wS, wI])
		self.vec_tau = sp.Matrix([1., tauF, tauS, tauI])
		self.vec_E = sp.Matrix([EL, EF, ES, EI])
		self.vec_K = sp.Matrix([1., KF, KS, KI])

		self.data_parameters.add_value(alpha, alpha_val)
		self.data_parameters.add_value(nuF, nuF_val)
		self.data_parameters.add_value(nuS, nuS_val)
		self.data_parameters.add_value(nuI, nuI_val)
		self.data_parameters.add_value(wF, wF_val)
		self.data_parameters.add_value(wS, wS_val)
		self.data_parameters.add_value(wI, wI_val)
		self.data_parameters.add_value(tauF, tauF_val)
		self.data_parameters.add_value(tauS, tauS_val)
		self.data_parameters.add_value(tauI, tauI_val)
		self.data_parameters.add_value(EL, EL_val)
		self.data_parameters.add_value(ET, ET_val)
		self.data_parameters.add_value(EF, EF_val)
		self.data_parameters.add_value(ES, ES_val)
		self.data_parameters.add_value(EI, EI_val)
		self.data_parameters.add_value(KF, KF_val)
		self.data_parameters.add_value(KS, KS_val)
		self.data_parameters.add_value(KI, KI_val)
		self.data_parameters.add_value(deltaT, deltaT_val)

		self.modulating_functions = self._define_modulating_functions()

	def _define_modulating_functions(self) -> sp.Matrix:
		return sp.Matrix([1., (1 - self.alpha), self.alpha, 1.])

	def generate_lambda_functions(self, expressions_list: List[sp.core.expr.Expr]) -> List[Callable]:
		return self.data_parameters.transform_expressions_to_functions(expressions_list)

class MFNMDA(ModelType):
	def __init__(self, alpha_val: float,
					nuA_val: float,
					nuN_val: float,
					nuI_val: float,
					wA_val: float,
					wN_val: float,
					wI_val: float,
					tauA_val: float,
					tauN_val: float,
					tauI_val: float,
					KA_val: float,
					KN_val: float, 
					KI_val: float,
					EL_val: float = -60, 
					EA_val: float = 0,
					EN_val: float = 0,
					EI_val: float = -80,
					Vth_val: float = -50,
					Vr_val: float = -60,
					tauL_val: float = 20,
					tauR_val: float = 2,
					beta_val: float = 0.062,
					gamma_val: float = 3.57,
					n_Mg_val: float = 1
				):
		super().__init__(Vth_val, Vr_val, tauL_val, tauR_val)

		alpha = sp.symbols('alpha', positive=True)
		EL = sp.symbols('E_L', constant=True)
		EA = sp.symbols('E_A', constant=True)
		EN = sp.symbols('E_N', constant=True)
		EI = sp.symbols('E_I', constant=True)
		nuA = sp.symbols('nu_A', constant=True, positive=True)
		nuN = sp.symbols('nu_N', constant=True, positive=True)
		nuI = sp.symbols('nu_I', constant=True, positive=True)
		wA = sp.symbols('w_A', constant=True, positive=True)
		wN = sp.symbols('w_N', constant=True, positive=True)
		wI = sp.symbols('w_I', constant=True, positive=True)
		tauA = sp.symbols('tau_A', constant=True, positive=True)
		tauN = sp.symbols('tau_N', constant=True, positive=True)
		tauI = sp.symbols('tau_I', constant=True, positive=True)
		KA = sp.symbols('K_A', constant=True, positive=True)
		KN = sp.symbols('K_N', constant=True, positive=True)
		KI = sp.symbols('K_I', constant=True, positive=True)
		beta = sp.symbols('beta', constant=True, positive=True)
		gamma = sp.symbols('gamma', constant=True, positive=True)
		n_Mg = sp.symbols('n_Mg', constant=True, positive=True)

		self.model_name = ModelName.NMDA
		self.alpha = sp.symbols('alpha', positive=True)
		self.nonlinear_term = 0
		self.vec_nu = sp.Matrix([1., nuA, nuN, nuI])
		self.vec_w = sp.Matrix([1., wA, wN, wI])
		self.vec_tau = sp.Matrix([1., tauA, tauN, tauI])
		self.vec_E = sp.Matrix([EL, EA, EN, EI])
		self.vec_K = sp.Matrix([1., KA, KN, KI])
		self.beta = beta
		self.gamma = gamma
		self.n_Mg = n_Mg

		self.data_parameters.add_value(alpha, alpha_val)
		self.data_parameters.add_value(nuA, nuA_val)
		self.data_parameters.add_value(nuN, nuN_val)
		self.data_parameters.add_value(nuI, nuI_val)
		self.data_parameters.add_value(wA, wA_val)
		self.data_parameters.add_value(wN, wN_val)
		self.data_parameters.add_value(wI, wI_val)
		self.data_parameters.add_value(tauA, tauA_val)
		self.data_parameters.add_value(tauN, tauN_val)
		self.data_parameters.add_value(tauI, tauI_val)
		self.data_parameters.add_value(EL, EL_val)
		self.data_parameters.add_value(EA, EA_val)
		self.data_parameters.add_value(EN, EN_val)
		self.data_parameters.add_value(EI, EI_val)
		self.data_parameters.add_value(KA, KA_val)
		self.data_parameters.add_value(KN, KN_val)
		self.data_parameters.add_value(KI, KI_val)
		self.data_parameters.add_value(beta, beta_val)
		self.data_parameters.add_value(n_Mg, n_Mg_val)
		self.data_parameters.add_value(gamma, gamma_val)

		self.modulating_functions = self._define_modulating_functions()

	def _define_modulating_functions(self) -> sp.Matrix:
		return sp.Matrix([1., (1 - self.alpha), self.alpha*self._s(), 1.])

	def generate_lambda_functions(self, expressions_list: List[sp.core.expr.Expr]) -> List[Callable]:
		return self.data_parameters.transform_expressions_to_functions(expressions_list)

	def _s(self) -> sp.Symbol:
		beta = self.beta
		n_Mg = self.n_Mg
		gamma = self.gamma
		V = self.V

		return 1 / (1 + (n_Mg / gamma) * sp.exp(-beta * V))

def main():
	model_type = MFNMDA(alpha_val = 0,
					EL_val = -60,
					EA_val = 0,
					EN_val = 0,
					EI_val = -80,
					nuA_val = 0.005,
					nuN_val = 0.005,
					nuI_val = 0.005,
					wA_val = 0.1,
					wN_val = 0.1,
					wI_val = 0.4,
					tauA_val = 5,
					tauN_val = 100,
					tauI_val = 10,
					KA_val = 400,
					KN_val = 400, 
					KI_val = 100,
					Vth_val= -50,
					Vr_val = -60,
					tauL_val = 20,
					tauR_val = 2
				)

	# model_type = MFCoBaIF(EL_val = -60,
	# 				EE_val = 0,
	# 				EI_val = -80,
	# 				nuE_val = 0.005,
	# 				nuI_val = 0.005,
	# 				wE_val = 0.1,
	# 				wI_val = 0.4,
	# 				tauE_val = 5,
	# 				tauI_val = 10,
	# 				KE_val = 400,
	# 				KI_val = 100,
	# 				Vth_val = -50,
	# 				Vr_val = -60,
	# 				tauL_val = 20,
	# 				tauR_val = 2
	# 				)

	vec_w = model_type.vec_w
	vec_K = model_type.vec_K
	vec_nu = model_type.vec_nu
	vec_tau = model_type.vec_tau
	expressions_list = sp.matrix_multiply_elementwise(sp.matrix_multiply_elementwise(sp.matrix_multiply_elementwise(vec_w, vec_K), vec_tau), vec_nu).tolist()
	expressions_list = [item for sublist in expressions_list for item in sublist]
	mu0, mu1, mu2, mu3 = model_type.generate_lambda_functions(expressions_list)

	print(mu0(-70), mu1(-70), mu2(-70), mu3(-70))
if __name__ == "__main__":
	main()
