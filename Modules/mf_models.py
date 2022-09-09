from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import sympy as sp

import model_type as mt


@dataclass(slots=True)
class LangevinModel:
	model_type: mt.ModelType

	vec_mu: sp.Matrix = field(init=False)
	vec_sigma: sp.Matrix = field(init=False)
	mu: sp.core.expr.Expr = field(init=False)
	tau: sp.core.expr.Expr = field(init=False)
	sigmaV: sp.core.expr.Expr = field(init=False)

	vec_h: sp.Matrix = field(init=False)

	def __post_init__(self):
		self._recalculate()

	def _recalculate(self):
		self.vec_mu = self._calculate_vector_mu()
		self.vec_sigma = self._calculate_vector_sigma()
		self.tau = self._calculate_tau()
		self.mu = self._calculate_mu()
		self.vec_h = self._calculate_vector_h()
		self.sigmaV = self._calculate_sigmaV()

	def _calculate_vector_mu(self) -> sp.Matrix:
		vec_w = self.model_type.vec_w
		vec_K = self.model_type.vec_K
		vec_nu = self.model_type.vec_nu
		vec_tau = self.model_type.vec_tau

		return sp.matrix_multiply_elementwise(sp.matrix_multiply_elementwise(sp.matrix_multiply_elementwise(vec_w, vec_K), vec_tau), vec_nu)

	def _calculate_vector_sigma(self) -> sp.Matrix:
		vec_w = self.model_type.vec_w
		vec_K = self.model_type.vec_K
		vec_nu = self.model_type.vec_nu
		vec_tau = self.model_type.vec_tau

		return sp.matrix_multiply_elementwise(vec_w, (sp.matrix_multiply_elementwise(sp.matrix_multiply_elementwise(vec_K, vec_tau), vec_nu).applyfunc(sp.sqrt)))

	def _calculate_tau(self) -> sp.core.expr.Expr:
		tau_L = self.model_type.tauL
		vec_mu = self.vec_mu
		mod_funcs = self.model_type.modulating_functions

		return tau_L / vec_mu.dot(mod_funcs)

	def _calculate_mu(self) -> sp.core.expr.Expr:
		tau_L = self.model_type.tauL
		tau = self.tau
		vec_E = self.model_type.vec_E
		vec_mu = self.vec_mu
		mod_funcs = self.model_type.modulating_functions
		
		return (tau / tau_L) * vec_E.dot(sp.matrix_multiply_elementwise(vec_mu, mod_funcs))

	def _calculate_sigmaV(self) -> sp.core.expr.Expr:
		sigmaV:sp.Symbol = 0  # type: ignore
		for i in range(1, len(self.vec_h)):
			sigmaV += self.tau**2*self.vec_h[i]**2/(self.tau + self.model_type.vec_tau[i])

		return sigmaV.subs(self.model_type.V, self.mu)

	def _calculate_vector_h(self) -> sp.Matrix:
		V = self.model_type.V
		tau_L = self.model_type.tauL
		tau = self.tau
		mu = self.mu
		vec_E = self.model_type.vec_E
		vec_tau = self.model_type.vec_tau
		vec_sigma = self.vec_sigma
		mod_funcs = self.model_type.modulating_functions
		
		vec_h = []
		
		vec_h.append((mu - V + self.model_type.nonlinear_term) / tau)

		for i in range(1, vec_E.rows):
			vec_h.append(mod_funcs[i] * (sp.sqrt(vec_tau[i]) / tau_L) * vec_sigma[i] * (vec_E[i] - V))

		return sp.Matrix(vec_h)
	
	def calculate_mu_value(self):
		return self.model_type.data_parameters.substitute_parameters_values([self.mu])[0]

	# def calculate_sigma_value(self):
	# 	sigma = 0
	# 	sigma = np.sqrt(((self.tau * self.W[1]**2) / (self.tau + self.tau_E)) + ((self.tau * self.h_I**2) / (self.tau + self.tau_I)))


class FokkerPlanckModel:
	def __init__(self, model: LangevinModel, multiplicative: bool):
		self.model = model
		self.multiplicative = multiplicative

		if not multiplicative:
			self.evaluate_effective_tc_approx()

		self.vec_S = self._calculate_S()
		self.Xi = self._calculate_Xi()
		self.B = self._calculate_B()
		self.H = self._calculate_H()

		self.lambda_B, self.lambda_H = model.model_type.generate_lambda_functions([self.B, self.H])
		
	
	def evaluate_effective_tc_approx(self):
		for i in range(1, self.model.vec_h.shape[0]):
			self.model.vec_h[i] = self.model.vec_h[i].subs(self.model.model_type.V, self.model.mu)


	def _calculate_S(self):
		epsilon = 0.5
		V = self.model.model_type.V
		vec_h = self.model.vec_h
		vec_tau = self.model.model_type.vec_tau
		S = []

		W = vec_h[0]
		dW = sp.diff(W, V)

		S.append(-W)

		for h_i, tau_i in zip(vec_h[1:], vec_tau[1:]):
			dh_i = sp.diff(h_i, V)

			denominator = (1 - tau_i * (dW - (dh_i / h_i) * W))
			S_i = (1/2) * h_i / sp.Piecewise((-epsilon, sp.Interval(-epsilon, 0).contains(denominator)), (epsilon, sp.Interval(0, epsilon).contains(denominator)), (denominator,  True))
			
			S.append(S_i)

		return sp.Matrix(S)

	def _calculate_Xi(self):
		vec_h_1p = sp.Matrix(self.model.vec_h[1:])
		vec_S_1p = sp.Matrix(self.vec_S[1:])

		return (vec_h_1p.dot(vec_S_1p))
	
	def _calculate_B(self):
		V = self.model.model_type.V
		Xi = self.Xi
		S0 = -self.model.vec_h[0]
		vec_W_1p = sp.Matrix(self.model.vec_h[1:])
		vec_S_1p = sp.Matrix(self.vec_S[1:])
		vec_dS_1p = sp.diff(vec_S_1p, V)
		
		B = (S0 + vec_W_1p.dot(vec_dS_1p)) / Xi
		
		return B

	def _calculate_H(self):
		V = self.model.model_type.V
		V_r = self.model.model_type.Vr
		Xi = self.Xi

		return sp.Heaviside(V - V_r) / Xi


	def integrate_p0(self, vec_Vk: npt.NDArray) -> npt.NDArray:
		"""Integrate the unnormalized probability distribution p0"""

		vec_p0_fliped = np.zeros_like(vec_Vk)
		dV = vec_Vk[1] - vec_Vk[0]
		vec_Vk_fliped = np.flip(vec_Vk)

		vec_p0_fliped[0] = 0

		for j, (Vk, p0) in enumerate(zip(vec_Vk_fliped[:-1], vec_p0_fliped[:-1])):
			Bk = self.lambda_B(Vk)
			Hk = self.lambda_H(Vk)
			if(-0.000001 <= Bk <= 0.000001):
				vec_p0_fliped[j+1] = p0 * np.exp(dV * Bk) + dV * Hk
			else:
				vec_p0_fliped[j+1] = p0 * np.exp(dV * Bk) + dV * Hk * (np.exp(dV*Bk) - 1) / (dV * Bk)

		vec_p0 = np.flip(vec_p0_fliped)

		return vec_p0

	def calculate_firing_rate(self, vec_Vk: npt.NDArray) -> float:
		tauR = self.model.model_type.tauR
		tauR_val = self.model.model_type.data_parameters.values[tauR]
		vec_p0 = self.integrate_p0(vec_Vk)
		dV = vec_Vk[1] - vec_Vk[0]

		return 1 / (tauR_val + dV * np.sum(vec_p0))


def main():
	# model_type = mt.MFCoBaIF(EL_val = -60,
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

	# model_type = mt.MFInterpolatedCoBaIF(alpha_val = 0.1, 
	#					 					EL_val = -60,
	#					 					EF_val = 0,
	#					 					ES_val = 0,
	#					 					EI_val = -80,
	#					 					wF_val = 0.1,
	#					 					wS_val = 0.1,
	#					 					wI_val = 0.4,
	#					 					tauF_val = 5,
	#					 					tauS_val = 100,
	#					 					tauI_val = 10,
	#					 					nuF_val = 0.005,
	#					 					nuS_val = 0.005,
	#					 					nuI_val = 0.005,
	#					 					KF_val = 400,
	#					 					KS_val = 400,
	#					 					KI_val = 100,
	#					 					Vth_val = -50,
	#					 					Vr_val = -60,
	#					 					tauL_val = 20,
	#					 					tauR_val = 2
	#					 				)

	model_type = mt.MFNMDA(alpha_val = 0.7,
					EL_val = -60,
					EA_val = 0,
					EN_val = 0,
					EI_val = -80,
					nuA_val = 0.020,
					nuN_val = 0.020,
					nuI_val = 0.020,
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
					tauR_val = 2,
					beta_val = 0.062,
					gamma_val = 3.57,
					n_Mg_val = 1
				)

	model = LangevinModel(model_type)
	integrator = FokkerPlanckModel(model, True)

	vec_Vk = np.linspace(-80, -50, 30000 + 1)
	f = model_type.data_parameters.transform_expressions_to_functions([integrator.vec_S[1]])
	print(f[0](-10))

	# sp.pprint(sp.simplify(expressions_list[2]))
	

if __name__ == "__main__":
	main()
