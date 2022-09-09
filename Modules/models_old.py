from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import sympy as sp

TNum = TypeVar('TNum', float, sp.core.expr.Expr)

class ModelName(Enum):
	COBAIF = auto()
	NMDA = auto()
	INTERPOLATED = auto()

class ModelType(ABC):
	def __init__(self):
		self.model_name: ModelName
		self.nu_i: float
		self.V_th: float = -50
		self.V_r: float = -60
		self.tau_L: float = 20
		self.tau_R: float = 2
		self.V: sp.core.expr.Expr = sp.symbols('V')

		self.nonlinear_term: TNum
		self.vec_w: npt.NDArray
		self.vec_tau: npt.NDArray
		self.vec_E: npt.NDArray
		self.vec_K: npt.NDArray
		self.modulating_functions: npt.NDArray

	@abstractmethod
	def _define_modulating_functions(self) -> npt.NDArray:
		pass


class MFCoBaIF(ModelType):
	def __init__(self, w_E: float,
					w_I: float,
					tau_E: float,
					tau_I: float,
					nu_i: float,
					E_E: float = 0,
					E_I: float = -80,
					E_L: float = -60,
					K_E: int = 400,
					K_I: int = 100
				):
		super().__init__()

		self.model_name = ModelName.COBAIF
		self.nu_i = nu_i
		self.nonlinear_term = 0
		self.vec_w = np.array([1, w_E, w_I])
		self.vec_tau = np.array([1, tau_E, tau_I])
		self.vec_E = np.array([E_L, E_E, E_I])
		self.vec_K = np.array([1, K_E, K_I])

		self.modulating_functions = self._define_modulating_functions()

	def _define_modulating_functions(self) -> npt.NDArray:
		return np.array([1., 1., 1.])


class MFInterpolatedCoBaIF(ModelType):
	def __init__(self, alpha:float, 
					w_F: float,
					w_S: float,
					w_I: float,
					tau_F: float,
					tau_S: float,
					tau_I: float,
					nu_i: float,
					E_F: float = 0,
					E_S: float = 0,
					E_I: float = -80,
					E_L: float = -60,
					K_F: int = 400,
					K_S: int = 400,
					K_I: int = 100
				):
		super().__init__()

		self.model_name = ModelName.INTERPOLATED
		self.alpha = alpha
		self.nu_i = nu_i
		self.nonlinear_term = 0
		self.vec_w = np.array([1, w_F, w_S, w_I])
		self.vec_tau = np.array([1, tau_F, tau_S, tau_I])
		self.vec_E = np.array([E_L, E_F, E_S, E_I])
		self.vec_K = np.array([1, K_F, K_S, K_I])

		self.modulating_functions = self._define_modulating_functions()

	def _define_modulating_functions(self) -> npt.NDArray:
		return np.array([1., (1 - self.alpha), self.alpha, 1.])


class MFNMDA(ModelType):
	def __init__(self, alpha:float, 
					w_A: float,
					w_N: float,
					w_I: float,
					tau_A: float,
					tau_N: float,
					tau_I: float,
					nu_i: float,
					E_A: float = 0,
					E_N: float = 0,
					E_I: float = -80,
					E_L: float = -60,
					K_A: int = 400,
					K_N: int = 400,
					K_I: int = 100
				):
		super().__init__()

		self.model_name = ModelName.NMDA
		self.alpha = alpha
		self.nu_i = nu_i
		self.nonlinear_term = 0
		self.vec_w = np.array([1, w_A, w_N, w_I])
		self.vec_tau = np.array([1, tau_A, tau_N, tau_I])
		self.vec_E = np.array([E_L, E_A, E_N, E_I])
		self.vec_K = np.array([1, K_A, K_N, K_I])

		self.beta = 0.062
		self.n_Mg = 1
		self.gamma = 3.57

		self.modulating_functions = self._define_modulating_functions()

	def _define_modulating_functions(self) -> npt.NDArray:
		return np.array([1., (1 - self.alpha), self.alpha*self._s(), 1.])

	def _s(self) -> TNum:
		beta = self.beta
		n_Mg = self.n_Mg
		gamma = self.gamma
		V = self.V

		return 1 / (1 + (n_Mg / gamma) * sp.exp(-beta * V))

def main():
	model = MFNMDA(alpha = 0.1,
					w_A = 0.1,
					w_N = 0.1,
					w_I = 0.4,
					tau_A = 5,
					tau_N = 100,
					tau_I = 10,
					nu_i = 0.005
					)
	print(model.modulating_functions)

if __name__ == "__main__":
	main()
