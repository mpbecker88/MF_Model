from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class _Model:
	w_E: float
	w_I: float
	tau_E: float
	tau_I: float
	nu_i: float
	V_th: float = -50
	V_r: float = -60
	E_E: float = 0
	E_I: float = -80
	E_L: float = -60
	K_E: int = 400
	K_I: int = 100
	tau_L: float = 20
	tau_R: float = 2

	mu_E: float = field(init=False)
	mu_I: float = field(init=False)
	sigma_E: float = field(init=False)
	sigma_I: float = field(init=False)
	mu: float = field(init=False)
	tau: float = field(init=False)


@dataclass(frozen=True, slots=True)
class CoBaIF:
	"""Mean-field model class for the conductance based integrate and fire neuron with
	the effective-timecostant approximation and the Fox approximation."""
	w_E: float
	w_I: float
	tau_E: float
	tau_I: float
	nu_i: float
	V_th: float = -50
	V_r: float = -60
	E_E: float = 0
	E_I: float = -80
	E_L: float = -60
	K_E: int = 400
	K_I: int = 100
	tau_L: float = 20
	tau_R: float = 2

	mu_E: float = field(init=False)
	mu_I: float = field(init=False)
	sigma_E: float = field(init=False)
	sigma_I: float = field(init=False)
	mu: float = field(init=False)
	tau: float = field(init=False)

	def __post_init__(self):
		object.__setattr__(self, "mu_E", self.w_E * self.K_E * self.nu_i * self.tau_E)
		object.__setattr__(self, "mu_I", self.w_I * self.K_I * self.nu_i * self.tau_I)
		object.__setattr__(self, "sigma_E", self.w_E * np.sqrt(self.K_E * self.nu_i * self.tau_E))
		object.__setattr__(self, "sigma_I",	self.w_I * np.sqrt(self.K_I * self.nu_i * self.tau_I))
		object.__setattr__(self, "tau", self.tau_L / (1 + self.mu_E + self.mu_I))
		object.__setattr__(self, "mu", self.tau * (self.E_L + self.mu_E * self.E_E + self.mu_I * self.E_I) / self.tau_L)

	@property
	def h_E(self) -> float:
		"""Return the value of the function h_E(V)"""

		return (np.sqrt(self.tau_E * self.tau) / self.tau_L) * self.sigma_E * (self.E_E - self.mu)

	@property
	def h_I(self) -> float:
		"""Return the value of the function h_I(V)"""

		return (np.sqrt(self.tau_I * self.tau) / self.tau_L) * self.sigma_I * (self.E_I - self.mu)

	def calculate_sigma(self) -> float:
		"""Return the noise standard deviation"""
		return np.sqrt(((self.tau * self.h_E**2) / (self.tau + self.tau_E)) + ((self.tau * self.h_I**2) / (self.tau + self.tau_I)))

	def _S_0(self, V: float) -> float:
		"""Return the value of the function S_0(V)"""

		return (V - self.mu) / self.tau

	def _B(self, V: float) -> float:
		"""Return the value of the linear coefficient B(V)"""

		return 2 * self.tau * self._S_0(V) / self.calculate_sigma()**2

	def _H(self, V: float) -> float:
		"""Return the value of the heterogeneity of the Fokker-Planck H(V)"""

		if V > self.V_r:
			return 2* self.tau / (self.calculate_sigma()**2)
		if V == self.V_r:
			return self.tau / (self.calculate_sigma()**2)
		return 0

	def integrate_p0(self, vec_Vk: npt.NDArray) -> npt.NDArray:
		"""Integrate the unnormalized probability distribution p0"""

		vec_p0_fliped = np.zeros_like(vec_Vk)
		dV = vec_Vk[1] - vec_Vk[0]
		vec_Vk_fliped = np.flip(vec_Vk)

		vec_p0_fliped[0] = 0

		for j, (Vk, p0) in enumerate(zip(vec_Vk_fliped[:-1], vec_p0_fliped[:-1])):
			Bk = self._B(Vk)
			Hk = self._H(Vk)
			if(-0.000001 <= Bk <= 0.000001):
				vec_p0_fliped[j+1] = p0 * np.exp(dV * Bk) + dV * Hk
			else:
				vec_p0_fliped[j+1] = p0 * np.exp(dV * Bk) + dV * Hk * (np.exp(dV*Bk) - 1) / (dV * Bk)

		vec_p0 = np.flip(vec_p0_fliped)

		return vec_p0

	def calculate_firing_rate(self, vec_Vk: npt.NDArray) -> float:
		vec_p0 = self.integrate_p0(vec_Vk)
		dV = vec_Vk[1] - vec_Vk[0]

		return 1 / (self.tau_R + dV * np.sum(vec_p0))


@dataclass(frozen=True, slots=True)
class ICoBaIF:
	"""Mean-field model class for the conductance based integrate and fire neuron with
	the effective-timecostant approximation and the Fox approximation."""
	alpha: float
	w_E: float
	w_I: float
	tau_F: float
	tau_S: float
	tau_I: float
	nu_i: float
	V_th: float = -50
	V_r: float = -60
	E_E: float = 0
	E_I: float = -80
	E_L: float = -60
	K_E: int = 400
	K_I: int = 100
	tau_L: float = 20
	tau_R: float = 2

	mu_F: float = field(init=False)
	mu_S: float = field(init=False)
	mu_I: float = field(init=False)
	sigma_F: float = field(init=False)
	sigma_S: float = field(init=False)
	sigma_I: float = field(init=False)
	mu: float = field(init=False)
	tau: float = field(init=False)

	def __post_init__(self):
		object.__setattr__(self, "mu_F", self.w_E * self.K_E * self.nu_i * self.tau_F)
		object.__setattr__(self, "mu_S", self.w_E * self.K_E * self.nu_i * self.tau_S)
		object.__setattr__(self, "mu_I", self.w_I * self.K_I * self.nu_i * self.tau_I)
		object.__setattr__(self, "sigma_F", self.w_E * np.sqrt(self.K_E * self.nu_i * self.tau_F))
		object.__setattr__(self, "sigma_S", self.w_E * np.sqrt(self.K_E * self.nu_i * self.tau_S))
		object.__setattr__(self, "sigma_I",	self.w_I * np.sqrt(self.K_I * self.nu_i * self.tau_I))
		object.__setattr__(self, "tau", self.tau_L / (1 + (1 - self.alpha) * self.mu_F + self.alpha * self.mu_S + self.mu_I))
		object.__setattr__(self, "mu", self.tau * (self.E_L + ((1 - self.alpha) * self.mu_F + self.alpha * self.mu_S) * self.E_E + self.mu_I * self.E_I) / self.tau_L)

	@property
	def h_F(self) -> float:
		"""Return the value of the function h_E(V)"""

		return ((1 - self.alpha) * np.sqrt(self.tau_F * self.tau) / self.tau_L) * self.sigma_F * (self.E_E - self.mu)

	@property
	def h_S(self) -> float:
		"""Return the value of the function h_E(V)"""

		return (self.alpha * np.sqrt(self.tau_S * self.tau) / self.tau_L) * self.sigma_S * (self.E_E - self.mu)

	@property
	def h_I(self) -> float:
		"""Return the value of the function h_I(V)"""

		return (np.sqrt(self.tau_I * self.tau) / self.tau_L) * self.sigma_I * (self.E_I - self.mu)

	def calculate_sigma(self) -> float:
		"""Return the noise standard deviation"""
		return np.sqrt(((self.tau * self.h_F**2) / (self.tau + self.tau_S)) + ((self.tau * self.h_S**2) / (self.tau + self.tau_S)) + ((self.tau * self.h_I**2) / (self.tau + self.tau_I)))

	def _S_0(self, V: float) -> float:
		"""Return the value of the function S_0(V)"""

		return (V - self.mu) / self.tau

	def _B(self, V: float) -> float:
		"""Return the value of the linear coefficient B(V)"""

		return 2 * self.tau * self._S_0(V) / self.calculate_sigma()**2

	def _H(self, V: float) -> float:
		"""Return the value of the heterogeneity of the Fokker-Planck H(V)"""

		if V > self.V_r:
			return 2* self.tau / (self.calculate_sigma()**2)
		if V == self.V_r:
			return self.tau / (self.calculate_sigma()**2)
		return 0

	def integrate_p0(self, vec_Vk: npt.NDArray) -> npt.NDArray:
		"""Integrate the unnormalized probability distribution p0"""

		vec_p0_fliped = np.zeros_like(vec_Vk)
		dV = vec_Vk[1] - vec_Vk[0]
		vec_Vk_fliped = np.flip(vec_Vk)

		vec_p0_fliped[0] = 0

		for j, (Vk, p0) in enumerate(zip(vec_Vk_fliped[:-1], vec_p0_fliped[:-1])):
			Bk = self._B(Vk)
			Hk = self._H(Vk)
			if(-0.000001 <= Bk <= 0.000001):
				vec_p0_fliped[j+1] = p0 * np.exp(dV * Bk) + dV * Hk
			else:
				vec_p0_fliped[j+1] = p0 * np.exp(dV * Bk) + dV * Hk * (np.exp(dV*Bk) - 1) / (dV * Bk)

		vec_p0 = np.flip(vec_p0_fliped)

		return vec_p0

	def calculate_firing_rate(self, vec_Vk: npt.NDArray) -> float:
		vec_p0 = self.integrate_p0(vec_Vk)
		dV = vec_Vk[1] - vec_Vk[0]

		return 1 / (self.tau_R + dV * np.sum(vec_p0))


@dataclass(frozen=True, slots=True)
class MCoBaIF:
	"""Mean-field model class for the full multiplicative conductance based integrate and fire neuron
	with the Fox approximation."""
	w_E: float
	w_I: float
	tau_E: float
	tau_I: float
	nu_i: float
	V_th: float = -50
	V_r: float = -60
	E_E: float = 0
	E_I: float = -80
	E_L: float = -60
	K_E: int = 400
	K_I: int = 100
	tau_L: float = 20
	tau_R: float = 2

	mu_E: float = field(init=False)
	mu_I: float = field(init=False)
	sigma_E: float = field(init=False)
	sigma_I: float = field(init=False)
	mu: float = field(init=False)
	tau: float = field(init=False)

	def __post_init__(self):
		object.__setattr__(self, "mu_E", self.w_E * self.K_E * self.nu_i * self.tau_E)
		object.__setattr__(self, "mu_I", self.w_I * self.K_I * self.nu_i * self.tau_I)
		object.__setattr__(self, "sigma_E", self.w_E * np.sqrt(self.K_E * self.nu_i * self.tau_E))
		object.__setattr__(self, "sigma_I",	self.w_I * np.sqrt(self.K_I * self.nu_i * self.tau_I))
		object.__setattr__(self, "tau", self.tau_L / (1 + self.mu_E + self.mu_I))
		object.__setattr__(self, "mu", self.tau * (self.E_L + self.mu_E * self.E_E + self.mu_I * self.E_I) / self.tau_L)

	def _h_E(self, V: float) -> float:
		"""Return the value of the function h_E(V)"""

		return (np.sqrt(self.tau_E * self.tau) / self.tau_L) * self.sigma_E * (self.E_E - V)

	def _h_I(self, V: float) -> float:
		"""Return the value of the function h_I(V)"""

		return (np.sqrt(self.tau_I * self.tau) / self.tau_L) * self.sigma_I * (self.E_I - V)

	def _dh_E(self, V: float) -> float:
		"""Return the value of derivative of h_E(V)"""

		return -(np.sqrt(self.tau_E * self.tau) / self.tau_L) * self.sigma_E

	def _dh_I(self, V: float) -> float:
		"""Return the value of derivative of h_I(V)"""

		return -(np.sqrt(self.tau_I * self.tau) / self.tau_L) * self.sigma_I

	def _S_0(self, V: float) -> float:
		"""Return the value of the function S_0(V)"""

		return (V - self.mu) / self.tau

	def _S_E(self, V: float) -> float:
		"""Return the value of the function S_E(V)"""

		return self._h_E(V) / (2 * (self.tau + self.tau_E * (1 - (self._dh_E(V) / self._h_E(V)) * (V - self.mu))))

	def _S_I(self, V: float) -> float:
		"""Return the value of the function S_I(V)"""
		return self._h_I(V) / (2 * (self.tau + self.tau_I * (1 - (self._dh_I(V) / self._h_I(V)) * (V - self.mu))))

	def _dS_E(self, V: float) -> float:
		"""Return the value of the derivative of S_E(V)"""

		tau = self.tau
		tau_E = self.tau_E
		tau_L = self.tau_L
		E_E = self.E_E
		mu = self.mu
		sigma_E = self.sigma_E

		numerator = (-sigma_E * np.sqrt(tau * tau_E) * (E_E - V) * (tau * (E_E - V) + 2 * tau_E * (E_E - mu)))
		denominator = 2 * tau_L * (tau * (E_E - V) + tau_E * (E_E - mu))**2

		return numerator / denominator

	def _dS_I(self, V: float) -> float:
		"""Return the value of the derivative of S_I(V)"""

		tau = self.tau
		tau_I = self.tau_I
		tau_L = self.tau_L
		E_I = self.E_I
		mu = self.mu
		sigma_I = self.sigma_I

		numerator = (-sigma_I * np.sqrt(tau * tau_I) * (E_I - V) * (tau * (E_I - V) + 2 * tau_I * (E_I - mu)))
		denominator = 2 * tau_L * (tau * (E_I - V) + tau_I * (E_I - mu))**2

		return numerator / denominator

	def _Xi(self, V: float):
		"""Return the value of the function Xi(V)"""

		return self._h_E(V) * self._S_E(V) + self._h_I(V) * self._S_I(V)

	def _B(self, V: float) -> float:
		"""Return the value of the linear coefficient B(V)"""

		return ((self._S_0(V) 
				+ self._h_E(V) * self._dS_E(V)
				+ self._h_I(V) * self._dS_I(V))
				/ self._Xi(V))

	def _H(self, V: float) -> float:
		"""Return the value of the heterogeneity of the Fokker-Planck H(V)"""

		if V > self.V_r:
			return 1 / self._Xi(V)
		if V == self.V_r:
			return 1 / (2 * self._Xi(V))
		return 0

	def integrate_p0(self, vec_Vk: npt.NDArray) -> npt.NDArray:
		"""Integrate the unnormalized probability distribution p0"""

		vec_p0_fliped = np.zeros_like(vec_Vk)
		dV = vec_Vk[1] - vec_Vk[0]
		vec_Vk_fliped = np.flip(vec_Vk)

		vec_p0_fliped[0] = 0

		for j, (Vk, p0) in enumerate(zip(vec_Vk_fliped[:-1], vec_p0_fliped[:-1])):
			Bk = self._B(Vk)
			Hk = self._H(Vk)
			if(-0.000001 <= Bk <= 0.000001):
				vec_p0_fliped[j+1] = p0 * np.exp(dV * Bk) + dV * Hk
			else:
				vec_p0_fliped[j+1] = p0 * np.exp(dV * Bk) + dV * Hk * (np.exp(dV*Bk) - 1) / (dV * Bk)

		vec_p0 = np.flip(vec_p0_fliped)

		return vec_p0

	def calculate_firing_rate(self, vec_Vk: npt.NDArray) -> float:
		vec_p0 = self.integrate_p0(vec_Vk)
		dV = vec_Vk[1] - vec_Vk[0]

		return 1 / (self.tau_R + dV * np.sum(vec_p0))


@dataclass(frozen=True, slots=True)
class MICoBaIF:
	"""Mean-field model class for the full multiplicative conductance based integrate and fire neuron
	with the Fox approximation."""
	alpha: float
	w_E: float
	w_I: float
	tau_F: float
	tau_S: float
	tau_I: float
	nu_i: float
	V_th: float = -50
	V_r: float = -60
	E_E: float = 0
	E_I: float = -80
	E_L: float = -60
	K_E: int = 400
	K_I: int = 100
	tau_L: float = 20
	tau_R: float = 2

	mu_F: float = field(init=False)
	mu_S: float = field(init=False)
	mu_I: float = field(init=False)
	sigma_F: float = field(init=False)
	sigma_S: float = field(init=False)
	sigma_I: float = field(init=False)
	mu: float = field(init=False)
	tau: float = field(init=False)

	def __post_init__(self):
		object.__setattr__(self, "mu_F", self.w_E * self.K_E * self.nu_i * self.tau_F)
		object.__setattr__(self, "mu_S", self.w_E * self.K_E * self.nu_i * self.tau_S)
		object.__setattr__(self, "mu_I", self.w_I * self.K_I * self.nu_i * self.tau_I)
		object.__setattr__(self, "sigma_F", self.w_E * np.sqrt(self.K_E * self.nu_i * self.tau_F))
		object.__setattr__(self, "sigma_S", self.w_E * np.sqrt(self.K_E * self.nu_i * self.tau_S))
		object.__setattr__(self, "sigma_I",	self.w_I * np.sqrt(self.K_I * self.nu_i * self.tau_I))
		object.__setattr__(self, "tau", self.tau_L / (1 + (1 - self.alpha) * self.mu_F + self.alpha * self.mu_S + self.mu_I))
		object.__setattr__(self, "mu", self.tau * (self.E_L + ((1 - self.alpha) * self.mu_F + self.alpha * self.mu_S) * self.E_E + self.mu_I * self.E_I) / self.tau_L)

	def _h_F(self, V: float) -> float:
		"""Return the value of the function h_E(V)"""

		return ((1 - self.alpha) * np.sqrt(self.tau_F * self.tau) / self.tau_L) * self.sigma_F * (self.E_E - V)

	def _h_S(self, V: float) -> float:
		"""Return the value of the function h_E(V)"""

		return (self.alpha * np.sqrt(self.tau_S * self.tau) / self.tau_L) * self.sigma_S * (self.E_E - V)

	def _h_I(self, V: float) -> float:
		"""Return the value of the function h_I(V)"""

		return (np.sqrt(self.tau_I * self.tau) / self.tau_L) * self.sigma_I * (self.E_I - V)

	def _dh_F(self, V: float) -> float:
		"""Return the value of derivative of h_E(V)"""

		return -((1 - self.alpha) * np.sqrt(self.tau_F * self.tau) / self.tau_L) * self.sigma_F

	def _dh_S(self, V: float) -> float:
		"""Return the value of derivative of h_E(V)"""

		return -(self.alpha * np.sqrt(self.tau_S * self.tau) / self.tau_L) * self.sigma_S

	def _dh_I(self, V: float) -> float:
		"""Return the value of derivative of h_I(V)"""

		return -(np.sqrt(self.tau_I * self.tau) / self.tau_L) * self.sigma_I

	def _S_0(self, V: float) -> float:
		"""Return the value of the function S_0(V)"""

		return (V - self.mu) / self.tau

	def _S_F(self, V: float) -> float:
		"""Return the value of the function S_E(V)"""

		return self._h_F(V) / (2 * (self.tau + self.tau_F * (1 - (self._dh_F(V) / self._h_F(V)) * (V - self.mu))))

	def _S_S(self, V: float) -> float:
		"""Return the value of the function S_E(V)"""

		return self._h_S(V) / (2 * (self.tau + self.tau_F * (1 - (self._dh_S(V) / self._h_S(V)) * (V - self.mu))))

	def _S_I(self, V: float) -> float:
		"""Return the value of the function S_I(V)"""
		return self._h_I(V) / (2 * (self.tau + self.tau_I * (1 - (self._dh_I(V) / self._h_I(V)) * (V - self.mu))))

	def _dS_F(self, V: float) -> float:
		"""Return the value of the derivative of S_E(V)"""

		alpha = self.alpha
		tau = self.tau
		tau_F = self.tau_F
		tau_L = self.tau_L
		E_E = self.E_E
		mu = self.mu
		sigma_F = self.sigma_F

		numerator = ((alpha - 1) * sigma_F * np.sqrt(tau * tau_F) * (E_E - V) * (tau * (E_E - V) + 2 * tau_F * (E_E - mu)))
		denominator = 2 * tau_L * (tau * (E_E - V) + tau_F * (E_E - mu))**2

		return numerator / denominator

	def _dS_S(self, V: float) -> float:
		"""Return the value of the derivative of S_E(V)"""

		alpha = self.alpha
		tau = self.tau
		tau_S = self.tau_S
		tau_L = self.tau_L
		E_E = self.E_E
		mu = self.mu
		sigma_S = self.sigma_S

		numerator = (-alpha * sigma_S * np.sqrt(tau * tau_S) * (E_E - V) * (tau * (E_E - V) + 2 * tau_S * (E_E - mu)))
		denominator = 2 * tau_L * (tau * (E_E - V) + tau_S * (E_E - mu))**2

		return numerator / denominator

	def _dS_I(self, V: float) -> float:
		"""Return the value of the derivative of S_I(V)"""

		tau = self.tau
		tau_I = self.tau_I
		tau_L = self.tau_L
		E_I = self.E_I
		mu = self.mu
		sigma_I = self.sigma_I

		numerator = (-sigma_I * np.sqrt(tau * tau_I) * (E_I - V) * (tau * (E_I - V) + 2 * tau_I * (E_I - mu)))
		denominator = 2 * tau_L * (tau * (E_I - V) + tau_I * (E_I - mu))**2

		return numerator / denominator

	def _Xi(self, V: float):
		"""Return the value of the function Xi(V)"""

		return self._h_F(V) * self._S_F(V) + self._h_S(V) * self._S_S(V) + self._h_I(V) * self._S_I(V)

	def _B(self, V: float) -> float:
		"""Return the value of the linear coefficient B(V)"""

		return ((self._S_0(V) 
				+ self._h_F(V) * self._dS_F(V)
				+ self._h_S(V) * self._dS_S(V)
				+ self._h_I(V) * self._dS_I(V))
				/ self._Xi(V))

	def _H(self, V: float) -> float:
		"""Return the value of the heterogeneity of the Fokker-Planck H(V)"""

		if V > self.V_r:
			return 1 / self._Xi(V)
		if V == self.V_r:
			return 1 / (2 * self._Xi(V))
		return 0

	def integrate_p0(self, vec_Vk: npt.NDArray) -> npt.NDArray:
		"""Integrate the unnormalized probability distribution p0"""

		vec_p0_fliped = np.zeros_like(vec_Vk)
		dV = vec_Vk[1] - vec_Vk[0]
		vec_Vk_fliped = np.flip(vec_Vk)

		vec_p0_fliped[0] = 0

		for j, (Vk, p0) in enumerate(zip(vec_Vk_fliped[:-1], vec_p0_fliped[:-1])):
			Bk = self._B(Vk)
			Hk = self._H(Vk)
			if(-0.000001 <= Bk <= 0.000001):
				vec_p0_fliped[j+1] = p0 * np.exp(dV * Bk) + dV * Hk
			else:
				vec_p0_fliped[j+1] = p0 * np.exp(dV * Bk) + dV * Hk * (np.exp(dV*Bk) - 1) / (dV * Bk)

		vec_p0 = np.flip(vec_p0_fliped)

		return vec_p0

	def calculate_firing_rate(self, vec_Vk: npt.NDArray) -> float:
		vec_p0 = self.integrate_p0(vec_Vk)
		dV = vec_Vk[1] - vec_Vk[0]

		return 1 / (self.tau_R + dV * np.sum(vec_p0))


@dataclass(frozen=True, slots=True)
class NMDACoBaIF:
	"""Mean-field model class for the full multiplicative conductance based integrate and fire neuron
	with the Fox approximation."""

	alpha: float
	w_E: float
	w_I: float
	tau_A: float
	tau_N: float
	tau_I: float
	nu_i: float
	V_th: float = -50
	V_r: float = -60
	E_E: float = 0
	E_I: float = -80
	E_L: float = -60
	K_E: int = 400
	K_I: int = 100
	tau_L: float = 20
	tau_R: float = 2
	beta: float = 0.062
	n_Mg: float = 1
	gamma: float = 3.57

	mu_A: float = field(init=False)
	mu_N: float = field(init=False)
	mu_I: float = field(init=False)
	sigma_A: float = field(init=False)
	sigma_N: float = field(init=False)
	sigma_I: float = field(init=False)

	def __post_init__(self):
		object.__setattr__(self, "mu_A", self.w_E * self.K_E * self.nu_i * self.tau_A)
		object.__setattr__(self, "mu_N", self.w_E * self.K_E * self.nu_i * self.tau_N)
		object.__setattr__(self, "mu_I", self.w_I * self.K_I * self.nu_i * self.tau_I)
		object.__setattr__(self, "sigma_A", self.w_E * np.sqrt(self.K_E * self.nu_i * self.tau_A))
		object.__setattr__(self, "sigma_N", self.w_E * np.sqrt(self.K_E * self.nu_i * self.tau_N))
		object.__setattr__(self, "sigma_I",	self.w_I * np.sqrt(self.K_I * self.nu_i * self.tau_I))
		object.__setattr__(self, "tau", self.tau_L / (1 + (1 - self.alpha) * self.mu_A + self.alpha * self.mu_N + self.mu_I))
		object.__setattr__(self, "mu", self.tau * (self.E_L + ((1 - self.alpha) * self.mu_A + self.alpha * self.mu_N) * self.E_E + self.mu_I * self.E_I) / self.tau_L)

	def _s(self, V: float) -> float:
		return 1 / (1 + (self.n_Mg / self.gamma) * np.exp(-self.beta * V))

	def _tau(self, V: float) -> float:
		return  self.tau_L / (1 + (1 - self.alpha) * self.mu_A + self.alpha * self._s(V) * self.mu_N + self.mu_I)

	def _mu(self, V: float) -> float:
		return self._tau(V) * (self.E_L + ((1 - self.alpha) * self.mu_A + self.alpha * self._s(V) * self.mu_N) * self.E_E + self.mu_I * self.E_I) / self.tau_L

	def _h_A(self, V: float) -> float:
		"""Return the value of the function h_E(V)"""

		return ((1 - self.alpha) * np.sqrt(self.tau_A * self._tau(V)) / self.tau_L) * self.sigma_A * (self.E_E - V)

	def _h_N(self, V: float) -> float:
		"""Return the value of the function h_E(V)"""

		return (self.alpha * np.sqrt(self.tau_N * self._tau(V)) / self.tau_L) * self._s(V) * self.sigma_N * (self.E_E - V)

	def _h_I(self, V: float) -> float:
		"""Return the value of the function h_I(V)"""

		return (np.sqrt(self.tau_I * self._tau(V)) / self.tau_L) * self.sigma_I * (self.E_I - V)

	def _dh_A(self, V: float) -> float:
		"""Return the value of derivative of h_E(V)"""
		alpha = self.alpha
		

		return -((1 - self.alpha) * np.sqrt(self.tau_F * self.tau) / self.tau_L) * self.sigma_F

	def _dh_S(self, V: float) -> float:
		"""Return the value of derivative of h_E(V)"""

		return -(self.alpha * np.sqrt(self.tau_S * self.tau) / self.tau_L) * self.sigma_S

	def _dh_I(self, V: float) -> float:
		"""Return the value of derivative of h_I(V)"""

		return -(np.sqrt(self.tau_I * self.tau) / self.tau_L) * self.sigma_I

	def _S_0(self, V: float) -> float:
		"""Return the value of the function S_0(V)"""

		return (V - self.mu) / self.tau

	def _S_F(self, V: float) -> float:
		"""Return the value of the function S_E(V)"""

		return self._h_F(V) / (2 * (self.tau + self.tau_F * (1 - (self._dh_F(V) / self._h_F(V)) * (V - self.mu))))

	def _S_S(self, V: float) -> float:
		"""Return the value of the function S_E(V)"""

		return self._h_S(V) / (2 * (self.tau + self.tau_F * (1 - (self._dh_S(V) / self._h_S(V)) * (V - self.mu))))

	def _S_I(self, V: float) -> float:
		"""Return the value of the function S_I(V)"""
		return self._h_I(V) / (2 * (self.tau + self.tau_I * (1 - (self._dh_I(V) / self._h_I(V)) * (V - self.mu))))

	def _dS_F(self, V: float) -> float:
		"""Return the value of the derivative of S_E(V)"""

		alpha = self.alpha
		tau = self.tau
		tau_F = self.tau_F
		tau_L = self.tau_L
		E_E = self.E_E
		mu = self.mu
		sigma_F = self.sigma_F

		numerator = ((alpha - 1) * sigma_F * np.sqrt(tau * tau_F) * (E_E - V) * (tau * (E_E - V) + 2 * tau_F * (E_E - mu)))
		denominator = 2 * tau_L * (tau * (E_E - V) + tau_F * (E_E - mu))**2

		return numerator / denominator

	def _dS_S(self, V: float) -> float:
		"""Return the value of the derivative of S_E(V)"""

		alpha = self.alpha
		tau = self.tau
		tau_S = self.tau_S
		tau_L = self.tau_L
		E_E = self.E_E
		mu = self.mu
		sigma_S = self.sigma_S

		numerator = (-alpha * sigma_S * np.sqrt(tau * tau_S) * (E_E - V) * (tau * (E_E - V) + 2 * tau_S * (E_E - mu)))
		denominator = 2 * tau_L * (tau * (E_E - V) + tau_S * (E_E - mu))**2

		return numerator / denominator

	def _dS_I(self, V: float) -> float:
		"""Return the value of the derivative of S_I(V)"""

		tau = self.tau
		tau_I = self.tau_I
		tau_L = self.tau_L
		E_I = self.E_I
		mu = self.mu
		sigma_I = self.sigma_I

		numerator = (-sigma_I * np.sqrt(tau * tau_I) * (E_I - V) * (tau * (E_I - V) + 2 * tau_I * (E_I - mu)))
		denominator = 2 * tau_L * (tau * (E_I - V) + tau_I * (E_I - mu))**2

		return numerator / denominator

	def _Xi(self, V: float):
		"""Return the value of the function Xi(V)"""

		return self._h_F(V) * self._S_F(V) + self._h_S(V) * self._S_S(V) + self._h_I(V) * self._S_I(V)

	def _B(self, V: float) -> float:
		"""Return the value of the linear coefficient B(V)"""

		return ((self._S_0(V) 
				+ self._h_F(V) * self._dS_F(V)
				+ self._h_S(V) * self._dS_S(V)
				+ self._h_I(V) * self._dS_I(V))
				/ self._Xi(V))

	def _H(self, V: float) -> float:
		"""Return the value of the heterogeneity of the Fokker-Planck H(V)"""

		if V > self.V_r:
			return 1 / self._Xi(V)
		if V == self.V_r:
			return 1 / (2 * self._Xi(V))
		return 0

	def integrate_p0(self, vec_Vk: npt.NDArray) -> npt.NDArray:
		"""Integrate the unnormalized probability distribution p0"""

		vec_p0_fliped = np.zeros_like(vec_Vk)
		dV = vec_Vk[1] - vec_Vk[0]
		vec_Vk_fliped = np.flip(vec_Vk)

		vec_p0_fliped[0] = 0

		for j, (Vk, p0) in enumerate(zip(vec_Vk_fliped[:-1], vec_p0_fliped[:-1])):
			Bk = self._B(Vk)
			Hk = self._H(Vk)
			if(-0.000001 <= Bk <= 0.000001):
				vec_p0_fliped[j+1] = p0 * np.exp(dV * Bk) + dV * Hk
			else:
				vec_p0_fliped[j+1] = p0 * np.exp(dV * Bk) + dV * Hk * (np.exp(dV*Bk) - 1) / (dV * Bk)

		vec_p0 = np.flip(vec_p0_fliped)

		return vec_p0

	def calculate_firing_rate(self, vec_Vk: npt.NDArray) -> float:
		vec_p0 = self.integrate_p0(vec_Vk)
		dV = vec_Vk[1] - vec_Vk[0]

		return 1 / (self.tau_R + dV * np.sum(vec_p0))


def main():
	model = MCoBaIF(
		w_E = 0.1, 
		w_I = 0.4,
		tau_E = 5,
		tau_I = 10,
		nu_i = 0.005
	)

	E_I = -80.0
	V_th = -50.0 
	n = 30000
	vec_Vk = np.linspace(E_I, V_th, n + 1)
	print(model.calculate_firing_rate(vec_Vk))


if __name__ == "__main__":
	main()
