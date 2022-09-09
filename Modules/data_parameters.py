from pickle import FALSE
from typing import Callable, Dict, List, Union

import sympy as sp


class DataParameters:
	def __init__(self):
		self.values: Dict[sp.Symbol, float] = {}
		self.is_variable: Dict[sp.Symbol, bool] = {}
		self.variables_list: List[sp.Symbol] = []

	def add_value(self, symbol: sp.Symbol, value: float) -> None:
		self.values[symbol] = value
		self.is_variable[symbol] = False

	def add_variable(self, symbol: sp.Symbol) -> None:
		self.values[symbol] = 0
		self.is_variable[symbol] = True
		self.variables_list.append(symbol)

	def _substitute_values(self, expressions_list: List[sp.core.expr.Expr], symbol: sp.Symbol, value: float) -> List[sp.core.expr.Expr]:
		for i in range(len(expressions_list)):
			expressions_list[i] = expressions_list[i].subs(symbol, value)

		return expressions_list

	def transform_expressions_to_functions(self, expressions_list: List[sp.core.expr.Expr]) -> List[Callable]:
		functions_list = []

		expressions_list = self.substitute_parameters_values(expressions_list)

		for i in range(len(expressions_list)):
			functions_list.append(sp.lambdify(self.variables_list, expressions_list[i], 'numpy'))

		return functions_list

	def substitute_parameters_values(self, expressions_list: List[sp.core.expr.Expr]) -> List[sp.core.expr.Expr]:
		for symbol in self.values:
			if(not self.is_variable[symbol]):
				expressions_list = self._substitute_values(expressions_list, symbol, self.values[symbol])

		return expressions_list

def main():
	x = sp.symbols('x')
	y = sp.symbols('y')
	data_parameters = DataParameters()

	data_parameters.add_value(x, 1)
	data_parameters.add_variable(y)


	print(data_parameters.values)
	print(data_parameters.is_variable)

	expressions_list = [x**2, x*y, y**2]

	functions_list = data_parameters.transform_expressions_to_functions(expressions_list)
	print(functions_list[2](2))


if __name__ == "__main__":
	main()
