# Numerical Optimization

[![Build Status](https://github.com/jieli-matrix/NumericalOptimization/actions/workflows/python-app.yml/badge.svg)](https://github.com/jieli-matrix/NumericalOptimization/actions)
The repository collects numerical experiments codes for SCMS637001数值优化(graduate course) in Fudan University.

## Install

``` shell
git clone https://github.com/jieli-matrix/NumericalOptimization.git
pip install virtualenv
# python version newer than 3.8 would be better
virtualenv venv
source venv/bin/activate
pip install -r requirements
python -m unittest # test for all cases
```

## Line Search

Line search tells about how to choose a proper stepsize when given the descent direction in unconstrained gradient
descent methods. I implement `exact_line_search`, `constant_step_search` and `armijo_line_search`. 
`exact_line_search` is implemented specially for homework, but you could use `constant_step_search`
and `armijo_line_search` in general cases.

### Usage

``` python
>>> from optalgs.line_search import exact_line_search, constant_step_search, armijo_line_search
>>> import numpy as np
>>> x_0 = np.array([2,1])
>>> alpha = 0.1
>>> def obj_f(x) :
. . . return (x[0]**2 + 2*x[1]**2 )
. . .
>>> def obj_grad(x) :
. . . return [x[0], 2*x[1]]
. . .
>>> exact_line_search(obj_f, obj_grad, x_0)

>>> constant_step_search(obj_f, obj_grad, x_0, alpha)

>>> armijo_line_search(obj_f, obj_grad, x_0)
```
