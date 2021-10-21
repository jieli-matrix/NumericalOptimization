from optalgs.line_search import exact_line_search, constant_step_search, armijo_line_search
import unittest
import numpy as np

EPISILON = 1e-8
def obj_f(x):
    return (x[0]**2 + 2*x[1]**2)

def obj_grad(x):
    return [x[0], 2*x[1]]

class TestLineSearch(unittest.TestCase):
    def setUp(self):
        self.x_0 = np.array([2,1])
        self.alpha = 0.1
    
    def test_exact_line_search(self):
        _, _, _, func_val = exact_line_search(obj_f, obj_grad, self.x_0)
        self.assertLessEqual(abs(func_val), EPISILON)
    
    def test_constant_step_search(self):
        _, _, func_val = constant_step_search(obj_f, obj_grad, self.x_0, self.alpha)
        self.assertLessEqual(abs(func_val), EPISILON)
    
    def test_armijo_line_search(self):
        _, _, _, func_val = armijo_line_search(obj_f, obj_grad, self.x_0)
        self.assertLessEqual(abs(func_val), EPISILON)

if __name__ == "__main__":
    unittest.main()

