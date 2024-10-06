from decimal import Decimal
from typing import List, Tuple
import math

def dprint(func_name, msg, result=None):
    print(f"[DEBUG - {func_name}]: {msg}")
    if result is not None:
        print(f"    Result: {result}")

class NumArray(tuple):
    def __new__(cls, iterable):
        return super().__new__(cls, (Decimal(str(x)) for x in iterable))

    def inner_product(self, other: 'NumArray') -> Decimal:
        result = sum(a * b for a, b in zip(self, other))
        dprint("inner_product", f"Calculating {self} · {other}", result)
        return result

    def self_product(self) -> Decimal:
        result = self.inner_product(self)
        dprint("self_product", f"Calculating {self} · {self}", result)
        return result

    def scalar_multiply(self, scalar: Decimal) -> 'NumArray':
        result = NumArray(x * scalar for x in self)
        dprint("scalar_multiply", f"Multiplying {self} by {scalar}", result)
        return result

    def vector_subtract(self, other: 'NumArray') -> 'NumArray':
        result = NumArray(a - b for a, b in zip(self, other))
        dprint("vector_subtract", f"Subtracting {other} from {self}", result)
        return result

    def projection_factor(self, other: 'NumArray') -> Decimal:
        result = self.inner_product(other) / self.self_product()
        dprint("projection_factor", f"Computing projection factor of {other} onto {self}", result)
        return result

    def project_onto(self, other: 'NumArray') -> 'NumArray':
        factor = self.projection_factor(other)
        result = self.scalar_multiply(factor)
        dprint("project_onto", f"Projecting {other} onto {self}", result)
        return result

def orthogonalize(vectors: List[NumArray]) -> List[NumArray]:
    dprint("orthogonalize", f"Starting with vectors: {vectors}")
    orthogonalized = []
    for i, v in enumerate(vectors):
        dprint("orthogonalize", f"Processing vector {i}: {v}")
        for u in orthogonalized:
            v = v.vector_subtract(u.project_onto(v))
        if any(v):
            orthogonalized.append(v)
        dprint("orthogonalize", f"Orthogonalized vector {i}", v)
    dprint("orthogonalize", "Final orthogonalized set", orthogonalized)
    return orthogonalized

def lattice_reduce(basis: List[List[int]], threshold: float) -> List[List[int]]:
    dprint("lattice_reduce", f"Starting with basis: {basis}")
    n = len(basis)
    basis = [NumArray(v) for v in basis]
    orthogonal = orthogonalize(basis)

    def compute_coefficient(i: int, j: int) -> Decimal:
        result = orthogonal[j].projection_factor(basis[i])
        dprint("compute_coefficient", f"Computing coefficient for basis[{i}] and orthogonal[{j}]", result)
        return result

    idx = 1
    while idx < n:
        dprint("lattice_reduce", f"Processing index {idx}")
        for j in range(idx - 1, -1, -1):
            coeff = compute_coefficient(idx, j)
            if abs(coeff) > Decimal('0.5'):
                dprint("lattice_reduce", f"Adjusting basis vector {idx} using vector {j}")
                basis[idx] = basis[idx].vector_subtract(basis[j].scalar_multiply(round(coeff)))
                orthogonal = orthogonalize(basis)

        condition = orthogonal[idx].self_product() >= (Decimal(str(threshold)) - compute_coefficient(idx, idx-1)**2) * orthogonal[idx-1].self_product()
        dprint("lattice_reduce", f"Checking condition for index {idx}", condition)
        
        if condition:
            idx += 1
        else:
            dprint("lattice_reduce", f"Swapping vectors {idx} and {idx-1}")
            basis[idx], basis[idx-1] = basis[idx-1], basis[idx]
            orthogonal = orthogonalize(basis)
            idx = max(idx - 1, 1)

    result = [[int(x) for x in v] for v in basis]
    dprint("lattice_reduce", "Final reduced basis", result)
    return result

if __name__ == "__main__":
    dprint("main", "Starting main execution")
    dimension = int(input("Enter the dimension of the basis: "))
    matrix = []
    for i in range(dimension):
        vector = list(map(int, input(f"Enter basis vector {i+1}: ").strip().split()))
        matrix.append(vector[:dimension])
    angle = float(input("Enter the threshold angle: "))
    
    dprint("main", f"Input matrix: {matrix}")
    dprint("main", f"Threshold angle: {angle}")
    
    result = lattice_reduce(matrix, angle)
    print("LLL Reduced basis:", result)