from cube import Cube
from cube_solver import Solver

if __name__ == "__main__":
    cubes = []
    for _ in range(100):
        c = Cube()
        c.random_scramble()
        cubes.append(c)
    solver = Solver()
    total = 0
    for i, cube in enumerate(cubes):
        solution = solver.solve(cube)
        n_moves = len(solution.split())
        total += n_moves
        print(f"Cube {i+1}: {solution} ({n_moves} moves)")
    print(f"Total moves: {total}")