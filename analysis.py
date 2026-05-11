from cube import Cube
from cube_solver import Solver

if __name__ == "__main__":
    cube = Cube()
    solver = Solver()
    tot = 0
    for i in range(100):
        scramble = cube.random_scramble()
        solution = solver.solve(cube)
        n = len(solution.split())
        tot += n
            
    tot /= 100
    print("Average solution length over 100 random scrambles:", tot)
        