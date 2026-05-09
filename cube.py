class Cube:
    cube_map_index = """
              |  9 10 11 |                                                                                  
              | 12 13 14 |   <- U (face 2)                                                                    
              | 15 16 17 |                                                                                    
  | 18 19 20 |  0  1  2 | 27 28 29 | 45 46 47 |                                                               
  | 21 22 23 |  3  4  5 | 30 31 32 | 48 49 50 |                                                               
  | 24 25 26 |  6  7  8 | 33 34 35 | 51 52 53 |                                                               
     L (3)      F (1)      R (4)      B (6)                                                                   
              | 36 37 38 |                                                                                    
              | 39 40 41 |   <- D (face 5)                                                                    
              | 42 43 44 |  
    """
    
    solved_state = list("FFFFFFFFF" + "UUUUUUUUU" + "LLLLLLLLL" + "RRRRRRRRR" + "DDDDDDDDD" + "BBBBBBBBB")
    
    ALL_MOVES = None
    
    def __init__(self, state=None, verbose=False):
        # default to solved_state if no state provided
        self.state = state if state is not None else self.solved_state
        self.verbose = verbose
        self.table = Cube._build_move_table()
        
    def __str__(self):
        # prints the cube
        to_print = f"""
              |  {self.state[9]} {self.state[10]} {self.state[11]} |                                                                                  
              |  {self.state[12]} {self.state[13]} {self.state[14]} |   <- U (face 2)                                                                    
              |  {self.state[15]} {self.state[16]} {self.state[17]} |                                                                                    
  |  {self.state[18]} {self.state[19]} {self.state[20]} |  {self.state[0]} {self.state[1]} {self.state[2]} |  {self.state[27]} {self.state[28]} {self.state[29]} |  {self.state[45]} {self.state[46]} {self.state[47]} |                                                               
  |  {self.state[21]} {self.state[22]} {self.state[23]} |  {self.state[3]} {self.state[4]} {self.state[5]} |  {self.state[30]} {self.state[31]} {self.state[32]} |  {self.state[48]} {self.state[49]} {self.state[50]} |                                                               
  |  {self.state[24]} {self.state[25]} {self.state[26]} |  {self.state[6]} {self.state[7]} {self.state[8]} |  {self.state[33]} {self.state[34]} {self.state[35]} |  {self.state[51]} {self.state[52]} {self.state[53]} |
     L (3)      F (1)      R (4)      B (6)                                                                   
              |  {self.state[36]} {self.state[37]} {self.state[38]} |                                                                                    
              |  {self.state[39]} {self.state[40]} {self.state[41]} |   <- D (face 5)                                                                    
              |  {self.state[42]} {self.state[43]} {self.state[44]} |  
        """
        return to_print
    
    def get_solved_state(self):    
        return self.solved_state
    
    def get_state(self):
        return self.state

    def _cycle(self, perm, positions):                                                                                 
      # rotate cycle: each position pulls from the previous one                                             
      # positions = [a, b, c, d] means a→b→c→d→a (sticker movement)                                           
      # so new[b]=old[a], new[c]=old[b], new[d]=old[c], new[a]=old[d]                                         
        for i in range(len(positions)):                                                                         
            perm[positions[(i + 1) % len(positions)]] = positions[i] 
    
    @staticmethod
    def _compose(p1, p2):
        # Apply p1 first, then p2: new[i] = p1[p2[i]]
        return [p1[p2[i]] for i in range(54)]
    
    def U_PERM(self):
        "Rotate the top face clockwise"
        """for example first move FUL FUU FUR <- R0R1R2 <- B0B1B2- LLL <- FFF"""
        
        U_PERM = list(range(54))  # identity        
        # Trace each column of the strip independently — 3 separate 4-cycles:                                       
        self._cycle(U_PERM, [0,  18, 45, 27])                                                       
        self._cycle(U_PERM, [1,  19, 46, 28])                                                        
        self._cycle(U_PERM, [2,  20, 47, 29])    
                                                                      
        # U face's own stickers (face 2, offset 9)                                                                  
        self._cycle(U_PERM, [9, 11, 17, 15])      # corners                                                               
        self._cycle(U_PERM, [10, 14, 16, 12])
        return U_PERM
            
    def D_PERM(self):
        "Rotate the bottom face clockwise"
        D_PERM = list(range(54))  # identity
        # Trace each column of the strip independently — 3 separate 4-cycles:
        self._cycle(D_PERM, [6, 33, 51, 24])                                                       
        self._cycle(D_PERM, [7, 34, 52, 25])                                                        
        self._cycle(D_PERM, [8, 35, 53, 26])    
                                                                      
        # D face's own stickers (face 6, offset 45)                                                                  
        self._cycle(D_PERM, [36, 38, 44, 42])      # corners                                                               
        self._cycle(D_PERM, [37, 41, 43, 39])     # edges
        
        return D_PERM
    
    def F_PERM(self):
        "Rotate the front face clockwise"
        F_PERM = list(range(54))  # identity
        # Trace each column of the strip independently — 3 separate 4-cycles:
        self._cycle(F_PERM, [15, 27, 38, 26])                                                       
        self._cycle(F_PERM, [16, 30, 37, 23])                                                        
        self._cycle(F_PERM, [17, 33, 36, 20])    
                                                                      
        # F face's own stickers (face 1, offset 0)                                                                  
        self._cycle(F_PERM, [0, 2, 8, 6])      # corners                                                               
        self._cycle(F_PERM, [1, 5, 7, 3])     # edges
        
        return F_PERM
    
    def R_PERM(self):
        "Rotate the right face clockwise"
        R_PERM = list(range(54))  # identity
        # Trace each column of the strip independently — 3 separate 4-cycles:
        self._cycle(R_PERM, [2, 11, 51, 38])                                                       
        self._cycle(R_PERM, [5, 14, 48, 41])                                                        
        self._cycle(R_PERM, [8, 17, 45, 44])    
                                                                      
        # R face's own stickers (face 4, offset 27)                                                                  
        self._cycle(R_PERM, [27, 29, 35, 33])      # corners                                                               
        self._cycle(R_PERM, [28, 32, 34, 30])     # edges
        
        return R_PERM
    
    def L_PERM(self):
        "Rotate the left face clockwise"
        L_PERM = list(range(54))  # identity
        # Trace each column of the strip independently — 3 separate 4-cycles:
        self._cycle(L_PERM, [0, 36, 53, 9])                                                       
        self._cycle(L_PERM, [3, 39, 50, 12])                                                        
        self._cycle(L_PERM, [6, 42, 47, 15])    
                                                                      
        # L face's own stickers (face 3, offset 18)                                                                  
        self._cycle(L_PERM, [18, 20, 26, 24])      # corners                                                               
        self._cycle(L_PERM, [19, 23, 25, 21])     # edges
        
        return L_PERM
    
    def B_PERM(self):
        "Rotate the back face clockwise"
        B_PERM = list(range(54))  # identity
        # Trace each column of the strip independently — 3 separate 4-cycles:
        self._cycle(B_PERM, [9, 24, 44, 29])                                                       
        self._cycle(B_PERM, [10, 21, 43, 32])                                                        
        self._cycle(B_PERM, [11, 18, 42, 35])    
                                                                      
        # B face's own stickers (face 6, offset 45)                                                                  
        self._cycle(B_PERM, [45, 47, 53, 51])      # corners                                                               
        self._cycle(B_PERM, [46, 50, 52, 48])     # edges
        
        return B_PERM
    
    

    @classmethod
    def _build_move_table(cls):
        # Precompute all 18 face moves (U, U2, U', D, D2, D', ...) once.
        if cls.ALL_MOVES is not None:
            return cls.ALL_MOVES
        inst = cls.__new__(cls)
        base = {
            "U": inst.U_PERM(), "D": inst.D_PERM(),
            "L": inst.L_PERM(), "R": inst.R_PERM(),
            "F": inst.F_PERM(), "B": inst.B_PERM(),
        }
        moves = {}
        for name, p in base.items():
            moves[name]       = p
            moves[name + "2"] = cls._compose(p, p)
            moves[name + "'"] = cls._compose(p, cls._compose(p, p))
        cls.ALL_MOVES = moves
        return moves

    def random_scramble(self, length=20):
        import random
        moves = list(self.table.keys())
        scramble = " ".join(random.choice(moves) for _ in range(length))
        self.move(scramble)
        if self.verbose:
            print(f"Random scramble: {scramble}")
        return scramble
    
    # not used in solver but for playing around and testing
    def move(self, moves: str):

        # moves input as a string e.g. "U R U' R2" — single, double, or prime turns
        
        new_perm = list(range(54))
        for tok in moves.split(' '):
            if not tok:
                continue
            # previous moves first, then this move
            new_perm = Cube._compose(new_perm, self.table[tok])

        new_state = [self.state[new_perm[i]] for i in range(54)]
        
        self.state = new_state
        
        if self.verbose:
            print(f"After moves {moves}: {self}")
        
    
def test_cube(cube):
    # 1. Four of any move = identity
    for m in "UDLRFB":
        c = Cube()
        c.move(" ".join([m] * 4))
        assert "".join(c.state) == solved, f"{m}*4 failed"

    # 2. Move then inverse = identity
    for m in "UDLRFB":
        c = Cube()
        c.move(f"{m} {m}'")
        assert "".join(c.state) == solved, f"{m} {m}' failed"

    # 3. Sexy move (R U R' U') x 6 = identity — catches compose-order bugs
    c = Cube()
    c.move(" ".join(["R", "U", "R'", "U'"] * 6))
    assert "".join(c.state) == solved, "sexy move x 6 failed"

    # 4. M2 == M M
    for m in "UDLRFB":
        c1 = Cube(); c1.move(f"{m}2")
        c2 = Cube(); c2.move(f"{m} {m}")
        assert c1.state == c2.state, f"{m}2 != {m} {m}"

    print("All tests passed.")
    
if __name__ == "__main__":
    solved = Cube.solved_state
    test_cube(Cube())
    
    cube = Cube()
    cube.random_scramble()
    
    print(cube)

    