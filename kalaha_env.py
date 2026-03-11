import numpy as np
from gymnasium import Env, spaces
from gymnasium.utils import seeding

class KalahaEnv(Env):
    """
    Kalaha board game environment with customizable board size.
    
    Observation: 
        shape (2*n_pits + 2,) array: 
            - pits 0 to n_pits-1: Player 0 pits
            - pit n_pits: Player 0 store
            - pits n_pits+1 to 2*n_pits: Player 1 pits
            - pit 2*n_pits+1: Player 1 store
        Player perspective: Always P1 (0) = south, P2 (1) = north.
        Values: number of seeds in each pit/store.
    
    Actions:
        0 to n_pits-1: Choose pit on current player's side (0 = leftmost, n_pits-1 = rightmost).
        If the chosen pit is empty or invalid, the move is illegal.
    
    Reward:
        +1 for winning, -1 for losing, 0 for draw/non-terminal.
        Alternatively, score difference (store_self - store_opp).
    """
    metadata = {'render.modes': ['human', 'ansi', 'rgb_array']}
    
    def __init__(self, 
                 pits_per_player=6, 
                 seeds_per_pit=4,
                 reward_type='score_diff', 
                 render_mode='human', 
                 illegal_move_penalty=-10,
                 win_reward=10):
        """
        Initialize Kalaha environment.
        
        Args:
            pits_per_player (int): Number of pits on each player's side (default: 6)
            seeds_per_pit (int): Number of seeds in each pit at start (default: 4)
            reward_type (str): 'win_loss' or 'score_diff'
            render_mode (str): 'human', 'ansi', or 'rgb_array'
            illegal_move_penalty (float): Penalty for illegal moves (score_diff mode)
        """
        super().__init__()
        
        self.pits_per_player = pits_per_player
        self.seeds_per_pit = seeds_per_pit
        
        self.total_pits = self.pits_per_player * 2 + 2
        
        self.p0_pit_start = 0
        self.p0_pit_end = self.pits_per_player - 1
        self.p0_store_idx = self.pits_per_player
        
        self.p1_pit_start = self.pits_per_player + 1
        self.p1_pit_end = 2 * self.pits_per_player
        self.p1_store_idx = 2 * self.pits_per_player + 1
        
        self.action_space = spaces.Discrete(self.pits_per_player)
                
        self.reward_type = reward_type
        self.render_mode = render_mode
        self.illegal_move_penalty = illegal_move_penalty
        self.win_reward = win_reward

        self.board = None
        self.current_player = None
        self.done = False
        self.winner = None
        
        self.screen = None
        self.np_random = None
        self.seed()
    
    def seed(self, seed=None):
        """Set random seed."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self, *, seed=None, options=None):
        """Reset the game to initial state."""
        if seed is not None:
            self.seed(seed)
        
        # Initialize board: seeds_per_pit in each pit, 0 in stores
        self.board = np.zeros(self.total_pits, dtype=np.int32)
        
        # Player 0 pits
        self.board[self.p0_pit_start:self.p0_pit_start + self.pits_per_player] = self.seeds_per_pit
        # Player 0 store
        self.board[self.p0_store_idx] = 0
        
        # Player 1 pits
        self.board[self.p1_pit_start:self.p1_pit_start + self.pits_per_player] = self.seeds_per_pit
        # Player 1 store
        self.board[self.p1_store_idx] = 0
        
        self.current_player = 0
        self.done = False
        self.winner = None
        
        return self._get_obs(), self._get_info()
    
    def _get_obs(self):
        """Return board from perspective of current player."""
        if self.current_player == 0:
            return self.board.copy()
        else:
            # Rotate board: swap pits and stores
            obs = np.zeros_like(self.board)
            
            # Player 1's pits become Player 0's pits in observation
            obs[self.p0_pit_start:self.p0_pit_start + self.pits_per_player] = \
                self.board[self.p1_pit_start:self.p1_pit_start + self.pits_per_player]
            
            # Player 1's store becomes Player 0's store
            obs[self.p0_store_idx] = self.board[self.p1_store_idx]
            
            # Player 0's pits become Player 1's pits
            obs[self.p1_pit_start:self.p1_pit_start + self.pits_per_player] = \
                self.board[self.p0_pit_start:self.p0_pit_start + self.pits_per_player]
            
            # Player 0's store becomes Player 1's store
            obs[self.p1_store_idx] = self.board[self.p0_store_idx]
            
            return obs
    
    def _get_info(self):
        return {
            'current_player': self.current_player,
            'board': self.board.copy(),
            'winner': self.winner,
            'score': (self.board[self.p0_store_idx], self.board[self.p1_store_idx]),
            'available_actions': self.available_actions(),
            'pits_per_player': self.pits_per_player,
            'seeds_per_pit': self.seeds_per_pit
        }
    
    def available_actions(self):
        if self.done:
            return []
        
        legal_actions = []
        
        if self.current_player == 0:
            # Player 0: pits from p0_pit_start to p0_pit_start + pits_per_player - 1
            for action in range(self.pits_per_player):
                board_idx = self.p0_pit_start + action
                if self.board[board_idx] > 0:
                    legal_actions.append(action)
        else:
            # Player 1: pits from p1_pit_start to p1_pit_start + pits_per_player - 1
            for action in range(self.pits_per_player):
                board_idx = self.p1_pit_start + action
                if self.board[board_idx] > 0:
                    legal_actions.append(action)
        
        return np.array(legal_actions)
    
    def is_legal_action(self, action):
        return action in self.available_actions()
    
    def action_masks(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        legal_actions = self.available_actions()
        mask[legal_actions] = True
        return mask
    
    def _get_opponent_store_idx(self):
        return self.p1_store_idx if self.current_player == 0 else self.p0_store_idx
    
    def _get_player_pit_range(self):
        if self.current_player == 0:
            return self.p0_pit_start, self.p0_pit_start + self.pits_per_player - 1
        else:
            return self.p1_pit_start, self.p1_pit_start + self.pits_per_player - 1
    
    def _get_opponent_pit_range(self):
        if self.current_player == 0:
            return self.p1_pit_start, self.p1_pit_start + self.pits_per_player - 1
        else:
            return self.p0_pit_start, self.p0_pit_start + self.pits_per_player - 1
    
    def _get_opposite_pit(self, pit_idx):
        if self.current_player == 0:
            # Player 0's pit maps to Player 1's pit symmetrically
            offset = pit_idx - self.p0_pit_start
            return self.p1_pit_start + (self.pits_per_player - 1 - offset)
        else:
            # Player 1's pit maps to Player 0's pit
            offset = pit_idx - self.p1_pit_start
            return self.p0_pit_start + (self.pits_per_player - 1 - offset)
    
    def step(self, action):
        assert not self.done, "Game is done. Call reset()."
        
        prev_scores = (self.board[self.p0_store_idx], self.board[self.p1_store_idx])

        if not self.is_legal_action(action):
            #self.done = True
            #self.winner = 1 - self.current_player
            reward = self.illegal_move_penalty

            return self._get_obs(), reward, self.done, False, self._get_info()
        
        if self.current_player == 0:
            board_idx = self.p0_pit_start + action
        else:
            board_idx = self.p1_pit_start + action
        
        seeds = self.board[board_idx]
        self.board[board_idx] = 0
        idx = board_idx
        opponent_store = self._get_opponent_store_idx()
        
        while seeds > 0:
            idx = (idx + 1) % self.total_pits
            if idx == opponent_store:
                continue
            self.board[idx] += 1
            seeds -= 1
        
        # Captures
        player_pit_start, player_pit_end = self._get_player_pit_range()
        
        if player_pit_start <= idx <= player_pit_end and self.board[idx] == 1:
            opposite_idx = self._get_opposite_pit(idx)
            if self.board[opposite_idx] > 0:
                capture_amount = self.board[opposite_idx] + 1 
                
                if self.current_player == 0:
                    self.board[self.p0_store_idx] += capture_amount
                else:
                    self.board[self.p1_store_idx] += capture_amount
                
                self.board[idx] = 0
                self.board[opposite_idx] = 0
        
        # Free turn
        free_turn = (self.current_player == 0 and idx == self.p0_store_idx) or \
                    (self.current_player == 1 and idx == self.p1_store_idx)
        p0_pits_empty = np.sum(self.board[self.p0_pit_start:self.p0_pit_start + self.pits_per_player]) == 0
        p1_pits_empty = np.sum(self.board[self.p1_pit_start:self.p1_pit_start + self.pits_per_player]) == 0
        
        if p0_pits_empty or p1_pits_empty:
            # Collect remaining seeds into stores
            if p0_pits_empty:
                # Player 0 side empty: Player 1 gets all seeds from their pits
                p1_remaining = np.sum(self.board[self.p1_pit_start:self.p1_pit_start + self.pits_per_player])
                self.board[self.p1_store_idx] += p1_remaining
                self.board[self.p1_pit_start:self.p1_pit_start + self.pits_per_player] = 0
            else:
                # Player 1 side empty: Player 0 gets all seeds from their pits
                p0_remaining = np.sum(self.board[self.p0_pit_start:self.p0_pit_start + self.pits_per_player])
                self.board[self.p0_store_idx] += p0_remaining
                self.board[self.p0_pit_start:self.p0_pit_start + self.pits_per_player] = 0
            
            self.done = True
            # Determine winner
            if self.board[self.p0_store_idx] > self.board[self.p1_store_idx]:
                self.winner = 0
            elif self.board[self.p1_store_idx] > self.board[self.p0_store_idx]:
                self.winner = 1
            else:
                self.winner = None  # Draw
        

        
        reward = self._compute_reward(prev_scores=prev_scores)
        if not self.done and not free_turn:
            self.current_player = 1 - self.current_player
        return self._get_obs(), reward, self.done, False, self._get_info()
    
    def _compute_reward(self, prev_scores=0):
        match self.reward_type:
            case 'win_loss':
                if not self.done:
                    return 0.0
                if self.winner == self.current_player:
                    return self.win_reward
                elif self.winner is None:
                    return 0.0  # Draw
                else:
                    return -self.win_reward
            case 'score_diff':
                # From current player's perspective
                if self.current_player == 0:
                    return self.board[self.p0_store_idx] - self.board[self.p1_store_idx]
                else:
                    return self.board[self.p1_store_idx] - self.board[self.p0_store_idx]
            case "score_delta":
                if self.current_player == 0:
                    return self.board[self.p0_store_idx] - prev_scores[0]
                else:
                    return self.board[self.p1_store_idx] - prev_scores[1]
            case _:
                raise ValueError(f"Unknown reward type: {self.reward_type}")
    
    def render(self):
        if self.render_mode == 'human':
            print(self._render_board())
            print(f"Available actions: {self.available_actions()}")
        elif self.render_mode == 'ansi':
            return self._render_board() + f"\nAvailable actions: {self.available_actions()}"
        elif self.render_mode == 'rgb_array':
            # Return dummy array for compatibility
            return np.zeros((400, 600, 3), dtype=np.uint8)
    
    def _render_board(self):
        """String representation of the board."""
        # Get pits and stores
        p0_pits = self.board[self.p0_pit_start:self.p0_pit_start + self.pits_per_player]
        p0_store = self.board[self.p0_store_idx]
        
        p1_pits = self.board[self.p1_pit_start:self.p1_pit_start + self.pits_per_player]
        p1_store = self.board[self.p1_store_idx]
        
        # Reverse player 1 pits for display (so pit 0 is on left)
        p1_pits_display = p1_pits[::-1]
        
        player_str = "Player 1 (South)" if self.current_player == 0 else "Player 2 (North)"
        
        # Build display string
        board_str = f"\n{'=' * (8 * self.pits_per_player + 1)}\n"
        board_str += f"Player 2 (North) Store: {p1_store}\n"
        board_str += f"{'-' * (8 * self.pits_per_player + 1)}\n"
        
        # Player 2 pits (top row, displayed right to left)
        board_str += "   ".join([f"{p:2d}" for p in p1_pits_display]) + "\n"
        
        # Player 1 pits (bottom row, displayed left to right)
        board_str += "   ".join([f"{p:2d}" for p in p0_pits]) + "\n"
        
        board_str += f"{'-' * (8 * self.pits_per_player + 1)}\n"
        board_str += f"Player 1 (South) Store: {p0_store}\n"
        board_str += f"Current Player: {player_str}\n"
        board_str += f"{'=' * (8 * self.pits_per_player + 1)}\n"
        
        return board_str
    
    def close(self):
        """Clean up rendering resources."""
        if self.screen is not None:
            self.screen = None