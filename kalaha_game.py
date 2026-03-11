import pygame
import numpy as np
import time
import torch
from kalaha_env import KalahaEnv
from agent import A2CAgent, AgentNetwork

N_ACTIONS = 6
BOARD_SIZE = 2*6+2
COLORS = {
    'background': (139, 69, 19),
    'pit': (240, 217, 181),
    'store': (240, 217, 181),
    'player0': (0, 150, 0),
    'player1': (150, 0, 0),
    'text' : (0, 0, 0)
}

class KalahaPygameVisualizer:
    def __init__(self, env, screen_size=(1200, 600)):

        self.env = env
        self.screen_size = screen_size
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None
        self.title_font = None
               
        width, height = self.screen_size
        

        self.pit_size = 120
        self.pit_margin = (width - 2*(self.pit_size // 2 + self.pit_size) -  self.pit_size*self.env.pits_per_player) // (self.env.pits_per_player+1)


        # Store positions
        self.p1_store_rect = pygame.Rect(
            self.pit_size // 2,
            height // 2 - self.pit_size // 2,
            self.pit_size,
            self.pit_size
        )
        self.p0_store_rect = pygame.Rect(
            width - self.pit_size // 2 - self.pit_size,
            height // 2 - self.pit_size // 2,
            self.pit_size,
            self.pit_size
        )


        self.pit_rects = []
        # Player 0 pits (bottom row) 
        for i in range(self.env.pits_per_player):
            rect = pygame.Rect(
                (self.pit_size//2 +self.pit_size) \
                    + i * (self.pit_size + self.pit_margin)+ self.pit_margin,
                height // 2 - self.pit_size // 2 + self.pit_size,
                self.pit_size,
                self.pit_size,
            )
            self.pit_rects.append(rect)

        # Player 1 pits (top row - reversed)
        for i in range(self.env.pits_per_player):
            rect = pygame.Rect(
                (width - self.pit_size // 2 - self.pit_size - self.pit_size) \
                    - i * (self.pit_size + self.pit_margin)- self.pit_margin,
                height // 2 - self.pit_size // 2 - self.pit_size,
                self.pit_size,
                self.pit_size,
            )
            self.pit_rects.append(rect)
        
        if not pygame.get_init():
            pygame.init()

    
    def _init_pygame(self):
        if self.screen is None:
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption(f"Kalaha")
            self.font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 24)
            self.title_font = pygame.font.Font(None, 48)
    
    def _draw_pit(self, rect, seeds):
        pygame.draw.ellipse(self.screen, COLORS['pit'], rect)
       
        font_size = min(rect.height // 2, 48)
        font = pygame.font.Font(None, font_size)
        text = font.render(str(seeds), True, COLORS['text'])
        text_rect = text.get_rect(center=rect.center)
        self.screen.blit(text, text_rect)

    
    def _draw_store(self, rect, seeds, player):
        color = COLORS['store']
        pygame.draw.rect(self.screen, color, rect, border_radius=10)
        
        player_text = self.font.render(f"P{player + 1}", True, COLORS['text'])
        text_rect = player_text.get_rect(center=(rect.centerx, rect.y + 30))
        self.screen.blit(player_text, text_rect)
        
        seeds_text = self.font.render(str(seeds), True, COLORS['text'])
        seeds_rect = seeds_text.get_rect(center=rect.center)
        self.screen.blit(seeds_text, seeds_rect)
    
    def _draw_board(self):
        self.screen.fill(COLORS['background'])

        for i in range(self.env.pits_per_player):
            pit_idx = self.env.p0_pit_start + i
            seeds = self.env.board[pit_idx]
            self._draw_pit(self.pit_rects[i], seeds)

        for i in range(self.env.pits_per_player):
            pit_idx = self.env.p1_pit_start + i
            seeds = self.env.board[pit_idx]
            self._draw_pit(self.pit_rects[i + self.env.pits_per_player], seeds)

        self._draw_store(self.p0_store_rect, self.env.board[self.env.p0_store_idx], 0)
        self._draw_store(self.p1_store_rect, self.env.board[self.env.p1_store_idx], 1)
        
        if not self.env.done:
            player_text = self.title_font.render(
                f"Player {self.env.current_player + 1}'s turn", 
                True, 
                COLORS[f'player{self.env.current_player}']
            )
            text_rect = player_text.get_rect(center=(self.screen_size[0] // 2, 20))
            self.screen.blit(player_text, text_rect)
        
        if self.env.done:
            if self.env.winner is not None:
                winner_text = self.title_font.render(
                    f"Player {self.env.winner+1} Wins!", 
                    True, 
                    COLORS['text']
                )
            text_rect = winner_text.get_rect(center=(self.screen_size[0] // 2, 20))
            self.screen.blit(winner_text, text_rect)  

        pygame.display.flip()
    
    def _handle_click(self, pos):
        if not self.env.done:
            for i in range(self.env.pits_per_player):
                if self.pit_rects[i].collidepoint(pos):
                    if self.env.current_player == 0 and i in self.env.available_actions():
                        return ('action', i)
            
            for i in range(self.env.pits_per_player):
                if self.pit_rects[i + self.env.pits_per_player].collidepoint(pos):
                    if self.env.current_player == 1 and i in self.env.available_actions():
                        return ('action', i)
        return None
    
    
    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                result = self._handle_click(event.pos)
                if result and result[0] == 'action':
                    action = result[1]
                    obs, reward, done, truncated, info = self.env.step(action)

        self._draw_board()

        return True
    
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


class PlayableKalaha:
    def __init__(self, 
                 pits_per_player=6, 
                 seeds_per_pit=4,
                 screen_size=(1200, 600)):
        self.env = KalahaEnv(
            pits_per_player=pits_per_player,
            seeds_per_pit=seeds_per_pit,
            reward_type="win_loss",
            render_mode='human'
        )
        self.visualizer = KalahaPygameVisualizer(self.env, screen_size)
        self.running = True
    
    def play(self, opponent=None):
        self.visualizer._init_pygame()
        if opponent == "P":
            self.play_vs_human()
        elif opponent == "AI":
            self.play_vs_ai()
        else:
            print(f"Unknown opponent")
            self.visualizer.close()

    def play_vs_human(self):
        self.env.reset()  
        while self.running:
            self.running = self.visualizer.render()
        self.visualizer.close()
    
    @torch.no_grad()
    def play_vs_ai(self):
        try:
            device='cuda' if torch.cuda.is_available() else 'cpu'
            agent = A2CAgent(
                AgentNetwork(input_size=BOARD_SIZE, N_actions=N_ACTIONS, hidden_size=256).to(device), 
                device=device)
            agent.model.load_state_dict(torch.load("agent_weights.pt"))
        except Exception as e:
            print(f"Failed to load agent's state dict: {e}")
            self.visualizer.close()
            return

        state, info = self.env.reset()
        while self.running and not self.env.done:
            if self.env.current_player == 0:
                self.running = self.visualizer.render()
            else:
                pygame.time.wait(1000)
                indices_tensor = torch.from_numpy(self.env._get_info()["available_actions"])
                action_mask = torch.zeros(N_ACTIONS, dtype=torch.bool).to(device)
                action_mask[indices_tensor] = True
                state = self.env._get_obs()
                action, _, _,_ = agent.act(state, action_mask, greedy=True, eps_greedy=False)
                _, _, _, _, _ = self.env.step(action)

        while self.running:
            self.running = self.visualizer.render()
        self.visualizer.close()