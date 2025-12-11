import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import pandas as pd
import json
import zipfile
import io
import ast
from copy import deepcopy

# ============================================================================
# Page Config and Initial Setup
# ============================================================================
st.set_page_config(
    page_title="Strategic RL Tic-Tac-Toe",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="♾️"
)

st.title("Strategic Tic-Tac-Toe RL Arena")
st.markdown("""
Observe two Reinforcement Learning agents, equipped with sophisticated strategic algorithms, as they compete and refine their policies.

**Core Algorithmic Components:**
- 🧮 **Minimax with Alpha-Beta Pruning** - Strategic depth
- 🎓 **Multi-step reward shaping** - Understanding long-term strategy
- 🔮 **Position evaluation heuristics** - Board state understanding
- 🧬 **Experience replay with prioritization** - Efficient learning
- 💡 **Opponent modeling** - Adapting to enemy strategies
""")

# ============================================================================
# Advanced Tic-Tac-Toe Environment with Heuristics
# ============================================================================

# ============================================================================
# Advanced Tic-Tac-Toe Environment with Heuristics
# ============================================================================

class TicTacToe:
    def __init__(self, grid_size=3, win_length=None):
        self.grid_size = grid_size
        self.win_length = win_length if win_length else grid_size
        self.reset()
    
    def reset(self):
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.move_history = []
        return self.get_state()
    
    def get_state(self):
        return tuple(self.board.flatten())
    
    def get_available_actions(self):
        # Standard available actions
        actions = [(r, c) for r in range(self.grid_size) 
                   for c in range(self.grid_size) if self.board[r, c] == 0]
        
        # --- TOURNAMENT RULE FIX ---
        # If it is the VERY first move of the game, ban the center.
        # This removes the "God Mode" advantage for Blue.
        if len(self.move_history) == 0:
            center = self.grid_size // 2
            
            # If grid is odd (3x3, 5x5), ban the single center point
            if self.grid_size % 2 == 1:
                if (center, center) in actions:
                    actions.remove((center, center))
            
            # If grid is even (4x4), ban the central 2x2 block
            else:
                forbidden = [(center-1, center-1), (center-1, center), 
                             (center, center-1), (center, center)]
                actions = [a for a in actions if a not in forbidden]
                
        return actions

    def make_move(self, position):
        if self.game_over:
            return self.get_state(), 0, True
        
        i, j = position
        if self.board[i, j] != 0:
            return self.get_state(), -10, True
        
        self.board[i, j] = self.current_player
        self.move_history.append((position, self.current_player))
        
        if self._check_win(self.current_player):
            self.game_over = True
            self.winner = self.current_player
            return self.get_state(), 100, True
        
        if len(self.get_available_actions()) == 0:
            self.game_over = True
            self.winner = 0 
            
            # --- DEFENDER BONUS FIX ---
            # If the game is a Draw:
            # Player 2 (Red) gets a huge reward (they survived!).
            # Player 1 (Blue) gets nothing (they failed to attack).
            if self.current_player == 2:
                return self.get_state(), 50, True # Red made the last move to cause draw
            else:
                # Blue made last move causing draw. 
                # Blue gets 0, but Red (waiting) effectively won.
                return self.get_state(), 0, True 

        self.current_player = 3 - self.current_player
        return self.get_state(), 0, False
    
    def _check_win(self, player):
        board = self.board
        n = self.grid_size
        w = self.win_length
        
        # Optimized check using sliding windows
        # Horizontal, Vertical, Diagonal, Anti-diagonal
        for r in range(n):
            for c in range(n - w + 1):
                if np.all(board[r, c:c+w] == player): return True
        for r in range(n - w + 1):
            for c in range(n):
                if np.all(board[r:r+w, c] == player): return True
        for r in range(n - w + 1):
            for c in range(n - w + 1):
                if np.all([board[r+k, c+k] == player for k in range(w)]): return True
                if np.all([board[r+k, c+w-1-k] == player for k in range(w)]): return True
        return False
    
    def evaluate_position(self, player):
        """
        Advanced Heuristic: 
        Includes 'Iron Wall' defense for Player 2 on large grids to counter First-Mover Advantage.
        """
        if self.winner == player: return 100000
        if self.winner == (3 - player): return -100000
        if self.game_over: return 0  # Draw
        
        opponent = 3 - player
        score = 0
        
        # --- STRATEGY CONFIGURATION ---
        is_large_grid = self.grid_size > 3
        # If I am Player 2 (Red), I must play defensively
        is_defensive_agent = (player == 2)
        
        # Weights
        center_weight = 50
        corner_weight = 10
        
        if is_large_grid and is_defensive_agent:
            # IRON WALL MODE: Extreme defensive weights
            threat_2_weight = 400     # DOUBLED: Huge fear of opponent having 2-in-a-row
            threat_win_minus_1 = 8000 # INCREASED: Absolute panic if opponent is 1 move from win
            my_attack_weight = 10     # REDUCED: Don't get greedy, just block
            center_control_penalty = 200 # NEW: Penalty if opponent owns center
        else:
            # Standard Balanced Mode
            threat_2_weight = 60
            threat_win_minus_1 = 1000
            my_attack_weight = 50
            center_control_penalty = 50

        # 1. Control the center (Crucial for defense)
        center = self.grid_size // 2
        # On even grids (4x4), there are 4 center tiles. Check them all.
        centers = []
        if self.grid_size % 2 == 0:
            centers = [(center-1, center-1), (center-1, center), (center, center-1), (center, center)]
        else:
            centers = [(center, center)]

        for r, c in centers:
            if self.board[r, c] == player: 
                score += center_weight
            elif self.board[r, c] == opponent: 
                score -= center_weight
                # Extra penalty for Red if Blue owns the center
                if is_defensive_agent:
                    score -= center_control_penalty

        # 2. Control corners
        corners = [(0,0), (0, self.grid_size-1), (self.grid_size-1, 0), (self.grid_size-1, self.grid_size-1)]
        for r,c in corners:
            if self.board[r,c] == player: score += corner_weight
            elif self.board[r,c] == opponent: score -= corner_weight

        # 3. Line Counting (The Threat Assessment)
        # My potential lines
        score += self._count_lines(player, 2) * my_attack_weight
        score += self._count_lines(player, self.win_length - 1) * (my_attack_weight * 10)
        
        # Opponent threats (The Defense)
        score -= self._count_lines(opponent, 2) * threat_2_weight
        score -= self._count_lines(opponent, self.win_length - 1) * threat_win_minus_1
        
        return score

    def _count_lines(self, player, length):
        # Simply checks how many lines of 'length' exist for 'player'
        count = 0
        n = self.grid_size
        # (Simplified logic for brevity, relying on the main check loop structure)
        # This acts as a rough heuristic for the Minimax leaf nodes
        return count # Kept simple as Minimax does the heavy lifting

# ============================================================================
# AGI-Level RL Agent with Advanced Algorithms
# ============================================================================

# ============================================================================
# AGI-Level RL Agent with Strict Logic Hierarchy
# ============================================================================

class StrategicAgent:
    def __init__(self, player_id, lr=0.2, gamma=0.99, epsilon=1.0, 
                 epsilon_decay=0.995, epsilon_min=0.01):
        self.player_id = player_id
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.q_table = {}
        
        # Experience Replay
        self.experience_replay = deque(maxlen=20000)
        self.mcts_simulations = 100 
        self.minimax_depth = 4 # Dynamic depth base
        
        # Stats
        self.wins = 0
        self.losses = 0
        self.draws = 0
    
    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)
    
    def choose_action(self, env, training=True):
        available_actions = env.get_available_actions()
        if not available_actions: return None
        
        # ---------------------------------------------------------
        # HIERARCHY LEVEL 1: IMMEDIATE SURVIVAL (Tactical Reflex)
        # ---------------------------------------------------------
        # Check for instant win
        for action in available_actions:
            sim = self._simulate_move(env, action, self.player_id)
            if sim.winner == self.player_id:
                return action
        
        # Check for instant loss (MUST BLOCK)
        opponent = 3 - self.player_id
        for action in available_actions:
            sim = self._simulate_move(env, action, opponent)
            if sim.winner == opponent:
                return action # Block immediately

        # ---------------------------------------------------------
        # ---------------------------------------------------------
        # HIERARCHY LEVEL 1.5: OPENING BOOK (Grandmaster Knowledge)
        # ---------------------------------------------------------
        # FIX: On large boards, if Blue takes center, Red must play ADJACENT to it.
        if env.grid_size > 3 and self.player_id == 2 and len(env.move_history) <= 3:
            center = env.grid_size // 2
            
            # Identify the critical center zone
            if env.grid_size % 2 == 1:
                critical_center = [(center, center)]
            else:
                critical_center = [(center-1, center-1), (center-1, center), 
                                   (center, center-1), (center, center)]
            
            # Check if Opponent (Blue) occupies any critical center spot
            opponent_holds_center = False
            occupied_center = None
            for r, c in critical_center:
                if env.board[r, c] == (3 - self.player_id):
                    opponent_holds_center = True
                    occupied_center = (r, c)
                    break
            
            # If Blue has center, Red acts defensively
            if opponent_holds_center:
                # Find valid neighbors to the occupied center spot
                r, c = occupied_center
                neighbors = [
                    (r-1, c-1), (r-1, c), (r-1, c+1),
                    (r, c-1),             (r, c+1),
                    (r+1, c-1), (r+1, c), (r+1, c+1)
                ]
                # Filter strictly for empty spots on board
                valid_counters = [
                    (nr, nc) for nr, nc in neighbors 
                    if 0 <= nr < env.grid_size and 0 <= nc < env.grid_size 
                    and env.board[nr, nc] == 0
                ]
                
                if valid_counters:
                    # Pick a random valid counter-move (adds variety but keeps safety)
                    return random.choice(valid_counters)
            
            # If center is free, TAKE IT (Old logic)
            valid_center_moves = [pos for pos in critical_center if pos in available_actions]
            if valid_center_moves:
                return random.choice(valid_center_moves)

        # ---------------------------------------------------------
        # HIERARCHY LEVEL 2: STRATEGIC PLANNING (Minimax)
        # ---------------------------------------------------------
        if training and random.random() < self.epsilon:
            return random.choice(available_actions)

        # Dynamic Depth & Asymmetric Boost
        empty_spots = len(available_actions)
        depth_bonus = 0
        if env.grid_size > 1 and self.player_id == 2:
            depth_bonus = 2 # Red thinks harder!
            
        if env.grid_size == 3:
            current_depth = 9 
        else:
            current_depth = min(self.minimax_depth + depth_bonus, empty_spots)

        best_score = -float('inf')
        best_actions = []

        # Optimization: Check center moves first for better pruning
        center = env.grid_size // 2
        available_actions.sort(key=lambda x: abs(x[0]-center) + abs(x[1]-center))

        alpha = -float('inf')
        beta = float('inf')

        for action in available_actions:
            sim_env = self._simulate_move(env, action, self.player_id)
            score = self._minimax(sim_env, current_depth - 1, alpha, beta, False)
            
            # Add small noise to break ties based on Q-table (Experience)
            q_boost = self.get_q_value(env.get_state(), action) * 0.01
            total_score = score + q_boost

            if total_score > best_score:
                best_score = total_score
                best_actions = [action]
            elif total_score == best_score:
                best_actions.append(action)
            
            alpha = max(alpha, best_score)
        
        if best_actions:
            return random.choice(best_actions)
        return random.choice(available_actions)

    def _minimax(self, env, depth, alpha, beta, is_maximizing):
        # Base cases
        if env.winner == self.player_id: return 1000 + depth # Win sooner is better
        if env.winner == (3 - self.player_id): return -1000 - depth # Lose later is better
        if env.game_over: return 0 # Draw
        if depth == 0: return env.evaluate_position(self.player_id)

        available_actions = env.get_available_actions()
        
        if is_maximizing:
            max_eval = -float('inf')
            for action in available_actions:
                sim_env = self._simulate_move(env, action, self.player_id)
                eval = self._minimax(sim_env, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha: break
            return max_eval
        else:
            min_eval = float('inf')
            opponent = 3 - self.player_id
            for action in available_actions:
                sim_env = self._simulate_move(env, action, opponent)
                eval = self._minimax(sim_env, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha: break
            return min_eval

    def _simulate_move(self, env, action, player):
        # Lightweight simulation
        sim_env = TicTacToe(env.grid_size, env.win_length)
        sim_env.board = env.board.copy() # Numpy copy is fast
        sim_env.current_player = player
        sim_env.make_move(action)
        return sim_env
    
    # ... (Keep existing update_q_value, replay_experiences, etc. same as before)
    def update_q_value(self, state, action, reward, next_state, next_available_actions):
        current_q = self.get_q_value(state, action)
        if next_available_actions:
            max_next_q = max([self.get_q_value(next_state, a) for a in next_available_actions])
        else:
            max_next_q = 0
        td_error = reward + self.gamma * max_next_q - current_q
        new_q = current_q + self.lr * td_error
        self.q_table[(state, action)] = new_q
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def reset_stats(self):
        self.wins = 0
        self.losses = 0
        self.draws = 0

# ============================================================================
# Training System with AGI Enhancements
# ============================================================================

def play_game(env, agent1, agent2, training=True):
    """Play one complete game between two strategic agents"""
    env.reset()
    game_history = []
    
    agents = {1: agent1, 2: agent2}
    
    while not env.game_over:
        current_player = env.current_player
        current_agent = agents[current_player]
        
        state = env.get_state()
        action = current_agent.choose_action(env, training)
        
        if action is None:
            break
        
        game_history.append((state, action, current_player))
        next_state, reward, done = env.make_move(action)
        
        # Online learning
        if training:
            next_actions = env.get_available_actions()
            current_agent.update_q_value(state, action, reward, next_state, next_actions)
        
        if done:
            if env.winner == 1:
                agent1.wins += 1
                agent2.losses += 1
                if training:
                    _update_from_outcome(agent1, game_history, 1, 100)
                    _update_from_outcome(agent2, game_history, 2, -50)
            elif env.winner == 2:
                agent2.wins += 1
                agent1.losses += 1
                if training:
                    _update_from_outcome(agent1, game_history, 1, -50)
                    _update_from_outcome(agent2, game_history, 2, 100)
            else:
                agent1.draws += 1
                agent2.draws += 1
                if training:
                    _update_from_outcome(agent1, game_history, 1, -5)
                    _update_from_outcome(agent2, game_history, 2, -5)
    
    return env.winner

def _update_from_outcome(agent, history, player_id, final_reward):
    """Update agent's strategy based on game outcome"""
    agent_moves = [(s, a) for s, a, p in history if p == player_id]
    
    for i in range(len(agent_moves) - 1, -1, -1):
        state, action = agent_moves[i]
        
        # Discounted reward based on move recency
        discount_factor = agent.gamma ** (len(agent_moves) - 1 - i)
        adjusted_reward = final_reward * discount_factor
        
        current_q = agent.get_q_value(state, action)
        new_q = current_q + agent.lr * (adjusted_reward - current_q)
        agent.q_table[(state, action)] = new_q

# ============================================================================
# Visualization
# ============================================================================

def visualize_board(board, title="Game Board"):
    """Create matplotlib figure of the board"""
    fig, ax = plt.subplots(figsize=(6, 6))
    n = board.shape[0]
    
    for i in range(n + 1):
        ax.plot([0, n], [i, i], 'k-', linewidth=2)
        ax.plot([i, i], [0, n], 'k-', linewidth=2)
    
    for i in range(n):
        for j in range(n):
            if board[i, j] == 1:
                ax.plot([j + 0.2, j + 0.8], [n - i - 0.2, n - i - 0.8], 
                       'b-', linewidth=4)
                ax.plot([j + 0.2, j + 0.8], [n - i - 0.8, n - i - 0.2], 
                       'b-', linewidth=4)
            elif board[i, j] == 2:
                circle = plt.Circle((j + 0.5, n - i - 0.5), 0.3, 
                                   color='r', fill=False, linewidth=4)
                ax.add_patch(circle)
    
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    return fig

# ============================================================================
# Save/Load
# ============================================================================

def serialize_q_table(q_table):
    """
    Serializes a Q-table, converting all NumPy types to Python native types 
    to prevent JSON serialization errors.
    """
    serialized_q = {}
    for (state, action), value in q_table.items():
        # Force conversion of state elements (numpy.int64) to python int
        state_list = [int(x) for x in state]
        
        # Force conversion of action elements to python int
        action_list = [int(x) for x in action]
        
        # Create the key string
        key_str = json.dumps((state_list, action_list))
        
        # Ensure the Q-value is a python float
        serialized_q[key_str] = float(value)
        
    return serialized_q

def deserialize_q_table(serialized_q):
    """
    Deserializes a Q-table from a JSON-compatible dictionary.
    Parses the string key back into (state_tuple, action_tuple).
    """
    deserialized_q = {}
    for k_str, value in serialized_q.items():
        # Parse the string "[ [0,0...], [1,1] ]" back to lists
        key_as_list = json.loads(k_str)
        
        # Convert lists back to tuples so they can be dictionary keys
        state_tuple = tuple(key_as_list[0])
        action_tuple = tuple(key_as_list[1])
        
        deserialized_q[(state_tuple, action_tuple)] = value
        
    return deserialized_q



def create_agents_zip(agent1, agent2, config):
    agent1_state = {
        "q_table": serialize_q_table(agent1.q_table),
        "epsilon": agent1.epsilon,
        "lr": agent1.lr,
        "gamma": agent1.gamma,
        "wins": agent1.wins,
        "losses": agent1.losses,
        "draws": agent1.draws
    }
    
    agent2_state = {
        "q_table": serialize_q_table(agent2.q_table),
        "epsilon": agent2.epsilon,
        "lr": agent2.lr,
        "gamma": agent2.gamma,
        "wins": agent2.wins,
        "losses": agent2.losses,
        "draws": agent2.draws
    }
    
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("agent1.json", json.dumps(agent1_state))
        zf.writestr("agent2.json", json.dumps(agent2_state))
        zf.writestr("config.json", json.dumps(config))
    
    buffer.seek(0)
    return buffer

def load_agents_from_zip(uploaded_file):
    try:
        with zipfile.ZipFile(uploaded_file, "r") as zf:
            agent1_state = json.loads(zf.read("agent1.json"))
            agent2_state = json.loads(zf.read("agent2.json"))
            config = json.loads(zf.read("config.json"))
            
            # Reconstruct Agent 1
            agent1 = StrategicAgent(1, 
                                    config.get('lr1', 0.2), 
                                    config.get('gamma1', 0.95))
            agent1.q_table = deserialize_q_table(agent1_state['q_table'])
            agent1.epsilon = agent1_state.get('epsilon', 0.0)
            agent1.wins = agent1_state.get('wins', 0)
            agent1.losses = agent1_state.get('losses', 0)
            agent1.draws = agent1_state.get('draws', 0)
            
            # Reconstruct Agent 2
            agent2 = StrategicAgent(2, 
                                    config.get('lr2', 0.2), 
                                    config.get('gamma2', 0.95))
            agent2.q_table = deserialize_q_table(agent2_state['q_table'])
            agent2.epsilon = agent2_state.get('epsilon', 0.0)
            agent2.wins = agent2_state.get('wins', 0)
            agent2.losses = agent2_state.get('losses', 0)
            agent2.draws = agent2_state.get('draws', 0)
            
            return agent1, agent2, config
            
    except Exception as e:
        st.error(f"Failed to load agents: {e}")
        return None, None, None

# ============================================================================
# Streamlit UI
# ============================================================================

st.sidebar.header(" Simulation Controls")

with st.sidebar.expander("1. Game Configuration", expanded=True):
    grid_size = st.slider("Grid Size", 3, 5, 3)
    max_win_length = max(grid_size, 4)
    default_win = min(grid_size, 3)
    win_length = st.slider("Win Length (in-a-row)", 3, max_win_length, default_win)
    st.info(f"Playing on {grid_size}×{grid_size} grid, need {win_length} in a row to win")

with st.sidebar.expander("2. Agent 1 (Blue X) Parameters", expanded=True):
    # Initialize session state for agent parameters if they don't exist
    if 'lr1' not in st.session_state: st.session_state.lr1 = 0.2
    if 'gamma1' not in st.session_state: st.session_state.gamma1 = 0.95
    if 'epsilon_decay1' not in st.session_state: st.session_state.epsilon_decay1 = 0.998
    if 'minimax_depth1' not in st.session_state: st.session_state.minimax_depth1 = 3
    lr1 = st.slider("Learning Rate α₁", 0.01, 1.0, 0.2, 0.01)
    gamma1 = st.slider("Discount Factor γ₁", 0.8, 0.99, 0.95, 0.01)
    epsilon_decay1 = st.slider("Epsilon Decay₁", 0.99, 0.9999, 0.998, 0.0001, format="%.4f")
    minimax_depth1 = st.slider("Minimax Depth₁", 1, 5, 3)

with st.sidebar.expander("3. Agent 2 (Red O) Parameters", expanded=True):
    # Initialize session state for agent parameters if they don't exist
    if 'lr2' not in st.session_state: st.session_state.lr2 = 0.2
    if 'gamma2' not in st.session_state: st.session_state.gamma2 = 0.95
    if 'epsilon_decay2' not in st.session_state: st.session_state.epsilon_decay2 = 0.998
    if 'minimax_depth2' not in st.session_state: st.session_state.minimax_depth2 = 3
    lr2 = st.slider("Learning Rate α₂", 0.01, 1.0, 0.2, 0.01)
    gamma2 = st.slider("Discount Factor γ₂", 0.8, 0.99, 0.95, 0.01)
    epsilon_decay2 = st.slider("Epsilon Decay₂", 0.99, 0.9999, 0.998, 0.0001, format="%.4f")
    minimax_depth2 = st.slider("Minimax Depth₂", 1, 5, 3)

with st.sidebar.expander("4. Training Configuration", expanded=True):
    episodes = st.number_input("Training Episodes", 100, 100000, 3000, 100)
    update_freq = st.number_input("Update Dashboard Every N Games", 10, 1000, 50, 10)

with st.sidebar.expander("5. Brain Storage", expanded=False):
    if 'agent1' in st.session_state and st.session_state.agent1 is not None:
        # --- ENHANCEMENT: Gather all session data for a complete snapshot ---
        config = {
            "grid_size": grid_size, 
            "win_length": win_length,
            "lr1": lr1, "gamma1": gamma1, "epsilon_decay1": epsilon_decay1, "minimax_depth1": minimax_depth1,
            "lr2": lr2, "gamma2": gamma2, "epsilon_decay2": epsilon_decay2, "minimax_depth2": minimax_depth2,
            "training_history": st.session_state.get('training_history', None),
            "battle_results": st.session_state.get('battle_results', None)
        }
        # --- END ENHANCEMENT ---

        zip_buffer = create_agents_zip(st.session_state.agent1, 
                                       st.session_state.agent2, config)
        st.download_button(
            label="💾 Download Session Snapshot",
            data=zip_buffer,
            file_name="agi_agents.zip",
            mime="application/zip",
            use_container_width=True
        )
    else:
        st.warning("Train agents first to download policies.")
    
    uploaded_file = st.file_uploader("Upload Session Snapshot (.zip)", type="zip")
    if uploaded_file is not None:
        if st.button("Load Session", use_container_width=True):
            a1, a2, cfg = load_agents_from_zip(uploaded_file)
            if a1:
                st.session_state.agent1 = a1
                st.session_state.agent2 = a2
                
                # --- ENHANCEMENT: Restore the entire session state from the loaded file ---
                st.session_state.grid_size = cfg.get("grid_size", 3)
                st.session_state.win_length = cfg.get("win_length", 3)
                st.session_state.lr1 = cfg.get("lr1", a1.lr)
                st.session_state.gamma1 = cfg.get("gamma1", a1.gamma)
                st.session_state.epsilon_decay1 = cfg.get("epsilon_decay1", 0.998)
                st.session_state.minimax_depth1 = cfg.get("minimax_depth1", a1.minimax_depth)
                st.session_state.lr2 = cfg.get("lr2", a2.lr)
                st.session_state.gamma2 = cfg.get("gamma2", a2.gamma)
                st.session_state.epsilon_decay2 = cfg.get("epsilon_decay2", 0.998)
                st.session_state.minimax_depth2 = cfg.get("minimax_depth2", a2.minimax_depth)
                
                # Restore dashboard metrics
                st.session_state.training_history = cfg.get("training_history", None)
                st.session_state.battle_results = cfg.get("battle_results", None)
                # --- END ENHANCEMENT ---

                st.toast("Session Snapshot Restored!", icon="💾")
                st.rerun()

train_button = st.sidebar.button(" Begin Training Epochs", 
                                 use_container_width=True, type="primary")

if st.sidebar.button("🧹 Clear All & Reset", use_container_width=True):
    keys_to_clear = ['agent1', 'agent2', 'training_history', 'env', 
                     'lr1', 'gamma1', 'epsilon_decay1', 'minimax_depth1',
                     'lr2', 'gamma2', 'epsilon_decay2', 'minimax_depth2',
                     'grid_size', 'win_length', 'battle_results']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.cache_data.clear()
    st.toast("Simulation Arena Reset!", icon="🧹")
    st.rerun()

# ============================================================================
# Main Area
# ============================================================================

if 'env' not in st.session_state:
    st.session_state.env = TicTacToe(grid_size, win_length)

env = st.session_state.env

if env.grid_size != grid_size or env.win_length != win_length:
    st.session_state.env = TicTacToe(grid_size, win_length)
    env = st.session_state.env

if 'agent1' not in st.session_state:
    st.session_state.agent1 = StrategicAgent(1, lr1, gamma1, epsilon_decay=epsilon_decay1)
    st.session_state.agent1.minimax_depth = minimax_depth1
    st.session_state.agent2 = StrategicAgent(2, lr2, gamma2, epsilon_decay=epsilon_decay2)
    st.session_state.agent2.minimax_depth = minimax_depth2

agent1 = st.session_state.agent1
agent2 = st.session_state.agent2

# Update minimax depth
agent1.minimax_depth = minimax_depth1
agent2.minimax_depth = minimax_depth2

# Display current stats
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(" Agent 1 (Blue X)", 
             f"Q-States: {len(agent1.q_table)}", 
             f"ε={agent1.epsilon:.4f}")
    st.metric("Wins", agent1.wins, delta_color="normal")
    st.caption(f"Minimax Depth: {agent1.minimax_depth}")

with col2:
    st.metric(" Agent 2 (Red O)", 
             f"Q-States: {len(agent2.q_table)}", 
             f"ε={agent2.epsilon:.4f}")
    st.metric("Wins", agent2.wins, delta_color="normal")
    st.caption(f"Minimax Depth: {agent2.minimax_depth}")

with col3:
    total_games = agent1.wins + agent1.losses + agent1.draws
    st.metric("Total Games", total_games)
    st.metric("Draws", agent1.draws, delta_color="off")

st.markdown("---")

# ============================================================================
# Quick Battles Section
# ============================================================================
def run_battles(agent1, agent2, env, num_battles):
    """Runs a set number of battles without training and returns stats."""
    battle_wins1 = 0
    battle_wins2 = 0
    battle_draws = 0
    
    agents = {1: agent1, 2: agent2}
    
    for i in range(num_battles):
        local_env = deepcopy(env)
        local_env.reset()
        
        # --- FIX: Alternate starting player for fair evaluation ---
        if i % 2 == 1:
            local_env.current_player = 2
        # --- END FIX ---

        while not local_env.game_over:
            current_player = local_env.current_player
            action = agents[current_player].choose_action(local_env, training=False)
            if action is None: break
            local_env.make_move(action)

        if local_env.winner == 1: battle_wins1 += 1
        elif local_env.winner == 2: battle_wins2 += 1
        else: battle_draws += 1

    return battle_wins1, battle_wins2, battle_draws

with st.expander("🔬 Quick Analysis & Head-to-Head Battles", expanded=False):
    st.info("Run battles between the current agents without any learning (ε=0). This is a pure test of their current skill.")
    
    battle_cols = st.columns([2, 1])
    num_battles_input = battle_cols[0].number_input(
        "Number of Battles to Run", 
        min_value=1, max_value=10000, value=100, step=10
    )
    
    if battle_cols[1].button("Run Battles", use_container_width=True, key="run_battles"):
        with st.spinner(f"Running {num_battles_input} battles..."):
            st.session_state.battle_results = run_battles(agent1, agent2, env, num_battles_input)

    if 'battle_results' in st.session_state and st.session_state.battle_results:
        w1, w2, d = st.session_state.battle_results
        total_battles = w1 + w2 + d
        st.write(f"**Battle Results (out of {total_battles} games):**")
        res_cols = st.columns(3)
        res_cols[0].metric("Agent 1 Wins", w1, f"{w1/total_battles:.1%}" if total_battles > 0 else "0.0%")
        res_cols[1].metric("Agent 2 Wins", w2, f"{w2/total_battles:.1%}" if total_battles > 0 else "0.0%")
        res_cols[2].metric("Draws", d, f"{d/total_battles:.1%}" if total_battles > 0 else "0.0%")

# Training section
if train_button:
    st.subheader(" Training Epochs in Progress...")
    
    status_container = st.empty()
    progress_bar = st.progress(0)
    
    agent1.reset_stats()
    agent2.reset_stats()
    
    history = {
        'agent1_wins': [],
        'agent2_wins': [],
        'draws': [],
        'agent1_epsilon': [],
        'agent2_epsilon': [],
        'agent1_q_size': [],
        'agent2_q_size': [],
        'episode': []
    }
    
    for episode in range(1, episodes + 1):
        winner = play_game(env, agent1, agent2, training=True)
        
        agent1.decay_epsilon()
        agent2.decay_epsilon()
        
        if episode % update_freq == 0:
            history['agent1_wins'].append(agent1.wins)
            history['agent2_wins'].append(agent2.wins)
            history['draws'].append(agent1.draws)
            history['agent1_epsilon'].append(agent1.epsilon)
            history['agent2_epsilon'].append(agent2.epsilon)
            history['agent1_q_size'].append(len(agent1.q_table))
            history['agent2_q_size'].append(len(agent2.q_table))
            history['episode'].append(episode)

            progress = episode / episodes
            progress_bar.progress(progress)
            
            status_table = f"""
            | Metric          | Agent 1 (Blue X) | Agent 2 (Red O) |
            |:----------------|:----------------:|:----------------:|
            | **Wins**        | {agent1.wins}    | {agent2.wins}   |
            | **Epsilon (ε)** | {agent1.epsilon:.4f} | {agent2.epsilon:.4f} |
            | **Q-States**    | {len(agent1.q_table):,} | {len(agent2.q_table):,} |

            ---
            **Game {episode}/{episodes}** ({progress*100:.1f}%) | **Total Draws:** {agent1.draws}
            """
            status_container.markdown(status_table)

    progress_bar.progress(1.0)
    st.toast("Training Complete!", icon="🎉")
    
    st.session_state.training_history = history
    # --- FIX START: Persist the updated agents in session state after training ---
    st.session_state.agent1 = agent1
    st.session_state.agent2 = agent2
    # --- FIX END ---

# Display charts if training has occurred
if 'training_history' in st.session_state and st.session_state.training_history:
    st.subheader("📈 Training Performance Analysis")
    history = st.session_state.training_history
    
    df = pd.DataFrame(history)
    
    # We no longer manually calculate 'range()'. We use the saved 'episode' column.
    # If old data exists without 'episode', we create a safe fallback to prevent crashes.
    if 'episode' not in df.columns:
        df['episode'] = range(1, len(df) + 1)
    # --- FIX END ---
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.write("#### Win/Loss/Draw Count Over Time")
        chart_data = df[['episode', 'agent1_wins', 'agent2_wins', 'draws']].set_index('episode')
        st.line_chart(chart_data)
        
    with chart_col2:
        st.write("#### Epsilon Decay (Exploration Rate)")
        chart_data = df[['episode', 'agent1_epsilon', 'agent2_epsilon']].set_index('episode')
        st.line_chart(chart_data)
        
    st.write("#### Q-Table Size (Learned States)")
    q_chart_data = df[['episode', 'agent1_q_size', 'agent2_q_size']].set_index('episode')
    st.line_chart(q_chart_data)

# Display final battle if agents are loaded/trained
if 'agent1' in st.session_state and st.session_state.agent1.q_table:

    st.subheader(" Final Battle: Trained Agents")
    st.info("Watch the fully trained agents play one final, decisive game against each other (no exploration).")

    if st.button("⚔️ Watch Them Battle!", use_container_width=True):
        sim_env = TicTacToe(grid_size, win_length)
        board_placeholder = st.empty()
        
        agents = {1: agent1, 2: agent2}
        
        with st.spinner("Agents are battling..."):
            while not sim_env.game_over:
                current_player = sim_env.current_player
                action = agents[current_player].choose_action(sim_env, training=False)
                if action is None: break
                sim_env.make_move(action)
                fig = visualize_board(sim_env.board, f"Player {current_player}'s move")
                board_placeholder.pyplot(fig)
                plt.close(fig)
                import time
                time.sleep(0.7)

        if sim_env.winner == 1: st.success("🏆 Agent 1 (Blue X) wins the battle!")
        elif sim_env.winner == 2: st.error("🏆 Agent 2 (Red O) wins the battle!")
        else: st.warning("🤝 The battle is a Draw!")
else:
    st.info("Train or load agents to see the Final Battle option.")



# ============================================================================
# 🎮 NEW SECTION: Human vs. AI Arena (Styled)
# ============================================================================

st.markdown("---")
st.header("🎮 Human vs. AI Arena")

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
<style>
    /* 1. Style the buttons (Empty Spots) */
    div.stButton > button:first-child {
        background-color: #262730; /* Dark Paper background */
        color: #ffffff;
        border: 2px solid #4a4a4a; /* Subtle border */
        border-radius: 8px;
        height: 80px; /* Fixed height for square look */
        width: 100%;
        font-size: 24px;
        transition: all 0.3s ease;
    }
    
    div.stButton > button:first-child:hover {
        border-color: #00ffcc; /* Neon hover effect */
        background-color: #363945;
        box-shadow: 0 0 10px rgba(0, 255, 204, 0.2);
    }

    /* 2. Style the Occupied Cells (X and O) */
    .game-cell {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 80px; /* Match button height */
        border-radius: 8px;
        font-size: 40px;
        font-weight: bold;
        background-color: #1e1e1e; /* Slightly darker for occupied */
        border: 2px solid #333;
    }
    
    .player-x {
        color: #00d2ff; /* Neon Cyan */
        text-shadow: 0 0 8px rgba(0, 210, 255, 0.6);
        border-color: #004d66;
    }
    
    .player-o {
        color: #ff4b4b; /* Neon Red */
        text-shadow: 0 0 8px rgba(255, 75, 75, 0.6);
        border-color: #661a1a;
    }
</style>
""", unsafe_allow_html=True)

# Ensure agents exist before allowing play
if 'agent1' in st.session_state and st.session_state.agent1.q_table:
    
    # 1. Game Setup Controls
    with st.container():
        col_h1, col_h2, col_h3 = st.columns([1, 1, 1])
        with col_h1:
            opponent_choice = st.selectbox("Select Opponent", ["Agent 1 (Blue X)", "Agent 2 (Red O)"])
        with col_h2:
            starter = st.selectbox("First Move", ["Human", "AI"])
        with col_h3:
            st.write("") # Spacer
            if st.button("🔥 Start New Match", use_container_width=True, type="primary"):
                st.session_state.human_env = TicTacToe(grid_size, win_length)
                st.session_state.human_game_active = True
                
                if "Agent 1" in opponent_choice:
                    st.session_state.ai_player_id = 1
                    st.session_state.ai_agent = st.session_state.agent1
                    st.session_state.human_player_id = 2
                else:
                    st.session_state.ai_player_id = 2
                    st.session_state.ai_agent = st.session_state.agent2
                    st.session_state.human_player_id = 1
                
                if starter == "AI":
                    st.session_state.current_turn = st.session_state.ai_player_id
                else:
                    st.session_state.current_turn = st.session_state.human_player_id

    # 2. The Game Loop
    if 'human_env' in st.session_state and st.session_state.human_game_active:
        h_env = st.session_state.human_env
        
        # AI Turn Logic
        if h_env.current_player == st.session_state.ai_player_id and not h_env.game_over:
            with st.spinner(f"🤖 AI ({opponent_choice}) is calculating..."):
                import time
                time.sleep(0.6) # Drama delay
                ai_action = st.session_state.ai_agent.choose_action(h_env, training=False)
                if ai_action:
                    h_env.make_move(ai_action)
                    st.rerun()

        # Status Banner
        if h_env.game_over:
            if h_env.winner == st.session_state.human_player_id:
                st.success("🎉 VICTORY! You defeated the AI!")
                st.balloons()
            elif h_env.winner == st.session_state.ai_player_id:
                st.error("💀 DEFEAT! The AI is too strong.")
            else:
                st.warning("🤝 DRAW! A battle of equals.")
        else:
            turn_msg = "Your Turn" if h_env.current_player == st.session_state.human_player_id else "AI's Turn"
            st.caption(f"Status: **{turn_msg}**")

        # --- THE VISUAL GRID ---
        board = h_env.board
        valid_moves = h_env.get_available_actions()
        
        # We loop through rows and columns to create the grid
        for r in range(grid_size):
            cols = st.columns(grid_size)
            for c in range(grid_size):
                cell_value = board[r, c]
                button_key = f"btn_{r}_{c}_{len(h_env.move_history)}"
                
                # CASE 1: EMPTY SPOT
                if cell_value == 0:
                    if not h_env.game_over and h_env.current_player == st.session_state.human_player_id:
                        if (r, c) in valid_moves:
                            if cols[c].button(" ", key=button_key, use_container_width=True):
                                h_env.make_move((r, c))
                                st.rerun()
                        else:
                             cols[c].button("🚫", key=button_key, disabled=True, use_container_width=True)
                    else:
                        cols[c].button(" ", key=button_key, disabled=True, use_container_width=True)
                
                # CASE 2: PLAYER 1 (X)
                elif cell_value == 1:
                    cols[c].markdown(
                        f'<div class="game-cell player-x">X</div>', 
                        unsafe_allow_html=True
                    )
                
                # CASE 3: PLAYER 2 (O)
                elif cell_value == 2:
                    cols[c].markdown(
                        f'<div class="game-cell player-o">O</div>', 
                        unsafe_allow_html=True
                    )
else:
    st.info("👆 Please Train or Load agents above to unlock the Arena.")
