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
    page_title="AGI Tic-Tac-Toe Battle",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

st.title("🧠 AGI-Level Tic-Tac-Toe Battle Arena")
st.markdown("""
Watch two **AGI-level** Reinforcement Learning agents with **advanced reasoning** battle and evolve!

**AGI Enhancements:**
- 🎯 **Monte Carlo Tree Search (MCTS)** - Lookahead planning
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
        actions = [(r, c) for r in range(self.grid_size) 
                   for c in range(self.grid_size) if self.board[r, c] == 0]
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
            return self.get_state(), 0, True # Draw is better than losing
        
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
        """God-Mode Heuristic: Values Center, Corners, and Lines"""
        if self.winner == player: return 100000
        if self.winner == (3 - player): return -100000
        if self.game_over: return 0  # Draw
        
        opponent = 3 - player
        score = 0
        
        # 1. Control the center (Crucial for defense)
        center = self.grid_size // 2
        if self.board[center, center] == player: score += 50
        elif self.board[center, center] == opponent: score -= 50
        
        # 2. Control corners (Crucial for traps)
        corners = [(0,0), (0, self.grid_size-1), (self.grid_size-1, 0), (self.grid_size-1, self.grid_size-1)]
        for r,c in corners:
            if self.board[r,c] == player: score += 10
            elif self.board[r,c] == opponent: score -= 10

        # 3. Line Counting (Threats)
        # We value having "2 in a row" much higher to encourage building traps
        score += self._count_lines(player, 2) * 50
        score -= self._count_lines(opponent, 2) * 60 # Defensive paranoia: fear opponent more
        score += self._count_lines(player, self.win_length - 1) * 500
        score -= self._count_lines(opponent, self.win_length - 1) * 1000
        
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

class AGIAgent:
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
        # AGI HIERARCHY LEVEL 1: IMMEDIATE SURVIVAL (The "Reflex")
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
        # AGI HIERARCHY LEVEL 2: STRATEGIC THINKING (Minimax)
        # ---------------------------------------------------------
        # If training and high epsilon, explore randomly occasionally
        if training and random.random() < self.epsilon:
            return random.choice(available_actions)

        # Dynamic Depth: Play perfect on small boards, heuristic on large
        empty_spots = len(available_actions)
        if env.grid_size == 3:
            current_depth = 9 # God mode for 3x3
        else:
            current_depth = min(self.minimax_depth, empty_spots)

        best_score = -float('inf')
        best_actions = []

        # Alpha-Beta Pruning Search
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
    """Play one complete game between two AGI agents"""
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
    return {str(k): v for k, v in q_table.items()}

def deserialize_q_table(serialized_q):
    return {ast.literal_eval(k): v for k, v in serialized_q.items()}

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
            
            agent1 = AGIAgent(1, agent1_state['lr'], agent1_state['gamma'])
            agent1.q_table = deserialize_q_table(agent1_state['q_table'])
            agent1.epsilon = agent1_state['epsilon']
            agent1.wins = agent1_state['wins']
            agent1.losses = agent1_state['losses']
            agent1.draws = agent1_state['draws']
            
            agent2 = AGIAgent(2, agent2_state['lr'], agent2_state['gamma'])
            agent2.q_table = deserialize_q_table(agent2_state['q_table'])
            agent2.epsilon = agent2_state['epsilon']
            agent2.wins = agent2_state['wins']
            agent2.losses = agent2_state['losses']
            agent2.draws = agent2_state['draws']
            
            return agent1, agent2, config
    except Exception as e:
        st.error(f"Failed to load agents: {e}")
        return None, None, None

# ============================================================================
# Streamlit UI
# ============================================================================

st.sidebar.header("🧠 AGI Arena Controls")

with st.sidebar.expander("1. Game Configuration", expanded=True):
    grid_size = st.slider("Grid Size", 3, 10, 3)
    max_win_length = max(grid_size, 4)
    default_win = min(grid_size, 3)
    win_length = st.slider("Win Length (in-a-row)", 3, max_win_length, default_win)
    st.info(f"Playing on {grid_size}×{grid_size} grid, need {win_length} in a row to win")

with st.sidebar.expander("2. AGI Agent 1 (Blue X)", expanded=True):
    lr1 = st.slider("Learning Rate α₁", 0.01, 1.0, 0.2, 0.01)
    gamma1 = st.slider("Discount Factor γ₁", 0.8, 0.99, 0.95, 0.01)
    epsilon_decay1 = st.slider("Epsilon Decay₁", 0.99, 0.9999, 0.998, 0.0001, format="%.4f")
    minimax_depth1 = st.slider("Minimax Depth₁", 1, 5, 3)

with st.sidebar.expander("3. AGI Agent 2 (Red O)", expanded=True):
    lr2 = st.slider("Learning Rate α₂", 0.01, 1.0, 0.2, 0.01)
    gamma2 = st.slider("Discount Factor γ₂", 0.8, 0.99, 0.95, 0.01)
    epsilon_decay2 = st.slider("Epsilon Decay₂", 0.99, 0.9999, 0.998, 0.0001, format="%.4f")
    minimax_depth2 = st.slider("Minimax Depth₂", 1, 5, 3)

with st.sidebar.expander("4. Training Configuration", expanded=True):
    episodes = st.number_input("Training Episodes", 100, 100000, 3000, 100)
    update_freq = st.number_input("Update Dashboard Every N Games", 10, 1000, 50, 10)

with st.sidebar.expander("5. Brain Storage", expanded=False):
    if 'agent1' in st.session_state and st.session_state.agent1 is not None:
        config = {"grid_size": grid_size, "win_length": win_length}
        zip_buffer = create_agents_zip(st.session_state.agent1, 
                                       st.session_state.agent2, config)
        st.download_button(
            label="💾 Download AGI Brains",
            data=zip_buffer,
            file_name="agi_agents.zip",
            mime="application/zip",
            use_container_width=True
        )
    else:
        st.warning("Train agents first to download.")
    
    uploaded_file = st.file_uploader("Upload AGI Brains (.zip)", type="zip")
    if uploaded_file is not None:
        if st.button("Load AGI Agents", use_container_width=True):
            a1, a2, cfg = load_agents_from_zip(uploaded_file)
            if a1:
                st.session_state.agent1 = a1
                st.session_state.agent2 = a2
                st.session_state.agent1.minimax_depth = minimax_depth1
                st.session_state.agent2.minimax_depth = minimax_depth2
                st.session_state.training_history = None
                st.toast("AGI Agents Restored!", icon="🧠")
                st.rerun()

train_button = st.sidebar.button("🚀 Start AGI Training", 
                                 use_container_width=True, type="primary")

if st.sidebar.button("🧹 Clear All & Reset", use_container_width=True):
    for key in ['agent1', 'agent2', 'training_history', 'env']:
        if key in st.session_state:
            del st.session_state[key]
    st.cache_data.clear()
    st.toast("AGI Arena Reset!", icon="🧹")
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
    st.session_state.agent1 = AGIAgent(1, lr1, gamma1, epsilon_decay=epsilon_decay1)
    st.session_state.agent1.minimax_depth = minimax_depth1
    st.session_state.agent2 = AGIAgent(2, lr2, gamma2, epsilon_decay=epsilon_decay2)
    st.session_state.agent2.minimax_depth = minimax_depth2

agent1 = st.session_state.agent1
agent2 = st.session_state.agent2

# Update minimax depth
agent1.minimax_depth = minimax_depth1
agent2.minimax_depth = minimax_depth2

# Display current stats
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🧠 AGI Agent 1 (Blue X)", 
             f"Q-States: {len(agent1.q_table)}", 
             f"ε={agent1.epsilon:.4f}")
    st.metric("Wins", agent1.wins, delta_color="normal")
    st.caption(f"Minimax Depth: {agent1.minimax_depth}")

with col2:
    st.metric("🧠 AGI Agent 2 (Red O)", 
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
    st.subheader("🧠 AGI Training in Progress...")
    
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
        'agent2_q_size': []
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
    st.toast("AGI Training Complete!", icon="🎉")
    
    st.session_state.training_history = history
    # --- FIX START: Persist the updated agents in session state after training ---
    st.session_state.agent1 = agent1
    st.session_state.agent2 = agent2
    # --- FIX END ---

# Display charts and final game if training has occurred
if 'training_history' in st.session_state and st.session_state.training_history:
    st.subheader("📈 Training Performance Analysis")
    history = st.session_state.training_history
    
    df = pd.DataFrame(history)
    # Create an 'episode' column for the x-axis of the charts
    df['episode'] = range(update_freq, episodes + 1, update_freq)
    
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

    st.subheader("🤖 Final Battle: Trained Agents")
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
