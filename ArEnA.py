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

# ============================================================================
# Page Config and Initial Setup
# ============================================================================
st.set_page_config(
    page_title="RL Tic-Tac-Toe Battle",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚔️"
)

st.title("⚔️ RL Tic-Tac-Toe Battle Arena")
st.markdown("""
Watch two Reinforcement Learning agents **battle, learn, and evolve** through self-play!

1. **Configure Game**: Set grid size and game rules in the sidebar
2. **Train Agents**: Let them fight and learn from thousands of games
3. **Watch Evolution**: See how strategies develop over time
4. **Test Agents**: Watch trained agents play optimally
""")

# ============================================================================
# Tic-Tac-Toe Environment
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
        return self.get_state()
    
    def get_state(self):
        """Returns immutable state representation"""
        return tuple(self.board.flatten())
    
    def get_available_actions(self):
        """Returns list of available positions"""
        return [(i, j) for i in range(self.grid_size) 
                for j in range(self.grid_size) if self.board[i, j] == 0]
    
    def make_move(self, position):
        """Execute a move and return (next_state, reward, done)"""
        if self.game_over:
            return self.get_state(), 0, True
        
        i, j = position
        if self.board[i, j] != 0:
            # Invalid move - punish heavily
            return self.get_state(), -10, True
        
        self.board[i, j] = self.current_player
        
        # Check for win
        if self._check_win(self.current_player):
            self.game_over = True
            self.winner = self.current_player
            reward = 10  # Win reward (reduced from 1 to differentiate better)
            return self.get_state(), reward, True
        
        # Check for draw
        if len(self.get_available_actions()) == 0:
            self.game_over = True
            self.winner = 0
            reward = 0.1  # Small draw reward (reduced from 0.5 to encourage winning)
            return self.get_state(), reward, True
        
        # Game continues - small penalty for each move to encourage faster wins
        self.current_player = 3 - self.current_player  # Switch between 1 and 2
        return self.get_state(), -0.01, False
    
    def _check_win(self, player):
        """Check if player has won"""
        board = self.board
        n = self.grid_size
        w = self.win_length
        
        # Check rows
        for i in range(n):
            for j in range(n - w + 1):
                if all(board[i, j+k] == player for k in range(w)):
                    return True
        
        # Check columns
        for i in range(n - w + 1):
            for j in range(n):
                if all(board[i+k, j] == player for k in range(w)):
                    return True
        
        # Check diagonals (top-left to bottom-right)
        for i in range(n - w + 1):
            for j in range(n - w + 1):
                if all(board[i+k, j+k] == player for k in range(w)):
                    return True
        
        # Check diagonals (top-right to bottom-left)
        for i in range(n - w + 1):
            for j in range(w - 1, n):
                if all(board[i+k, j-k] == player for k in range(w)):
                    return True
        
        return False

# ============================================================================
# RL Agent Class
# ============================================================================

class QLearningAgent:
    def __init__(self, player_id, lr=0.1, gamma=0.95, epsilon=1.0, 
                 epsilon_decay=0.9995, epsilon_min=0.01):
        self.player_id = player_id
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.q_table = {}
        self.init_q_value = 0.0
        
        # Statistics
        self.episode_rewards = []
        self.wins = 0
        self.losses = 0
        self.draws = 0
    
    def get_q_value(self, state, action):
        return self.q_table.get((state, action), self.init_q_value)
    
    def choose_action(self, env, training=True):
        """Choose action using epsilon-greedy strategy"""
        available_actions = env.get_available_actions()
        
        if not available_actions:
            return None
        
        state = env.get_state()
        
        if training and random.random() < self.epsilon:
            return random.choice(available_actions)
        
        # Greedy action selection
        q_values = [self.get_q_value(state, action) for action in available_actions]
        max_q = max(q_values)
        best_actions = [action for action, q in zip(available_actions, q_values) if q == max_q]
        return random.choice(best_actions)
    
    def update_q_value(self, state, action, reward, next_state, next_available_actions):
        """Q-Learning update"""
        current_q = self.get_q_value(state, action)
        
        if next_available_actions:
            max_next_q = max([self.get_q_value(next_state, a) for a in next_available_actions])
        else:
            max_next_q = 0
        
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action)] = new_q
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def reset_stats(self):
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.episode_rewards = []

# ============================================================================
# Training System
# ============================================================================

def play_game(env, agent1, agent2, training=True):
    """Play one complete game between two agents"""
    env.reset()
    game_history = []  # Store (state, action, player) tuples
    
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
        
        if done:
            # Game ended - assign rewards
            if env.winner == 1:
                agent1.wins += 1
                agent2.losses += 1
                # Update both agents - winner gets +10, loser gets -5
                if training:
                    _update_agent_from_history(agent1, game_history, 1, 10)
                    _update_agent_from_history(agent2, game_history, 2, -5)
            elif env.winner == 2:
                agent2.wins += 1
                agent1.losses += 1
                if training:
                    _update_agent_from_history(agent1, game_history, 1, -5)
                    _update_agent_from_history(agent2, game_history, 2, 10)
            else:  # Draw
                agent1.draws += 1
                agent2.draws += 1
                if training:
                    # Draws get very small reward - we want to discourage this
                    _update_agent_from_history(agent1, game_history, 1, 0.1)
                    _update_agent_from_history(agent2, game_history, 2, 0.1)
    
    return env.winner

def _update_agent_from_history(agent, history, player_id, final_reward):
    """Update agent's Q-values based on game outcome with intermediate rewards"""
    # Filter history for this agent's moves
    agent_moves = [(s, a) for s, a, p in history if p == player_id]
    
    # Create a potential win/block detector for intermediate rewards
    env_temp = TicTacToe(3, 3)  # Temporary env for checking
    
    # Backward update from end to start
    for i in range(len(agent_moves) - 1, -1, -1):
        state, action = agent_moves[i]
        
        # Calculate base reward
        if i == len(agent_moves) - 1:
            # Last move gets the full final reward
            reward = final_reward
        else:
            # Intermediate moves get discounted reward
            reward = final_reward * (agent.gamma ** (len(agent_moves) - 1 - i))
            
            # STRATEGIC BONUS: Reward creating threats or blocking opponent
            # Reconstruct board from state
            board = np.array(state).reshape(env_temp.grid_size, env_temp.grid_size)
            
            # Check if this move created a winning threat (2 in a row)
            row, col = action
            if _creates_threat(board, row, col, player_id, env_temp.grid_size):
                reward += 0.5  # Bonus for creating threats
            
            # Check if this move blocked opponent's threat
            opponent_id = 3 - player_id
            if _blocks_threat(board, row, col, opponent_id, env_temp.grid_size):
                reward += 0.3  # Bonus for defensive plays
        
        if i < len(agent_moves) - 1:
            next_state, next_action = agent_moves[i + 1]
            next_q = agent.get_q_value(next_state, next_action)
        else:
            next_q = 0
        
        current_q = agent.get_q_value(state, action)
        new_q = current_q + agent.lr * (reward + agent.gamma * next_q - current_q)
        agent.q_table[(state, action)] = new_q

def _creates_threat(board, row, col, player, grid_size):
    """Check if a move creates a winning threat (2 in a row with empty third)"""
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for dr, dc in directions:
        count = 1
        empty = 0
        
        # Check both directions
        for direction in [1, -1]:
            r, c = row + dr * direction, col + dc * direction
            for _ in range(2):
                if 0 <= r < grid_size and 0 <= c < grid_size:
                    if board[r, c] == player:
                        count += 1
                    elif board[r, c] == 0:
                        empty += 1
                        break
                    else:
                        break
                else:
                    break
                r += dr * direction
                c += dc * direction
        
        if count >= 2 and empty >= 1:
            return True
    
    return False

def _blocks_threat(board, row, col, opponent, grid_size):
    """Check if move blocks opponent's threat"""
    # Temporarily place opponent piece to see if they had a threat
    temp_board = board.copy()
    temp_board[row, col] = opponent
    return _creates_threat(temp_board, row, col, opponent, grid_size)

# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_board(board, title="Game Board"):
    """Create matplotlib figure of the board"""
    fig, ax = plt.subplots(figsize=(6, 6))
    n = board.shape[0]
    
    # Draw grid
    for i in range(n + 1):
        ax.plot([0, n], [i, i], 'k-', linewidth=2)
        ax.plot([i, i], [0, n], 'k-', linewidth=2)
    
    # Draw X's and O's
    for i in range(n):
        for j in range(n):
            if board[i, j] == 1:
                # Draw X
                ax.plot([j + 0.2, j + 0.8], [n - i - 0.2, n - i - 0.8], 
                       'b-', linewidth=4)
                ax.plot([j + 0.2, j + 0.8], [n - i - 0.8, n - i - 0.2], 
                       'b-', linewidth=4)
            elif board[i, j] == 2:
                # Draw O
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
# Save/Load Functions
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
            
            agent1 = QLearningAgent(1, agent1_state['lr'], agent1_state['gamma'])
            agent1.q_table = deserialize_q_table(agent1_state['q_table'])
            agent1.epsilon = agent1_state['epsilon']
            agent1.wins = agent1_state['wins']
            agent1.losses = agent1_state['losses']
            agent1.draws = agent1_state['draws']
            
            agent2 = QLearningAgent(2, agent2_state['lr'], agent2_state['gamma'])
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

st.sidebar.header("⚙️ Battle Arena Controls")

with st.sidebar.expander("1. Game Configuration", expanded=True):
    grid_size = st.slider("Grid Size", 3, 10, 3)
    # Ensure min and max are different for the slider
    max_win_length = max(grid_size, 4)  # Always at least 4 to avoid min=max
    default_win = min(grid_size, 3)
    win_length = st.slider("Win Length (in-a-row)", 3, max_win_length, default_win)
    st.info(f"Playing on {grid_size}×{grid_size} grid, need {win_length} in a row to win")

with st.sidebar.expander("2. Agent 1 (Blue X) Hyperparameters", expanded=True):
    lr1 = st.slider("Learning Rate α₁", 0.01, 1.0, 0.3, 0.01)
    gamma1 = st.slider("Discount Factor γ₁", 0.8, 0.99, 0.9, 0.01)
    epsilon_decay1 = st.slider("Epsilon Decay₁", 0.99, 0.9999, 0.999, 0.0001, format="%.4f")

with st.sidebar.expander("3. Agent 2 (Red O) Hyperparameters", expanded=True):
    lr2 = st.slider("Learning Rate α₂", 0.01, 1.0, 0.3, 0.01)
    gamma2 = st.slider("Discount Factor γ₂", 0.8, 0.99, 0.9, 0.01)
    epsilon_decay2 = st.slider("Epsilon Decay₂", 0.99, 0.9999, 0.999, 0.0001, format="%.4f")

with st.sidebar.expander("4. Training Configuration", expanded=True):
    episodes = st.number_input("Training Episodes", 100, 100000, 5000, 100)
    update_freq = st.number_input("Update Dashboard Every N Games", 10, 1000, 100, 10)

with st.sidebar.expander("5. Brain Storage (Save/Load)", expanded=False):
    if 'agent1' in st.session_state and st.session_state.agent1 is not None:
        config = {
            "grid_size": grid_size,
            "win_length": win_length
        }
        zip_buffer = create_agents_zip(st.session_state.agent1, 
                                       st.session_state.agent2, config)
        st.download_button(
            label="💾 Download Both Agents",
            data=zip_buffer,
            file_name="tictactoe_agents.zip",
            mime="application/zip",
            use_container_width=True
        )
    else:
        st.warning("Train agents first to download.")
    
    uploaded_file = st.file_uploader("Upload Agents (.zip)", type="zip")
    if uploaded_file is not None:
        if st.button("Load Agents", use_container_width=True):
            a1, a2, cfg = load_agents_from_zip(uploaded_file)
            if a1:
                st.session_state.agent1 = a1
                st.session_state.agent2 = a2
                st.session_state.training_history = None
                st.toast("Agents Restored Successfully!", icon="🧠")
                st.rerun()

train_button = st.sidebar.button("⚔️ Start Battle Training", 
                                 use_container_width=True, type="primary")

if st.sidebar.button("🧹 Clear All & Reset", use_container_width=True):
    for key in ['agent1', 'agent2', 'training_history', 'env']:
        if key in st.session_state:
            del st.session_state[key]
    st.cache_data.clear()
    st.toast("Arena Reset!", icon="🧹")
    st.rerun()

# ============================================================================
# Main Area
# ============================================================================

# Initialize environment
if 'env' not in st.session_state:
    st.session_state.env = TicTacToe(grid_size, win_length)

env = st.session_state.env

# Update env if grid size changed
if env.grid_size != grid_size or env.win_length != win_length:
    st.session_state.env = TicTacToe(grid_size, win_length)
    env = st.session_state.env

# Initialize agents
if 'agent1' not in st.session_state:
    st.session_state.agent1 = QLearningAgent(1, lr1, gamma1, epsilon_decay=epsilon_decay1)
    st.session_state.agent2 = QLearningAgent(2, lr2, gamma2, epsilon_decay=epsilon_decay2)

agent1 = st.session_state.agent1
agent2 = st.session_state.agent2

# Display current stats
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Agent 1 (Blue X)", 
             f"Q-Table: {len(agent1.q_table)}", 
             f"ε={agent1.epsilon:.4f}")
    st.metric("Wins", agent1.wins, delta_color="normal")

with col2:
    st.metric("Agent 2 (Red O)", 
             f"Q-Table: {len(agent2.q_table)}", 
             f"ε={agent2.epsilon:.4f}")
    st.metric("Wins", agent2.wins, delta_color="normal")

with col3:
    total_games = agent1.wins + agent1.losses + agent1.draws
    st.metric("Total Games", total_games)
    st.metric("Draws", agent1.draws, delta_color="off")

# Training section
if train_button:
    st.subheader("🥊 Battle in Progress...")
    
    status_container = st.empty()
    progress_bar = st.progress(0)
    
    # Reset stats
    agent1.reset_stats()
    agent2.reset_stats()
    
    # Training history
    history = {
        'agent1_wins': [],
        'agent2_wins': [],
        'draws': [],
        'agent1_epsilon': [],
        'agent2_epsilon': []
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
            
            win_rate_1 = agent1.wins / episode
            win_rate_2 = agent2.wins / episode
            draw_rate = agent1.draws / episode
            
            status_markdown = f"""
            | Metric | Value |
            |---|---|
            | **Episode** | `{episode}` / `{episodes}` |
            | **Agent 1 Win Rate** | `{win_rate_1:.2%}` |
            | **Agent 2 Win Rate** | `{win_rate_2:.2%}` |
            | **Draw Rate** | `{draw_rate:.2%}` |
            | **Agent 1 Q-Table** | `{len(agent1.q_table)}` states |
            | **Agent 2 Q-Table** | `{len(agent2.q_table)}` states |
            | **Agent 1 ε** | `{agent1.epsilon:.4f}` |
            | **Agent 2 ε** | `{agent2.epsilon:.4f}` |
            """
            status_container.markdown(status_markdown)
            progress_bar.progress(episode / episodes)
    
    st.session_state.training_history = history
    st.success(f"Training Complete! {episodes} battles fought.")
    st.rerun()

# Show training graphs
if 'training_history' in st.session_state and st.session_state.training_history:
    st.subheader("📊 Evolution of Battle Performance")
    
    history = st.session_state.training_history
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        episodes_axis = np.arange(len(history['agent1_wins'])) * update_freq
        ax.plot(episodes_axis, history['agent1_wins'], 'b-', label='Agent 1 (X)', linewidth=2)
        ax.plot(episodes_axis, history['agent2_wins'], 'r-', label='Agent 2 (O)', linewidth=2)
        ax.plot(episodes_axis, history['draws'], 'g--', label='Draws', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Count')
        ax.set_title('Win/Draw Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(episodes_axis, history['agent1_epsilon'], 'b-', label='Agent 1 ε', linewidth=2)
        ax.plot(episodes_axis, history['agent2_epsilon'], 'r-', label='Agent 2 ε', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon (Exploration Rate)')
        ax.set_title('Exploration Decay')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# Test section
st.subheader("🎮 Watch Trained Agents Battle")

col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ Play One Game (No Learning)", use_container_width=True):
        env.reset()
        play_game(env, agent1, agent2, training=False)
        
        st.session_state.test_board = env.board.copy()
        st.session_state.test_winner = env.winner
        st.rerun()

with col2:
    num_test_games = st.number_input("Number of Test Games", 1, 1000, 10, 1)
    if st.button(f"▶️ Play {num_test_games} Games (Statistics)", use_container_width=True):
        test_results = {'agent1': 0, 'agent2': 0, 'draw': 0}
        
        for _ in range(num_test_games):
            env.reset()
            winner = play_game(env, agent1, agent2, training=False)
            if winner == 1:
                test_results['agent1'] += 1
            elif winner == 2:
                test_results['agent2'] += 1
            else:
                test_results['draw'] += 1
        
        st.session_state.test_results = test_results
        st.rerun()

# Display test results
if 'test_board' in st.session_state:
    winner = st.session_state.test_winner
    if winner == 1:
        result_text = "🔵 Agent 1 (X) Wins!"
    elif winner == 2:
        result_text = "🔴 Agent 2 (O) Wins!"
    else:
        result_text = "🤝 Draw!"
    
    fig = visualize_board(st.session_state.test_board, title=result_text)
    st.pyplot(fig)

if 'test_results' in st.session_state:
    results = st.session_state.test_results
    st.subheader("Test Results Summary")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Agent 1 Wins", results['agent1'])
    col2.metric("Agent 2 Wins", results['agent2'])
    col3.metric("Draws", results['draw'])
