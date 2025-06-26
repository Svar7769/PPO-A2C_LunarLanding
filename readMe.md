\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{listings} % For code blocks
\usepackage{hyperref} % For clickable links
\usepackage{geometry} % For page margins
\usepackage{enumitem} % For custom list indents

% Set page margins
\geometry{a4paper, margin=1in}

% Listing style for Python code
\lstset{
    basicstyle=\ttfamily\small,
    breaklines=true,
    frame=single,
    showstringspaces=false,
    keywordhighlighting=true,
    numbers=left,
    numberstyle=\tiny\color{gray},
    xleftmargin=1em,
    tabsize=4,
    captionpos=b,
    escapeinside={(*@}{@*)},
}

\begin{document}

\title{Proximal Policy Optimization (PPO) and Advantage Actor-Critic (A2C) Implementation}
\author{} % You can add your name here if desired
\date{} % Omit date or use \today for current date
\maketitle

This project presents clean, modular implementations of two prominent reinforcement learning algorithms: Proximal Policy Optimization (PPO) and Advantage Actor-Critic (A2C). Applied to continuous control tasks within the OpenAI Gymnasium environment (specifically \texttt{LunarLanderContinuous-v3}), this project aims to provide a foundational understanding of core RL mechanics and demonstrate their practical application on challenging benchmark environments.

\section*{Features}

\begin{itemize}
    \item \textbf{PPO Algorithm}: Full implementation of the PPO algorithm including:
    \begin{itemize}
        \item Actor (Policy) Network and Critic (Value) Network in PyTorch.
        \item Generalized Advantage Estimation (GAE) for stable advantage calculation.
        \item Clipped Surrogate Objective for robust policy updates.
        \item Entropy regularization to encourage exploration.
    \end{itemize}
    \item \textbf{A2C Algorithm (Baseline)}: A direct implementation of the Advantage Actor-Critic algorithm, sharing common components with PPO but utilizing the standard policy gradient objective and typically performing a single optimization epoch per data collection.
    \item \textbf{Continuous Action Space Support}: Designed to handle environments with continuous action spaces (e.g., \texttt{LunarLanderContinuous-v3}).
    \item \textbf{Rollout Buffer}: Efficient storage and processing of collected trajectories, common to both on-policy algorithms.
    \item \textbf{TensorBoard Logging}: Comprehensive logging of training metrics (rewards, lengths, losses, entropy, clip fraction for PPO) for easy monitoring and analysis, with separate log directories per algorithm.
    \item \textbf{Model Saving and Loading}: Functionality to save and load trained actor and critic network weights for both algorithms.
    \item \textbf{Reproducibility}: Seed management for environments and PyTorch/NumPy.
    \item \textbf{Enhanced Rendering}: Proper setup for visualizing trained agents in the environment.
\end{itemize}

\section*{Project Structure}

\begin{lstlisting}[language=bash, caption=Project Directory Structure]
ppo_project/
├── main.py             # Main script to run training/evaluation, handles algorithm selection
├── config.py           # Stores all hyperparameters and configurable settings for both algorithms
├── ppo_agent.py        # PPO agent class (orchestrates training, data collection, updates)
├── a2c_agent.py        # A2C agent class (baseline implementation)
├── networks.py         # Actor and Critic neural network definitions (PyTorch), shared by both agents
├── buffer.py           # Rollout buffer for storing experiences and computing advantages/returns, shared
├── utils.py            # Helper functions (weight initialization, seeding, env creation, with render_mode)
├── requirements.txt    # Python dependencies
├── runs/               # Directory for TensorBoard logs (created automatically for each run)
└── trained_models/     # Directory to save trained model checkpoints (created automatically)
\end{lstlisting}

\section*{Setup and Installation}

\subsection*{Prerequisites}
\begin{itemize}
    \item Python 3.10
\end{itemize}

\subsection*{Steps}

\begin{enumerate}
    \item \textbf{Clone the repository} (or create the project structure manually):
    \begin{lstlisting}[language=bash]
git clone https://github.com/Svar7769/PPO-A2C_LunarLanding.git
cd ppo_project
    \end{lstlisting}
    \textit{If not using Git, manually create the \texttt{ppo\_project} directory and all the \texttt{.py} files within it as provided in the project structure.}

    \item \textbf{Create and activate a Python virtual environment}:
    \begin{lstlisting}[language=bash]
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
    \end{lstlisting}

    \item \textbf{Install dependencies}:
    \begin{lstlisting}[language=bash]
pip install -r requirements.txt
    \end{lstlisting}
\end{enumerate}

\section*{Usage}

\subsection*{Training an Agent (PPO or A2C)}

Use the \texttt{--algo} argument to specify which algorithm to train:

\begin{itemize}
    \item \textbf{To train PPO}:
    \begin{lstlisting}[language=bash]
python main.py --algo ppo
    \end{lstlisting}
    \item \textbf{To train A2C}:
    \begin{lstlisting}[language=bash]
python main.py --algo a2c
    \end{lstlisting}
\end{itemize}
You will see console output detailing episode rewards and lengths as training progresses. Model checkpoints will be saved periodically to the \texttt{trained\_models/} directory according to \texttt{SAVE\_INTERVAL} in \texttt{config.py}.

\subsection*{Monitoring Training with TensorBoard}

While training is running, you can monitor the learning progress visually:

\begin{enumerate}
    \item Open a new terminal (keep the training terminal running).
    \item Navigate to the \texttt{ppo\_project} directory.
    \item \textbf{Run TensorBoard}:
    \begin{lstlisting}[language=bash]
tensorboard --logdir runs
    \end{lstlisting}
    \item Open your web browser and go to the address provided by TensorBoard (usually \url{http://localhost:6006}). Here you can view graphs for:
    \begin{itemize}
        \item \texttt{Episode/Reward}
        \item \texttt{Episode/Length}
        \item \texttt{Loss/Actor\_Loss}
        \item \texttt{Loss/Critic\_Loss}
        \item \texttt{Metrics/Entropy}
        \item \texttt{Metrics/Clip\_Fraction} (PPO only)
        \item And others that indicate overall training progress.
    \end{itemize}
    \textit{Note: TensorBoard will display separate curves for PPO and A2C runs if you train both, identifiable by their unique log directory names (e.g., \texttt{...-LunarLanderContinuous-v3} vs \texttt{...-A2C-LunarLanderContinuous-v3}).}
\end{enumerate}

\subsection*{Evaluating a Trained Agent}

After training (or to evaluate a previously saved model), you can use the \texttt{evaluate} method.
First, ensure you have a trained model saved in the \texttt{trained\_models/} directory.

\begin{enumerate}
    \item \textbf{Modify \texttt{main.py}}:
    Comment out \texttt{agent.train()} and uncomment the \texttt{agent.evaluate()} call. If you want to load a specific model, add a \texttt{load\_model} call before evaluate.
    \begin{lstlisting}[language=python, caption=\texttt{main.py} modification for evaluation]
# ppo_project/main.py (example modification for evaluation)

# ... (config and directory creation) ...

def main():
    # ... (argument parsing and config setup) ...

    # Decide which agent type to instantiate (PPO or A2C) based on --algo
    # The config parameters for N_EPOCHS and MINIBATCH_SIZE will be set
    # by the main function based on the chosen algorithm.
    if args.algo == "ppo":
        config.N_EPOCHS = config.PPO_N_EPOCHS
        config.MINIBATCH_SIZE = config.PPO_MINIBATCH_SIZE
        agent = PPOAgent(config)
    elif args.algo == "a2c":
        config.N_EPOCHS = config.A2C_N_EPOCHS
        config.MINIBATCH_SIZE = config.A2C_MINIBATCH_SIZE
        config.ACTOR_LR = config.A2C_ACTOR_LR # Use A2C specific LR
        config.CRITIC_LR = config.A2C_CRITIC_LR # Use A2C specific LR
        agent = A2CAgent(config)
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}.")

    # agent.train() # Comment this out for evaluation

    # To load a specific model for evaluation:
    # Replace 'a2c' or 'ppo' and '1000000' with your saved model's name and step
    # model_path = os.path.join(config.SAVE_DIR, f"{args.algo}_agent_step_1000000.pth")
    # agent.load_model(model_path)

    # Evaluate the agent (set render=True to visualize)
    # Note: Rendering for Box2D environments (like LunarLander) requires an X server on Linux/WSL.
    agent.evaluate(num_episodes=10, render=True)
# ...
    \end{lstlisting}

    \item Run \texttt{python main.py --algo <your\_algorithm>} again (e.g., \texttt{--algo ppo} or \texttt{--algo a2c}).
\end{enumerate}

\subsection*{Customizing Hyperparameters}

All hyperparameters are defined in \texttt{config.py}. Separate parameters (\texttt{PPO\_N\_EPOCHS}, \texttt{A2C\_N\_EPOCHS}, etc.) are provided for each algorithm, allowing you to fine-tune them independently. You are encouraged to experiment with values like learning rates, \texttt{CLIP\_EPSILON} (for PPO), \texttt{ENTROPY\_COEFF}, \texttt{GAE\_LAMBDA}, and network \texttt{HIDDEN\_SIZES} to optimize performance.

\section*{Results}

\textit{(This section is for you to fill in after you've run your training for both PPO and A2C!)}

\begin{itemize}
    \item \textbf{PPO Performance}: Describe the observed performance on \texttt{LunarLanderContinuous-v3}. What was the typical maximum reward achieved? How many steps/episodes did it take to converge?
    \item \textbf{A2C Performance}: Describe the baseline performance. How did it compare to PPO in terms of maximum reward, stability, and convergence speed?
    \item \textbf{Comparison Insights}: Discuss the practical differences you observed between PPO and A2C based on your training runs and TensorBoard graphs. Why do you think PPO performed better (or differently)?
    \item \textbf{Visualizations}: Include screenshots of your TensorBoard graphs, specifically comparing the \texttt{Episode/Reward} curves for PPO and A2C on the same plot. You can also include plots of Actor/Critic losses, Entropy, etc.
    \item \textit{(Optional but highly recommended)} If you set up rendering, include a short GIF or video of your well-trained agent playing the game.
\end{itemize}

\section*{Future Work and Enhancements}

This project serves as a robust foundation. Here are areas for further exploration to demonstrate advanced research capabilities:

\begin{itemize}
    \item \textbf{In-depth Hyperparameter Sensitivity Analysis}: Conduct systematic grid/random searches for hyperparameter optimization and analyze the results across multiple random seeds for both algorithms.
    \item \textbf{More Challenging Environments}: Test the PPO/A2C implementations on more complex continuous control environments such as \texttt{BipedalWalker-v3} or even MuJoCo environments (if you have access), requiring further tuning and robustness.
    \item \textbf{Distributed Training}: Explore how to scale the data collection process using multiple parallel environments (e.g., using \texttt{gymnasium.vector.make}) to speed up training.
    \item \textbf{Performance Optimizations}: Investigate techniques like observation normalization, reward scaling, or different network architectures to further improve performance and stability.
    \item \textbf{Code Refinements}: Add more comprehensive error handling, type hinting, and unit tests.
\end{itemize}

\end{document}
