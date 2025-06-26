Proximal Policy Optimization (PPO) and Advantage Actor-Critic (A2C) Implementation

This project presents clean, modular implementations of two prominent reinforcement learning algorithms: Proximal Policy Optimization (PPO) and Advantage Actor-Critic (A2C). Applied to continuous control tasks within the OpenAI Gymnasium environment (specifically LunarLanderContinuous-v3), this project aims to provide a foundational understanding of core RL mechanics and demonstrate their practical application on challenging benchmark environments.
Features

    PPO Algorithm: Full implementation of the PPO algorithm including:

        Actor (Policy) Network and Critic (Value) Network in PyTorch.

        Generalized Advantage Estimation (GAE) for stable advantage calculation.

        Clipped Surrogate Objective for robust policy updates.

        Entropy regularization to encourage exploration.

    A2C Algorithm (Baseline): A direct implementation of the Advantage Actor-Critic algorithm, sharing common components with PPO but utilizing the standard policy gradient objective and typically performing a single optimization epoch per data collection.

    Continuous Action Space Support: Designed to handle environments with continuous action spaces (e.g., LunarLanderContinuous-v3).

    Rollout Buffer: Efficient storage and processing of collected trajectories, common to both on-policy algorithms.

    TensorBoard Logging: Comprehensive logging of training metrics (rewards, lengths, losses, entropy, clip fraction for PPO) for easy monitoring and analysis, with separate log directories per algorithm.

    Model Saving and Loading: Functionality to save and load trained actor and critic network weights for both algorithms.

    Reproducibility: Seed management for environments and PyTorch/NumPy.

    Enhanced Rendering: Proper setup for visualizing trained agents in the environment.

Project Structure

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

Setup and Installation
Prerequisites

    Python 3.10

Steps

    Clone the repository (or create the project structure manually):

    git clone https://github.com/Svar7769/PPO-A2C_LunarLanding.git
    cd ppo_project

    If not using Git, manually create the ppo_project directory and all the .py files within it as provided in the project structure.

    Create and activate a Python virtual environment:

    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate

    Install dependencies:

    pip install -r requirements.txt

Usage
Training an Agent (PPO or A2C)

Use the --algo argument to specify which algorithm to train:

    To train PPO:

    python main.py --algo ppo

    To train A2C:

    python main.py --algo a2c

You will see console output detailing episode rewards and lengths as training progresses. Model checkpoints will be saved periodically to the trained_models/ directory according to SAVE_INTERVAL in config.py.
Monitoring Training with TensorBoard

While training is running, you can monitor the learning progress visually:

    Open a new terminal (keep the training terminal running).

    Navigate to the ppo_project directory.

    Run TensorBoard:

    tensorboard --logdir runs

    Open your web browser and go to the address provided by TensorBoard (usually http://localhost:6006). Here you can view graphs for:

        Episode/Reward

        Episode/Length

        Loss/Actor_Loss

        Loss/Critic_Loss

        Metrics/Entropy

        Metrics/Clip_Fraction (PPO only)

        And others that indicate overall training progress.

    Note: TensorBoard will display separate curves for PPO and A2C runs if you train both, identifiable by their unique log directory names (e.g., ...-LunarLanderContinuous-v3 vs ...-A2C-LunarLanderContinuous-v3).

Evaluating a Trained Agent

After training (or to evaluate a previously saved model), you can use the evaluate method.
First, ensure you have a trained model saved in the trained_models/ directory.

    Modify main.py:
    Comment out agent.train() and uncomment the agent.evaluate() call. If you want to load a specific model, add a load_model call before evaluate.

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

    Run python main.py --algo <your_algorithm> again (e.g., --algo ppo or --algo a2c).

Customizing Hyperparameters

All hyperparameters are defined in config.py. Separate parameters (PPO_N_EPOCHS, A2C_N_EPOCHS, etc.) are provided for each algorithm, allowing you to fine-tune them independently. You are encouraged to experiment with values like learning rates, CLIP_EPSILON (for PPO), ENTROPY_COEFF, GAE_LAMBDA, and network HIDDEN_SIZES to optimize performance.
Results

(This section is for you to fill in after you've run your training for both PPO and A2C!)

    PPO Performance: Describe the observed performance on LunarLanderContinuous-v3. What was the typical maximum reward achieved? How many steps/episodes did it take to converge?

    A2C Performance: Describe the baseline performance. How did it compare to PPO in terms of maximum reward, stability, and convergence speed?

    Comparison Insights: Discuss the practical differences you observed between PPO and A2C based on your training runs and TensorBoard graphs. Why do you think PPO performed better (or differently)?

    Visualizations: Include screenshots of your TensorBoard graphs, specifically comparing the Episode/Reward curves for PPO and A2C on the same plot. You can also include plots of Actor/Critic losses, Entropy, etc.

    (Optional but highly recommended) If you set up rendering, include a short GIF or video of your well-trained agent playing the game.

Future Work and Enhancements

This project serves as a robust foundation. Here are areas for further exploration to demonstrate advanced research capabilities:

    In-depth Hyperparameter Sensitivity Analysis: Conduct systematic grid/random searches for hyperparameter optimization and analyze the results across multiple random seeds for both algorithms.

    More Challenging Environments: Test the PPO/A2C implementations on more complex continuous control environments such as BipedalWalker-v3 or even MuJoCo environments (if you have access), requiring further tuning and robustness.

    Distributed Training: Explore how to scale the data collection process using multiple parallel environments (e.g., using gymnasium.vector.make) to speed up training.

    Performance Optimizations: Investigate techniques like observation normalization, reward scaling, or different network architectures to further improve performance and stability.

    Code Refinements: Add more comprehensive error handling, type hinting, and unit tests.
