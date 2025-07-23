from env.env_wrapper import MultiAgentEnv
from agents.iql_agent import IQLAgent
import numpy as np
import matplotlib.pyplot as plt

NUM_AGENTS = 3
EPISODES = 100
MAX_STEPS = 25
TARGET_UPDATE_FREQ = 20
LOG_INTERVAL = 10  # Print progress every 10 episodes

def main():
    env = MultiAgentEnv(num_agents=NUM_AGENTS)
    agents = {
        agent_name: IQLAgent(env.observation_spaces[agent_name], env.action_spaces[agent_name])
        for agent_name in env.agents
    }
    reward_history = {agent: [] for agent in env.agents}

    # Training Loop
    for episode in range(1, EPISODES + 1):
        state = env.reset()
        total_rewards = {agent: 0 for agent in env.agents}

        for step in range(MAX_STEPS):
            actions = {}
            for agent_name in env.agents:
                actions[agent_name] = agents[agent_name].select_action(state[agent_name])

            # Debugging--Validate actions
            for agent_name, act in actions.items():
                if not env.env.action_space(agent_name).contains(act):
                    print(f"[Warning] Invalid action for {agent_name}: {act}")

            next_state, rewards, dones, _ = env.step(actions)

            for agent_name in env.agents:
                agents[agent_name].store_transition(
                    state[agent_name],
                    actions[agent_name],
                    rewards[agent_name],
                    next_state[agent_name],
                    dones[agent_name]
                )
                agents[agent_name].train_step()

                total_rewards[agent_name] += rewards[agent_name]

            state = next_state
            if all(dones.values()):
                break

        # Decay epsilon
        for agent in agents.values():
            agent.decay_epsilon()

        # Update target networks
        if episode % TARGET_UPDATE_FREQ == 0:
            for agent in agents.values():
                agent.update_target()

        # Logging per agent
        for agent_name in env.agents:
            reward_history[agent_name].append(total_rewards[agent_name])

        # Summary per interval
        if episode % LOG_INTERVAL == 0 or episode == 1:
            avg_total_reward = np.mean(
                [sum(reward_history[agent][-LOG_INTERVAL:]) for agent in env.agents]
            )
            print(f"Episode {episode}/{EPISODES} | "
                  f"Average Total Reward (last {LOG_INTERVAL}): {avg_total_reward:.2f}")
            for agent_name in env.agents:
                print(f"  {agent_name} reward: {total_rewards[agent_name]:.2f}")

    # Visualization
    for agent_name, rewards in reward_history.items():
        plt.plot(rewards, label=f"Agent {agent_name}")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Learning Curves")
    plt.legend()
    plt.savefig("training_rewards.png")
    print("Plot saved as training_rewards.png")
    plt.show()

    env.close()

if __name__ == "__main__":
    main()
