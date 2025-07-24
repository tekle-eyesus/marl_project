#  MARL

A modular **Multi-Agent Reinforcement Learning (MARL)** system built using **PettingZoo** and **PyTorch**, designed for **cooperative multi-agent environments**. The current implementation provides a working baseline of **Independent Q-Learning (IQL)** and serves as a foundation for more advanced MARL approaches.

---

##  Features

- ✅ Environment wrapper for PettingZoo (`simple_spread_v3`)
- ✅ Modular agent architecture (`BaseAgent`, `IQLAgent`)
- ✅ Centralized training script with logging and plotting
- ✅ Epsilon-greedy exploration
- ✅ Target network updates
- ✅ Replay buffer for experience reuse
- 📈 Reward visualization (`matplotlib`)
- 🧩 Extensible design for future algorithms (QMIX, MADDPG, etc.)

---

## 🎯 Algorithm: Independent Q-Learning (IQL)

Each agent:
- Trains independently using DQN-like architecture.
- Uses its **own observations** and **Q-network** to select actions.
- Maintains its own replay buffer and updates via Q-learning.

> **Goal**: Learn policies that let agents cooperatively reach landmarks in `simple_spread_v3`.

---

##  Dependencies

Install via pip:

```bash
pip install -r requirements.txt
```

Run:
```bash
python train.py
```
