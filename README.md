# Mouse DQN Simulation

Multi-agent Deep Q-Network simulation where 25 mice in parallel houses choose each day between **breadcrumbs** (survive, +1 score, rising capture risk) or **cheese** (+100 score, certain death).

## Features

- **Partial observability** — true capture probability is hidden behind a random confidence interval
- **Personality modes** — Bold (CESUR), Timid (KORKAK), Balanced (DENGELI); change per house via right-click
- **Peer learning** — leader weight mixing and shared experience replay every 10 episodes
- **Live Qt UI** — 5×5 house grid, score chart, and neural network weight visualization

## Requirements

- Python 3.10+
- See [requirements.txt](requirements.txt)

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Trained weights are saved under `saved_models/` at runtime. Pre-trained `.pth` files are not included in the repository; the simulation starts fresh or loads weights you train locally.

## Project structure

```
mouse_dqn_simulation/
├── main.py              # Application entry point
├── requirements.txt
├── src/
│   ├── environment.py   # House / risk / reward logic
│   ├── agent.py         # DQN agent and Q-network
│   ├── simulation.py    # Multi-house manager
│   └── ui.py            # PySide6 control panel
└── saved_models/        # Runtime checkpoints (gitignored)
```

## License

MIT © [mrtfff](https://github.com/mrtfff)
