import os
import json
import random
from src.environment import HouseEnvironment
from src.agent import DQNAgent

class SimulationManager:
    def __init__(self, num_houses=25, save_dir="saved_models"):
        self.num_houses = num_houses
        self.save_dir = save_dir
        
        self.environments = [HouseEnvironment(house_id=i) for i in range(num_houses)]
        self.agents = [DQNAgent(state_dim=3, action_dim=2, agent_id=i) for i in range(num_houses)]
        
        # Varsayılan kişilik ataması (Kayıttan okunmazsa devreye girer)
        self.personalities = []
        for i in range(num_houses):
            if i < 5:
                self.personalities.append("CESUR")
            elif i < 10:
                self.personalities.append("KORKAK")
            else:
                self.personalities.append("DENGELI")
        
        self.states = [None] * num_houses
        self.done_flags = [False] * num_houses
        
        self.episode = 1
        self.history_avg_scores = []
        
        self.recent_best_score = 0
        self.recent_best_house = -1
        
        self.reset_simulation_run()
        self.load_all_agents()
        self.load_metadata()  # Kaldığı yeri ve değiştirilen kişilikleri yükler

    def reset_simulation_run(self):
        self.done_flags = [False] * self.num_houses
        for i in range(self.num_houses):
            self.states[i] = self.environments[i].reset()

    def step_all(self):
        all_done = True
        for i in range(self.num_houses):
            if self.done_flags[i]:
                continue
            
            all_done = False
            state = self.states[i]
            agent = self.agents[i]
            env = self.environments[i]
            pers = self.personalities[i]
            
            # Yapay zeka eylem seçer
            action = agent.act(state)
            
            # KİŞİLİK ÖZELLİKLERİ FİLTRESİ
            if pers == "CESUR" and env.day < 15:
                action = 0  # Ekmek kırıntısı zorlaması
            elif pers == "KORKAK":
                upper_risk = state[2]
                if upper_risk > 0.05:
                    action = 1  # Korkudan hemen peynire yönelme
            
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            
            self.states[i] = next_state
            self.done_flags[i] = done
            
            if done:
                agent.decay_epsilon()
                
        return all_done

    def share_knowledge(self):
        scores = [env.score for env in self.environments]
        best_score = max(scores)
        leader_idx = scores.index(best_score)
        leader_agent = self.agents[leader_idx]
        
        tau = 0.10 
        leader_state_dict = leader_agent.model.state_dict()
        
        for i in range(self.num_houses):
            if i == leader_idx:
                continue
            
            agent_state_dict = self.agents[i].model.state_dict()
            for key in leader_state_dict:
                agent_state_dict[key] = (1.0 - tau) * agent_state_dict[key] + tau * leader_state_dict[key]
            
            self.agents[i].model.load_state_dict(agent_state_dict)
            
        leader_memory = list(leader_agent.memory)
        if len(leader_memory) > 0:
            share_size = min(len(leader_memory), 50)
            shared_experiences = random.sample(leader_memory, share_size)
            for i in range(self.num_houses):
                if i == leader_idx:
                    continue
                for exp in shared_experiences:
                    self.agents[i].remember(*exp)
                    
        return leader_idx, best_score

    def get_all_statuses(self):
        statuses = []
        for i in range(self.num_houses):
            env = self.environments[i]
            agent = self.agents[i]
            
            if not env.is_dead:
                q_vals = agent.get_q_values(self.states[i])
                q_breadcrumbs, q_cheese = q_vals[0], q_vals[1]
            else:
                q_breadcrumbs, q_cheese = 0.0, 0.0

            statuses.append({
                "id": i,
                "day": env.day,
                "score": env.score,
                "is_dead": env.is_dead,
                "death_reason": env.death_reason,
                "p_caught": env.p_caught,
                "epsilon": agent.epsilon,
                "personality": self.personalities[i],
                "q_breadcrumbs": q_breadcrumbs,
                "q_cheese": q_cheese
            })
        return statuses

    def save_all_agents(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        for i in range(self.num_houses):
            path = os.path.join(self.save_dir, f"agent_{i}.pth")
            self.agents[i].save(path)
        self.save_metadata()

    def load_all_agents(self):
        if not os.path.exists(self.save_dir):
            return
        for i in range(self.num_houses):
            path = os.path.join(self.save_dir, f"agent_{i}.pth")
            if os.path.exists(path):
                self.agents[i].load(path)

    def save_metadata(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        meta_data = {
            "episode": self.episode,
            "history_avg_scores": self.history_avg_scores,
            "personalities": self.personalities  # Kişilikleri de json dosyasına ekledik
        }
        with open(os.path.join(self.save_dir, "simulation_meta.json"), "w") as f:
            json.dump(meta_data, f)

    def load_metadata(self):
        meta_path = os.path.join(self.save_dir, "simulation_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta_data = json.load(f)
                    self.episode = meta_data.get("episode", 1)
                    self.history_avg_scores = meta_data.get("history_avg_scores", [])
                    
                    saved_personalities = meta_data.get("personalities")
                    if saved_personalities and len(saved_personalities) == self.num_houses:
                        self.personalities = saved_personalities
            except Exception:
                pass

    def reset_brains(self):
        self.agents = [DQNAgent(state_dim=3, action_dim=2, agent_id=i) for i in range(self.num_houses)]
        self.history_avg_scores = []
        self.episode = 1
        
        # Sıfırlandığında varsayılan kişilik düzenine döner
        self.personalities = []
        for i in range(self.num_houses):
            if i < 5:
                self.personalities.append("CESUR")
            elif i < 10:
                self.personalities.append("KORKAK")
            else:
                self.personalities.append("DENGELI")
                
        self.reset_simulation_run()
        
        if os.path.exists(self.save_dir):
            for file_name in os.listdir(self.save_dir):
                file_path = os.path.join(self.save_dir, file_name)
                try:
                    os.remove(file_path)
                except OSError:
                    pass