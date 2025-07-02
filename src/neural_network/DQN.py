import torch
import torch.nn as nn
import random

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(81, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.model(x)
    
def select_action(model, state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 3)
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = model(state_tensor)
        return q_values.argmax(dim=1).item()
    
def train(model, target_model, memory, optimizer, gamma, batch_size):
    if len(memory) < batch_size:
        return
    
    batch = memory.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    next_states = torch.FloatTensor(next_states)
    dones = torch.BoolTensor(dones).unsqueeze(1)

    q_values = model(states).gather(1, actions)

    with torch.no_grad():
        max_next_q = target_model(next_states).max(1, keepdim=True)[0]
        target = rewards + gamma * max_next_q * (~dones)
    
    loss = nn.MSELoss()(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()