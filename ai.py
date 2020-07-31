# Importação das bibliotecas

import numpy as np  # trabahar com arrays
import random # gerar bacths
import os # partes do sistema operacional
import torch
import torch.nn as nn
import torch.nn.functional as F # funcões de ativação
import torch.optim as optim # otimizadores
import torch.autograd as autograd
from torch.autograd import Variable

# Criação da arquitetura da rede neural
# Só temos apenas uma camada oculta
class Network(nn.Module):
    def __init__(self, input_size, nb_action):  # Criaremos uma funcção para iniciarmos com o __init__ o self(trabalhando com objeto e não com a classe) neuronios de entrada = 5 p/ camada de entrada e o nb_action serão 3 ações que podem ser 0 para frente, 20 para esquerda e -20 para direita
        super(Network, self).__init__() # chamando o construtor da classe 
        self.input_size = input_size # atributos para esse obejto no arquivo do mapa
        self.nb_action = nb_action
        
         # precisamos fazer conezão entra as camadas
         # 5 camada de entrada -> 30 oculta -> 3 saída conectados (Experimentos) -> full connection
         # Abixo será a ligação da camada de entrada e camada oculta
        
        # 5 -> 30 -> 3 - full connection (dense)
        self.fc1 = nn.Linear(input_size, 30) # full connection(dense) rede neural densa, 1 neuronio que está conectado a vários outros neuronios
        self.fc2 = nn.Linear(30, nb_action) # conectando a camada oculta com a camada de saída
        
        # ACIME JÁ ESTAMOS CONSIDERANDO A NOSSA REDE DE MODO BÁSICO
        
    # Precisamos está efetuando ativação da rede
    def forward(self, state): # recebe estado
        x = F.relu(self.fc1(state)) # o x tem 30 neuronios om a funcção relu e depois será multiplicado 
        q_values = self.fc2(x)
        return q_values
    
# REPLAY DE EXPERIENCIA
# Implementação do replay de experiência
class ReplayMemory(object):
    def __init__(self, capacity): # capacidade de memoria
        self.capacity = capacity
        self.memory = [] # 100 ultoms eventos que serão adcionados
        
    # D,E,F,G,H
    # 4 valores: último estado, novo estado, última ação, última recompensa
    def push(self, event):
        self.memory.append(event) # adciomos o evento dentro da memoria
        if len(self.memory) > self.capacity: # veriica o tamanho
            del self.memory[0] # se for maior será necessário apagar
            
    def sample(self, batch_size): # retorna uma amostra
        # ((1,2,3), (4,5,6)) -> ((1,4), (2,5), (3,6))
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
        
# Implementação de Deep Q-Learning
class Dqn():
    def __init__(self, input_size, nb_action, gamma): # Criar a função que iremos passar 3 valores ao mapa
        self.gamma = gamma # criando um atributo para o valor 0.9
        self.reward_window = [] # janela de recompensa 0 que sera depois tirado a média
        self.model = Network(input_size, nb_action) # criação da rede neural (camada de saída)
        self.memory = ReplayMemory(100000) # memoria das ultimas ações com 100 mil eventos
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001) # optimizadores com taxa de aprendizagem de 0.001 (como primeira opção)
        self.last_state = torch.Tensor(input_size).unsqueeze(0) # armazena qual foi o ultimo estado (vetor com 5 elementos ou seja 3 sensores com 2 eventos + ou -)
        self.last_action = 0
        self.last_reward = 0
        
      # SELEÇÃO DE UMA DETERMINADA AÇÃO EM UM DETERMINADO TEMPO (ESQUERDA, DIREITA OU PARA FRENTE) Evita bater nos obstáculos
        
    def select_action(self, state): 
        # sofmax(1,2,3) -> (0.04, 0.11, 0.85) -> (0, 0.02, 0.98) somando os três valores será 100%
        probs = F.softmax(self.model(Variable(state, volatile = True)) *100) # T = 7  (temperatura)
        
        
        action = probs.multinomial() # seleciona aleatoriamente (sorteio) para retornar ação
        return action.data[0,0] # posição 0 e 0
    
    # FUNÇÃO DE APRENDIZAGEM
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad() # reinicia a variável a cada vez que entra 
        td_loss.backward(retain_variables = True) # melhora o desempenho e aproveita a variável
        self.optimizer.step() # atualiza os pesos
        
    
    # FUNÇÃO DE ATUALIZAÇÃO MEDIANTE AO ESTADO    
    def update(self, reward, new_signal): 
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]),
                     torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100: # lista com as memorias se for maior que 100 é hora do algoritimo efetuar aprendizagem
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000: # tamanho da janela
            del self.reward_window[0]
        return action
    
    # computar o score das recompensas
    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.)
    
    # salvar o modelo
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, 'last_brain.pth') # salva no last_brain.pth
    # carregar o modelo
    def load(self):
        if os.path.isfile('last_brain.pth'):
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('Carregado com sucesso') # salvo
        else:
            print('Erro ao carregar') # apenas se houver error
    
    
    # TESTE DE NÍVEIS DE APRENDIZAGEM AUTONOMA
        # Nível 01
        
        # Sair do ponto A p/ B na (diagonal)
        # será mais confiante quando aumentarmos o valor da temperatura para 100. Note que efetuou a exploração do ambiente 
        # está mais com um formato de carro, usa parametros como o softmax, tendo em vista se aumterar muito a Temperatura
        # irá atrapalhar o caminnho pela diagonal
        # Vamos usar o botão salvar o modelo
        # Na viariavel SCORE, verifica a médias das 1000 ultmas recompensas,
        # começa com negativo(-) e depois irá para recomensa positiva (+)
        
        # Nível 02
        
        # Load carrega o Nível 01
        # Vamos definir uma estrada para nosso carro autonomo
        # encostando-se na areaia, note que irá diminuir a velocidade
        
        #nivel 03
        
        # Alguns testes como traçados no map de areia |
        
        # Nível04 
        
        # Traçar barreiras e melhorar o algoritimo
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        