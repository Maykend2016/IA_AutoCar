# IA_AutoCar
 
**Construindo nosso primeiro carro autônomo com Inteligência Artificial e Deep Q-Learning**

Nesse projeto, tenho o intuito de simular um carro autônomo usando aprendizagem por reforço, como também demonstrar algumas técnicas de Deep Learning com a biblioteca PyTorch e a linguagem Python!

**Requisitos:**

- Anaconda 
- Spyder na versão 3.3.4
- Usar a verão Python 3.6
- Baixar esse projeto em sua máquina 
- Descompactar o projeto

**Configuração de Ambiente**

No Anaconda teremos que acessar o **environments** e definir o **python** na **versão 3.6**, certifique-se o estado do ambiente no python 3.6, defina também um **nome para seu projeto**, por fim instalaremos a **IDE Spyder** na **versão 3.3.4**.


Iremos usar o **Anaconda Prompt** para efetuarmos algumas instalações no pacote e digite:

<br>

**Comandos no Anaconda Prompt**

No Windows:

**Ativação do ambiente:**

(base) C:\Users\mayke> activate carro_autonomo

**Instalação do pytorch na versão 0.3.1:**

(base) C:\Users\mayke> conda install -c peterjc123 pytorch-cpu

**biblioteca para desenhar o mapa do carro:**

(base) C:\Users\mayke> conda install -c conda-forge kivy

<br>

Em nossa aplicação haverá **3 sensores** com o objetivo de sair do ponto **A** ao ponto **B** e assim vice-versa. Nossa rede neural será responsável por treinar o veículo mediante a sua aprendizagem por reforço circular pelo ambiente, faremos algumas barreiras de "areia" com o intuito de simular o desvio. Assim podendo circular pela diagonal do player.

Nosso carro autonomo é bastante interessante pelo fato que podermos dar entradas nos pesos para que gerem uma saída relavante a nossa análise, tendo em vista que quando ele esbarra em um traçado de areia irá ganhar pontuação (-) negativa e pontuação (+) ao sair do ponto **A** ao ponto **B** sem que se esbarre em caixa de areia. Usaremos 3 funcionalidades, 1 será a opção de limpar nossos dados, 2 terá o objetivo de salvar nosso aprendizagem levando o veículo à aprender mais rápido fazendo com que percorra a diagonal e por fim, a opção 3 "Load" para efetuar o carregamento de nossa análise.

<br>

# TESTE DE NÍVEIS DE APRENDIZAGEM AUTONOMA

**Nível 01**
        
        - Sair do ponto A p/ B na (diagonal).
        - Será mais confiante quando aumentarmos o valor da temperatura "T" para 100. Note que efetuou a exploração do ambiente. 
        - Note que está mais com um formato de carro, usa-se parâmetros como o Softmax, tendo em vista se aumterar muito a Temperatura irá atrapalhar o caminnho pela diagonal.
        - Vamos usar o botão salvar o modelo.
        - Na viáriavel SCORE, verifica a médias das 1000 ultimas recompensas,
        - começa com negativo(-) e depois irá para recomensa positiva (+)
        
**Nível 02**
        
        - Load carrega o Nível 01.
        - Vamos definir uma estrada para nosso carro autonomo.
        - Encostando-se na areaia, note que irá diminuir a velocidade.
        
**Nível 03**
        
        - Alguns testes como traçados no MAP de areia na vertical.
        
**Nível 04**
        
        - Traçar barreiras e melhorar o algoritimo com mais camadas de T.
        
Segue o link no driver compartilhado de modo demonstrativo do projeto executando: <>
        

 
