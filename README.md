### README.md

# Trabalho Prático - Inteligência Computacional I

## Introdução

Este repositório contém a implementação dos exercícios computacionais descritos no Homework #1 e no Homework #2 do professor Yaser Abu-Mostafa, conforme solicitado no Trabalho Prático da disciplina de Inteligência Computacional I, oferecida pelo Prof. Carlos Eduardo Pedreira no PESC/COPPE/UFRJ.

## Estrutura do Repositório

- `main.py`: Código principal com todas as funções e experimentos implementados.
- `README.md`: Este arquivo de documentação.
- `report.pdf`: Relatório prático contendo respostas e interpretações dos resultados.

## Descrição dos Exercícios

### Perceptron

Neste exercício, implementamos o Algoritmo de Aprendizagem Perceptron (PLA) para classificar pontos em um espaço bidimensional. O objetivo é entender o funcionamento do PLA e observar seu desempenho.

#### Questões

1. **Número médio de iterações até a convergência para N = 10 pontos de treinamento.**
2. **Probabilidade de divergência \( P[f(x) != g(x)] \) para N = 10.**
3. **Número médio de iterações até a convergência para N = 100 pontos de treinamento.**
4. **Probabilidade de divergência \( P[f(x) != g(x)] \) para N = 100.**
5. **Estabelecimento de uma regra para a relação entre N, o número de iterações até a convergência e \( P[f(x) != g(x)] \).**

### Regressão Linear

Neste exercício, exploramos como a Regressão Linear pode ser usada para tarefas de classificação, utilizando o mesmo esquema de produção de pontos do Perceptron.

#### Questões

1. **Erro médio dentro da amostra \( E_in \) para N = 100.**
2. **Erro médio fora da amostra \( E_out \) para os \( g \) encontrados na questão anterior.**
3. **Número médio de iterações do PLA usando os pesos da Regressão Linear como inicialização para N = 10.**
4. **Desempenho da versão pocket do PLA em um conjunto de dados não-linearmente separável.**

### Regressão Não-Linear

Aqui, aplicamos a Regressão Linear para classificação com transformação não-linear dos dados.

#### Questões

1. **Erro médio dentro da amostra \( E_in \) usando o vetor de atributos \((1, x_1, x_2)\).**
2. **Transformação dos dados seguindo o vetor de atributos não-linear \((1, x_1, x_2, x_1 x_2, x_1^2, x_2^2)\) e comparação das hipóteses geradas.**
3. **Erro médio fora da amostra \( E_out \) para a hipótese encontrada na questão anterior.**

## Como Executar

1. Clone o repositório:
   ```sh
   git clone https://github.com/SylvioMello/Computational-Inteligence-Class.git
   cd Computational-Inteligence-Class
   ```

2. Certifique-se de ter o Python instalado e as bibliotecas necessárias:
   ```sh
   pip install numpy matplotlib
   ```

3. Execute o código principal:
   ```sh
   python main.py
   ```

## Interpretação dos Resultados

Cada seção do relatório prático deve conter uma interpretação detalhada dos resultados obtidos.

### Exemplo de Interpretação

Para a questão sobre o número médio de iterações até a convergência do PLA com N = 10 pontos de treinamento, encontramos que, em média, o algoritmo leva 9 iterações. Este resultado demonstra a eficiência do PLA em conjuntos de dados pequenos e sua capacidade de encontrar uma fronteira de decisão que separa perfeitamente os pontos de treinamento.

## Conclusão

Este trabalho prático proporcionou uma compreensão mais profunda dos algoritmos de aprendizado de máquina, especialmente o Perceptron e a Regressão Linear, além de destacar a importância da transformação de características em problemas de classificação não-lineares.

## Referências

- Yaser Abu-Mostafa, Homework #1 e Homework #2. Disponível em:
  - https://work.caltech.edu/homework/hw1.pdf
  - https://work.caltech.edu/homework/hw2.pdf

Para mais detalhes sobre os experimentos e interpretações, consulte o relatório `report.pdf`.

---