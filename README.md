# Decision tree

Esse projeto foi concebido com a ideia de aperfeiçoar meus conhecimentos no funcionamento de algoritmos de ML, mais especificamente em árvores de decisão, que é um algoritmo base para muitos outros como Random Forest e árvores com Gradient Boosting, Nele foi construído do zero o código para gerar uma arvore de decisão e usa-la para fazer predições com dados tabulares reais e posteriormente foi comparado o resultado com a biblioteca scikit-learn que é muito usada no mercado para atestar a eficiência do algoritmo construído. O resultado final foi que na maioria dos casos o algoritmo obteve desempenho muito parecido ou igual ao do scikit-learn

## requisitos para rodar o código:
    python
    Numpy
    Pandas
    Scikit-learn

    caso não tenha as bibliotecas instaladas:
    pip install numpy pandas scikit-learn

## Como rodar o código:
    
1. Entre na pasta do projeto com sua IDE
1. Crie um arquivo .py ou use o main.py
1. Import `decision_tree` e as bibliotecas que você precisa como o `pandas` e `sklearn`
1. Baixe o DataSet que você vai usar
1. Separe o DataSet em treino e teste, pode usar o train_test_split do sklearn
1. Instancie a classe `Decision_tree`
1. Use o método `create_tree` no objeto criado passando como parâmetro o conjuntos de dados (tipo **pd.DataFrame**) e target (tipo **pd.Series**) de teste para treinar o modelo, você pode conferir os demais parâmetros olhando o código desse método na classe assim como é explicado posteriormente nesse readme
1. Use o método `predict` no objeto já treinado passando como parâmetro uma **pd.Series** com o index sendo os nomes das colunas desse dataset referente a uma linha do dataset, e você vai ter a predição para esse linha de dados. Caso você queria aplicar em um df inteiro você pode fazer: `pd.Series([ tree.predict(entrada_teste.iloc[i]) for i in range(len(entrada_teste)) ])`
1. Rode o código

## Como árvores de decisão funcionam:
Para começar a entender como árvores de decisão funcionam, vamos imaginar o seguinte exemplo: queremos saber qual a chance de uma pessoa sair de casa em determinado dia baseado na quantidade de chuva (eixo x) e na quantidade de luz solar (eixo y).

vamos considerar o seguinte dataset:

![dataset](graficos/dataset.png)

que gera o seguinte gráfico onde os pontos verdes são pessoas que sairam de casa e os vermelhos são pessoas que não sairam de casa:

![Gráfico](graficos/Grafico_chuvas_01.png)

A pergunta é: Como podemos separar esses grupos?
uma forma muito simples de fazer isso seria separar os grupos usando retas da seguinte forma:

Olhando primeiramente para o eixo x podemos traçar uma linha que separa os valores <= 1.05 dos valores > 1.05 e teriamos o seguinte gráfico:

![Gráfico_linha1](graficos/Grafico_chuvas_linha1.png)

Depois podemos ir para o eixo y e separar os valores que são >= 0.9 dos que são < 0.9 para ter o seguinte gráfico:

![Gráfico_linha2](graficos/Grafico_chuvas_linha2.png)

Dessa maneira podemos seguir a seguinte lógica para definir se alguém vai o não sair de casa:

![arvore_decisão](graficos/arvore_decisao.png)

Acabamos de montar uma arvore de decisão para resolver esse problema!

Agora que já entendemos a teoria por trás do funcionamento de uma arvore de decisão vamos entender como funciona na prática o algoritmo para montar uma delas passo a passo cobrindo os seguintes pontos:

- Como escolher retas usando Gini
- Quando a arvore deve parar de crescer

## Como escolher as melhores retas (Threshold)
Como vimos anteriormente para montar nossa arvore precisamos traçar várias retas para separar diferentes grupos, mas como podemos escolher a melhor reta para dividir um gráfico dentre infinitas retas? Uma das formas de resolver esse problema é a seguinte:

O primeiro passo é conhecer o conjunto de retas realmente viáveis, pois existe infinitas retas que podem ser construídas entre um ponto A e B consecutívos (um do lado do outro do gráfico) do eixo X ou Y, mas todas as retas vão separar examente o mesmo conjunto de dados nesse respectivo eixo já que os pontos são consecutivos, um exemplo disso no gráfico a baixo com relação ao eixo X:

![infitas_retas_gráfico](graficos/escolha_threshold.png)

podemos ver que cada uma das diferentes retas que separam dois pontos consecutivos em determinado eixo delimitam exatamente o mesmo grupo de dados, ou seja, se escolhermos qualquer uma delas os pontos a esquerda e a direita vão ser exatamente os mesmos. podemos escolher um representante dessas retas que fica bem no meio dos pontos consecutivos para deixar a reta o mais longe possível de "encostar" nos pontos, isso ajuda que novos dados com valores muito semelhantes aos antigos não fiquem de um lado errado da separação, usamos a seguinte formula para pegar a reta entre dois pontos consecutivos:

    p1 = valor do ponto no respectivo eixo mais a esquerda (menor)
    p2 = valor do ponto no respectivo eixo mais a direita (maior)

    (p2 + p1) / 2

Dessa forma podemos pegar todas as retas possíveis entre cada um dos pontos para analisar qual a melhor, da seguinte maneira:

    1- pegar todos os valores únicos do respectivo eixo
    2- listar em ordem crescente
    3- calcular todos os Thresholds entre dois pontos consecutivos do eixo em questão

seguindo esses passos nós teriamos os seguintes Thresholds para o eixo x:

![todos_thresholds](graficos/todos_thresholds_x.png)

Agora que sabemos quais são todos os Thresholds possíveis para esse eixo precisamos saber realmente qual desses é o melhor para dividir os dados, para isso nós vamos usar a formula `Gini` ela serve para calcular quão mal dividido está um está um conjunto de dados. quanto maior o valor do gini mais mal dividido estão os dados, recomento pesquisar como funciona a formula de Gini para ter maior compressão dessa parte.

    Gini = 1 - somatório( p1**2 )

A função `gini_calculate` em `decision_tree.py` é responsárel por esse calculo no código

Para escolher o melhor Threshold vamos calcular qual o Gini dos targets que ficaram no lado esquerdo (valor <= threshold) e do lado direito (valor > threshold) e calcular a média ponderada entre esses valores usando a quantidade de targets em cada grupo como peso (precisa ser uma média ponderada pois cada parte vai ter uma quantidade diferente de valores)

A função `threshold_gini` em `decision_tree.py` é responsável por essa parte no código

um exemplo para entender melhor, vamos considerar o seguinte threshold dessa lista:

![grafico](graficos/Grafico_chuvas_linha1.png)

Temos os seguintes dados nesse Threshold:
- Quantidade de targets a esquerda = 10 (6 pessoas sairam de casa 4 pessoas não sairam de casa)
- Quantidade de dados a direita = 8 (0 pessoas sairam de casa e 8 pessoas não sairam de casa)

com essas informações vamos calcular o gini ponderado para esse Threhold

    gini esquerda = 1 - ( (6/10)**2 + (4/10)**2 ) = 0.48
    gini direita = 1  - ( (0/8)**2 + (8/8)**2 )   = 0

    gini ponderado = (0.48 * 10) + (0 * 8) / 18 = 0.2666

vamos fazer o mesmo processo para cada um dos Thresholds, no final vamos ter vários valores para o `gini ponderado` vamos escolher o menor valor, isso signiufica que esse Threshold foi o que mais bem separou os dados. a situação ideal seria um Threshold que separa os dados de forma que todas as amostras de uma classe fiquem de uma lado e todas as amostras de outra classe fiquem de outro, fazendo isso achamos a linha que mais chega perto de fazer isso disso.

## Como isso funciona na ao montar a árvore de decisão

Agora que já sabemos como selecionar os Thresholds podemos começar a entender como a arvore de decisão é montada. Normalmente temos um Dataset de dados tabulares para nos basear para construção da nossa arvore, o primeiro passo é selecionar em qual coluna (eixo do gráfico, mas lembrando que podem existir N dimensões) vai ser usada dentre tantas que existem, para isso precisamos escolher o melhor Threshold de **cada uma dessas colunas** e avaliar qual o seu gini, dessa forma vamos saber qual é a melhor coluna e seu respectivo threshold para separar os dados baseado no menor gini entre elas, e vamos seleciona-la para o nó da arvore, após isso vamos passar a tabela para os nós em diante, considerando o threshold para criar um no a esquerda (linhas da tabela onde o valor na coluna selecionada <= threshold) e um nó a direita (linhas da tabela o valor na coluna selecionada > threshold) e vamos fazer o mesmo processo para os nós seguintes

No código a função que é responsável por essa etapa é `create_tree` em `decision_tree.py`

## Quando a arvore para de crescer:

Se não definirmos nenhum critério de parada a arvore vai crescer até que todas as linhas tenham sido usadas e não sobre mais nenhuma amostra, mas isso faz com que a arvore Overfitting, isso basicamente é quando a arvore se molda muito aos dados usados durante o treinamento de forma que se forem usados dados novos a arvore vai errar por estar "presa" demais aos dados antigos, vale a pena estudar mais sobre o tópico caso você se interesse em ML. para evitar que isso aconteça precisamos definir os critérios de parada, os principais e que foram usados nesse código foram:

- Quando conjunto de targets é puro, ou seja todas as amostras do nó são da mesma classe
- arvore muito profunda (parametro **max_deep** da classe `Decision_tree`)
- Quantidade de amostras for pequena demais (parametro **min_samples_split** da classe `Decision_tree`)
- Quantidade de amostras de alguma folha gerada for pequena demais (parametro **min_samples_leaf** da classe `Decision_tree`)
- Decrescimento da impureza dos nós é muito pequena (parametro **min_impurity_decrease** da classe `Decision_tree`)
- Caso todas as colunas de uma tabela sejam compostas por somente um valor único (um mesmo valor para todas as linhas de todas as colunas, podem ser valores diferente entre uma coluna e outra) esse é importante pois não tem mais como aprender um padrão com esse tipo de dado então é melhor transformar em uma folha

Quando qualquer um desses critérios de parada é satisfeito o nó da arvore vira uma folha, ou seja, ele passa a conter o valor de retorno da predição ("o sair de casa" ou "não sair de casa" do nosso problema). Caso não seja um nó puro (somente amostras de uma classe) o valor da folha vai ser a classe que mais se repete dentro das amostras daquele nó.
