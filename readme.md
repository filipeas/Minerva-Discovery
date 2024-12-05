# Minerva-Discovery
- Experiments using Minerva-Dev!

## How clone this repositore
1) First, head this README.md!
2) In your machine, execute:
    - ``` git clone https://github.com/filipeas/Minerva-Discovery.git ```
    - ``` cd Minerva-Discovery ```
    - ``` git submodule update --init --recursive ```

## How to execute in local (for test)
- Run ``` ./setup_and_run.sh ``` if you run outside of container.
- Run ``` ./setup_and_run.sh --container ``` if you run outside of container.


## How execute into Ogbon (Petrobras)
1) Read this README.md file first!
2) Make sure you have Singularity on server.
3) Open **Singularity.def** and see your configuration. Add new packeges into environment, if you need.
4) Add new packages in **requirements.txt**, if you need.
5) Build a image: 
    - ``` singularity build Singularity.sif Singularity.def ```. 
    Make sure rename labels **sam_FAS** to a name that you want (check into **Singularity.def**, because there is a flag called **Maintainer** with the same name).
6) (optional) You can run singularity image with: 
    - ``` singularity exec --nv Singularity.sif bash ```. 
    - But, in Ogbon (Petrobras), you need run with Slurm (step 7), SO, BE CAREFUL!
7) (optional) In Ogbon (Petrobras), execute:
    - ``` srun --partition gpulongd --account asml-gpu --job-name=sam_FAS --time 120:00:00 --pty bash ```. 
    - This will open a bash for use Slurm. So, after this, run **step 6)**.
8) (opcional) If you need run using sbatch, use this: 
    - ``` sbatch --output=meu_job.log --error=meu_job.err run_sbatch.sh ```.
9) **PS: REMEMBER AWAYS BUID SINGULARITY.SIF**

## Tips
- In Ogbon (Petrobras), execute ``` srun --partition gpulongd --account asml-gpu --job-name=NOME_DO_PROJETO_OU_EXPERIMENTO --time 120:00:00 --pty bash ``` for interative bash.
- More details, see:
    - https://hpc.senaicimatec.com.br/docs/clusters/job-scheduling/

## How to run with TMUX
- Access some node (discovery) and execute: ``` tmux new -s sam_FAS ```
- Go to directory (your-directory) and execute: ``` python main.py ```
- After this, exit of session executing this: ``` Ctrl + B ``` and press ``` D ```.
- To access the session, execute: ``` tmux attach -t sam_FAS ```
- To list TMUX sessions, execute: ``` tmux ls ```

## How to send datasets to some server
- If you need use some dataset, for example F3 or Seam ai and if this datasets stay in your local or in another server, you can transfer dataset between machines using scp command. For example:

- To transfer seam ai from Ogbon to Coaraci (server of unicamp's phisics course), execute: ``` scp -r pasta-do-dataset-na-maquina-de-origem username@host.name.com.:/home/users/local-de-destino ```

## Explicando o fluxo do processo do SAM
1. Pré-processamento
- Considerando que a imagem de entrada (seção sísmica) tem (255,701,3), que é o caso do Dataset F3, a imagem vai passar pela função preprocess(), que está dentro da classe Sam(), e realiza esse processamento em tempo de execução:
    - Aplica normalização: $$x = \frac{x-pixel_mean}{pixel_std}$$
        - Isso centraliza os valores em torno de 0 com desvio padrão unitário.
    - Aplica preenchimento (pad): A imagem é preenchida para se tornar quadrada, com uma dimensão padrão de 1024x1024, porque o ImageEncoderViT opera com entradas quadradas.
        - Para uma entrada de tamanho 255x701:
            - Altura (255): 1024 - 255 = 769 pixels de padding aplicados a direita da imagem, preenchendo-a.
            - Largura (701): 1024 - 701 = 323 pixels de padding aplicados a baixo da imagem, preenchendo-a.
        - O padding é feito com zeros, posicionados á direita e abaixo da imagem original.

2. ImageEncoderViT
    1. Patch Embedding
        - Dentro da classe Sam(), após o pré-processamento, imagem passa pelo módulo (classe) PatchEmbed(), que irá:
            - Dividir em patches de tamanho 16x16 (esse é o padrão, mas pode ser mudado em patch_size via parametro).
            - Cada patch é projetado em um vetor de dimensão 768 (esse é o padrão, mas pode ser mudado em embed_dim via parametro), usando uma convolução 2D:
                ```bash
                    self.proj = nn.Conv2d(
                        in_chans, embed_dim, kernel_size=kernel_size,stride=stride, padding=padding)
            - Aqui:
                - kernel: 16x16
                - stride: 16x16 (salta blocos de 16 pixels)
                - output: a imagem, que foi pré-processada para 1024x1024 é convertida em um mapa de features com tamanho 64x64x768.
            - Após isso, o tensor é permutado para o formato (BxHxWxC)
    
    2. Adição de Embedding Posicional
        - No ImageEncoderViT() pode ser passado embeddings posicionais absolutos (no parametro pos_embed). Eles são somados ao embedding dos patches. Isso permite ao modelo considerar informações espaciais sobre a posição dos patches dentro da imagem.
        - O tamanho do pos_embed é configurado para (1x64x64x768), o que é compatível com o tamanho de mapa de patches.
    
    3. Passagem pelos Blocos Transformer
        - A classe Block() é o componente do transformer e implementa duas operações:
            1. Atenção Multi-Cabeça (Attention):
                - É aplicada no tensor de entrada, após normalização (norm1).
                - Caso window_size > 0, o bloco realiza a partição em janelas para limitar a atenção a sub-regiões (atenção local). As funções window_partition e window_unpartition garantem que o processamento volte ao formato original após a operação.
                - window_partition() e window_unpartition() são usadas para dividir e recompor a imagem (ou tensores 2D) em blocos menores (janelas) e, em seguida, reconectá-las. Ao que parece, esse é um processo comum em arquiteturas com Transformer, onde as operações de atenção são aplicadas localmente a janelas não sobrepostas para reduzir o custo computacional.
                    - window_partition():
                        - O objetivo é dividir um tensor de entrada em pequenas janelas (submatrizes) de tamanho fixo (window_size=14 por padrão). Se necessário, a função adiciona preenchimento (padding) para garantir que o tamanho seja divisível por (window_size=14 por padrão).
                        - Entrada:
                            - x: (BxHxWxC)
                            - window_size=14: tamanho da janela a ser usada.
                        - Saída:
                            - windows: tensor particionado em janelas de forma [B*num_windows, window_size, window_size, C]
                            - (Hp, Wp): altura e largura após o padding.
                        - Passo a passo:
                            1. Cálculo do preenchimento necessário:
                                ```bash
                                    pad_h = (window_size - H % window_size) % window_size
                                    pad_w = (window_size - W % window_size) % window_size
                            - Se a altura ou largura não for divisívvel pelo tamanho da janela, é adicionado o preenchimento para garantir divisibilidade.
                            2. Adição de preenchimento ao tensor:
                                ```bash
                                    x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
                            - Isso vai criar uma nova dimensão [Hp, Wp], onde:
                                - Hp = H + pad_h
                                - Wp = W + pad_w
                            3. Reorganização em blocos (janelas):
                                ```bash
                                    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
                            - Aqui o tensor é dividido em janelas de tamanho fixo.
                            4. Permutação e reshaping:
                                ```bash
                                    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
                            - Cada janela é transformada em um tensor de forma [B*num_windows, window_size, window_size, C].
                    - window_unpartition():
                        - Essa função tem como objetivo reverter o processo de particionamento feito pela função window_partition(). Ele reconstrói o tensor original das janelas, removendo qualquer preenchimento (padding) adicionado anteriormente.
                        - Entrada:
                            - windows: tensor de janelas de forma [B*num_windows, window_size, window_size, C].
                            - window_size: tamanho da janela usada para particionar.
                            - pad_hw: altura e largura com padding (Hp, Wp).
                            - hw: altura e largura original antes do padding (H, W).
                        - Saída:
                            - x: tensor reconstruído de forma [B, H, W, C].
                        - Passo a passo:
                            1. Cálculo do número de batches:
                                ```bash
                                B = windows.shape[0] // (Hp * Wp // window_size // window_size)
                            - Isso determina quantos batches existem com base no número total de janelas e suas dimensões.
                            2. Reorganizar janelas em blocos maiores:
                                ```bash
                                    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
                            - As janelas são rearranjadas para criar uma estrutura de forma [B, Hp, Wp, C].
                            3. Permutar e reconectar os blocos:
                                ```bash
                                    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
                            4. Remover o padding (se necessário):
                                ```bash
                                    if Hp > H or Wp > W:
                                        x = x[:, :H, :W, :].contiguous()
                            - O tensor é cortado para retornar ás dimensões originais [H,W].
            2. Perceptron Multi-Camada (MLP):
                - Após a aplicação da atenção, um MLP com ativação é usado para processar os dados linearmente.
                - O resultado é adicionado ao input original (residual connection).
            3. Fluxo do Block(): Entrada -> Normalização -> Atenção -> Soma com entrada original -> Normalização -> MLP -> Soma com anterior.
        - A classe Attention() é o componente de atenção que realiza a seguinte sequencia de operações:
            1. Projeção de QKV:
                - Usa uma única camada linear para calcular queries (q), keys (k) e values (v), organizados para múltiplas cabeças.
            2. Cálculo do mapa de atenção:
                - Escala as queries (q) e realiza o produto escalar com as keys transpostas.
                - Caso use_rel_pos=True, é adicionado embeddings posicionais relativas (rel_pos_h) e (rel_pos_w) com a função add_decomposed_rel_pos().
                - A função add_decomposed_rel_pos() implementa embeddings posicionais relativas decompostas. Os embeddings posicionais relativos ajudam o modelo a entender as relações espaciais entre diferentes partes de uma entrada (como uma imagem), sem a necessidade de incorporar explicitamente a posição absoluta dos elementos.
                    - get_rel_pos():
                        - O objetivo é obter os embeddings posicionais relativos para duas dimensões específicas: altura (q_size) e largura (w_size). Isso ajusta os embeddings relativos predefinidos para corresponder às dimensões atuais do query e key.
                        - Calcula embeddings relativas ajustadas às dimensões das queries e keys.
                        - Interpola os embeddings, se necessário, para garantir compatibilidade.
                    - add_decomposed_rel_pos():
                        - O objetivo é adicionar embeddings posicionais relativos ao mapa de atenção no processo de cálculo da autoatenção. Os embeddings são calculados separadamente para as dimensões de altura e largura.
                        - Adiciona as contribuições das embeddings relativas em altura (rel_pos_h) e largura (rel_pos_w) ao mapa de atenção.
                        - Usa operações eficientes como torch.einsum para combinar embeddings com o tensor de atenção.
                    - Essa abordagem reduz a complexidade espacial e melhora a eficiência para entradas grandes.
            3. Softmax e atualização:
                - Aplica softmax ao mapa de atenção para normalização.
                - Multiplica pelo vetor de valores (v), resultando na saída transformada.
            4. Projeção final:
                - Uma camada linear projeta a saída de volta para a dimensão original.
    4. Passagem pelo Neck
        - A última etapa do encoder é o módulo neck, onde:
            1. reduz a dimensão dos canais:
                - O mapa de features é transformado de 64x64x768 para 64x64x256, usando convolução.
            2. normalização dos valores dos pixels em cada canal.
        - O tensor final é retornado no formato BxCxHxW.

3. PromptEncoder
    - A classe PromptEncoder() é responsável por processar diferentes tipos de entradas (prompts) usadas como contexto para o modelo de segmentação SAM.
    - O objetivo geral do PromptEncoder() é pegar entradas específicas de prompts (pontos, caixas delimitadoras e máscaras) e gerar dois tipos de embeddings:
      1. Sparse embeddings (esparsos): representações vetoriais de pontos e caixas.
      2. Dense embeddings (densos): representações 2D de máscaras, com o mesmo formato espacial do mapa de características do modelo. 
    - Esses embeddings servem como entrada para o decodificador de máscaras do SAM.
    - Parametros de entrada:
        - embed_dim: Dimensão dos embeddings gerados.
        - image_embedding_size: Dimensão espacial do mapa de características da imagem (saída do encoder de imagem).
        - input_image_size: Dimensão da imagem original.
        - mask_in_chans: Número de canais usados para processar as máscaras.
        - activation: Função de ativação usada no processamento de máscaras (default: GELU).
    - Para calcular o positional encoding (pe_layer), é usado a classe PositionEmbeddingRandom() para gerar codificações posicionais baseadas em frequências espaciais aleatórias.
    - Os point embeddings são vetores aprendidos que codificam pontos positivos e negativos, além de cantos de caixas.
        - Além disso, tem os not_a_point_embed, que são embeddings para representar a auseência de pontos.
    - Mask Downscaling:
        - Conjunto de camadas convolucionais e de normalização que reduz o tamanho espacial da máscara para corresponder ao mapa de características da imagem e transforma a máscara em embeddings densos.
        - Em caso de não usar máscaras como prompts, é usado o no_mask_embed, que é um vetor aprendido usado quando nenhuma máscara é fornecida.
    - TODO: ADICIONAR MAIS DETALHES...
4. MaskDecoder
    - O MaskDecoder é uma arquitetura baseada em transformers que prevê máscaras segmentadas em imagens, usando embeddings de imagem e prompt.
        1. Entrada:
            - image_embeddings: Representações da imagem extraídas pelo image encoder.
            - image_pe: Codificações posicionais associadas às embeddings da imagem.
            - sparse_prompt_embeddings: Embeddings de prompts esparsos (que pode ser pontos ou caixas delimitadoras).
            - dense_prompt_embeddings: Embeddings de prompts densos (máscaras iniciais).
            - multimask_output: Indica se o modelo deve prever múltiplas máscaras ou uma única.
        2. Tokens especiais
            - iou_token: Um token único treinável que aprende a prever a qualidade (IoU - Intersection over Union) das máscaras.
            - mask_tokens: Um conjunto de tokens que são usados para prever as máscaras segmentadas. Um token extra é adicionado para suportar múltiplas saídas (desambiguação de máscaras).
        3. Rede de upscaling
            - output_upscaling: Uma sequência de convoluções transpostas para aumentar a resolução das máscaras preditas.
            - Cada máscara passa por uma MLP que ajusta a geração das máscaras para uma granularidade mais precisa.
        4. Predição de qualidade
            - Um MLP (iou_prediction_head) avvalia a qualidade das máscaras preditas, gerando uma métrica IoU.
        5. Transformer
            - Recebe representações da imagem e prompt.
            - Os tokens especiais para prever máscaras e qualidade.
            - Gera saídas (hs) que são usadas para reconstruir as máscaras e avaliar sua qualidade.
    - Funcionamento:
        - Ao chamar forward(), é passado o fluxo para predict_masks(), que:
            - faz concatenação de tokens: os mask_tokens e iou_token sao concatenados com os sparse_prompt_embeddings para formar os tokens de entrada do transformer.
            - propaga pelo transformer: os tokens interagem com as embeddings da imagem e os prompts, gerando representações refinadas para máscaras (mask_tokens_out) e qualidade (iou_token_out).
            - upscaling das máscaras: as embeddings da imagem passam por convoluções transpostas para aumentar sua resolução (output_upscaling).
            - Cada mask_token é combinado com essas representações usando hypernetworks individuais (output_hypernetworks_mlps), que ajudam na previsão precisa das máscaras.
            - cálculo das máscaras: As representações refinadas (hyper_in) são multiplicadas pela saída upscalada, gerando as máscaras finais com a resolução desejada.
            - predição de qualidade: O iou_token_out é processado pelo MLP (iou_prediction_head), gerando uma previsão da qualidade para cada máscara.

### Exemplo de fluxo do SAM
- Dado uma imagem (255,701,3):
    - Imagem Original:           (255, 701, 3)
        - -> Pré-processamento:    (1024, 1024, 3)
        - -> Patch Embedding:      (64, 64, 768)
        - -> Blocos Transformer:   (64, 64, 768)
        - -> Saída do Encoder:     (64, 64, 256)

    - Prompts:
        - -> Sparse Embeddings:    (N_prompts, 256)
        - -> Dense Embeddings:     (64, 64, 256)

    - Decoder:
        - -> Tokens de Entrada:    (N_prompts + N_masks + 1, 256)
        - -> Máscaras Upscaladas:  (256, 256, 32)
        - -> Saída Final:          (N_masks, 256, 256)