# Sistema de Tradução Fala-para-Fala

Este projeto implementa um sistema modular de tradução de fala-para-fala (Inglês para Português) com baixa latência, utilizando diversos backends para Speech-to-Text (STT), Tradução e Text-to-Speech (TTS).

---

### Configuração e Execução

Siga os passos abaixo para configurar e rodar o sistema:

1.  **Crie um Ambiente Virtual (venv):**
    É altamente recomendado usar um ambiente virtual para gerenciar as dependências do projeto.

    ```bash
    python -m venv .venv
    ```

2.  **Ative o Ambiente Virtual:**
    * No Windows:
        ```bash
        .venv\Scripts\activate
        ```
    * No macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```

3.  **Instale as Dependências:**
    Com o ambiente virtual ativado, instale as bibliotecas necessárias. Certifique-se de ter um arquivo `requirements.txt` na raiz do seu projeto contendo todas as dependências (por exemplo, `sounddevice`, `numpy`, `SpeechRecognition`, `googletrans`, `pyttsx3`, `pygame`, `openai`, `whisper`, `groq`, `librosa`).

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure o Sistema:**
    * Copie o arquivo de exemplo de configuração. Se você tiver um `config_example.json`, use:
        ```bash
        cp config_example.json config.json
        ```
        Caso contrário, crie um arquivo chamado `config.json` na raiz do projeto com o seguinte conteúdo (ou similar, dependendo da sua configuração padrão):

        ```json
        {
          "sample_rate": 16000,
          "chunk_duration": 2.0,
          "silence_threshold": 0.01,
          "stt_backend": "google",
          "translation_backend": "groq",
          "tts_backend": "pyttsx3",
          "openai_api_key": null,
          "groq_api_key": "SUA_CHAVE_AQUI",
          "max_queue_size": 10,
          "processing_timeout": 10.0,
          "source_language": "en",
          "target_language": "pt",
          "log_level": "INFO"
        }
        ```
    * Abra o arquivo `config.json`.
    * Substitua o placeholder `"SUA_CHAVE_AQUI"` (ou `"YOUR_GROQ_API_KEY"`) pela sua **chave de API da Groq**. Você pode obtê-la em [console.groq.com](https://console.groq.com/).
    * Se for usar outros backends como OpenAI, preencha também `openai_api_key` no `config.json` ou defina-o via variável de ambiente.
    * Ajuste outros parâmetros no `config.json` conforme suas preferências (ex: `stt_backend`, `translation_backend`, `tts_backend`). Lembre-se de que o modelo `llama-3.1-70b-versatile` foi desativado, então certifique-se de que a configuração do Groq aponte para um modelo suportado, como `"llama3-8b-8192"` ou `"llama3-70b-8192"`.

5.  **Defina a Chave de API Groq (Recomendado):**
    Para maior segurança e flexibilidade, defina sua chave Groq como uma **variável de ambiente**. Isso é uma boa prática e evita que a chave fique diretamente no código ou em arquivos de configuração públicos. Se definida, ela geralmente substituirá a chave no `config.json` (dependendo da sua lógica de carregamento).

    * No Windows (para a sessão atual do terminal):
        ```bash
        set GROQ_API_KEY="SUA_CHAVE_AQUI"
        ```
    * No macOS/Linux (para a sessão atual do terminal):
        ```bash
        export GROQ_API_KEY="SUA_CHAVE_AQUI"
        ```
    Para que a variável seja persistente, adicione-a ao seu arquivo de perfil do shell (`.bashrc`, `.zshrc`, etc.) ou às variáveis de ambiente do sistema.

6.  **Execute o Sistema:**
    Com tudo configurado, execute o script principal. Se o seu arquivo principal se chama `main.py` (o que é uma convenção comum):

    ```bash
    python main.py
    ```
    Se o seu arquivo principal se chama `speech-to-speech-translator.py` (conforme mencionado no seu pedido):

    ```bash
    python test.py
    ```

    O sistema iniciará a captura de áudio e as traduções. Fale em inglês e ouça a tradução em português!