# Installation et Lancement d'h2ogpt

## Mise en place de l'environnement Python 

### Créez l'environnement Python via Conda

    spack load miniconda3@22.11.1
    conda init
    conda create -n h2ogpt -y
    conda activate h2ogpt
    conda install python=3.10 -c conda-forge -y

### Testez le fonctionnement de Python 

    python --version
    python -c "import os, sys ; print('hello world')"

## Installation de h2ogpt
    
### Clonez le git de h2ogpt

    git clone https://github.com/h2oai/h2ogpt.git
    cd h2ogpt

### Installez le cudatoolkit

    conda install cudatoolkit=11.8 -c conda-forge -y
    export CUDA_HOME=$CONDA_PREFIX 
    export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu118"
    
    
### Installez les dépendances
    pip uninstall -y pandoc pypandoc pypandoc-binary flash-attn
    pip install -r requirements.txt
    pip install -r reqs_optional/requirements_optional_langchain.txt
    pip install -r reqs_optional/requirements_optional_gpt4all.txt
    pip install -r reqs_optional/requirements_optional_langchain.gpllike.txt
    pip install -r reqs_optional/requirements_optional_langchain.urls.txt
    conda install -y -c conda-forge pygobject
    pip install -r reqs_optional/requirements_optional_doctr.txt
    pip install onnxruntime==1.15.0 onnxruntime-gpu==1.15.0
    pip uninstall weasyprint
    conda install -y -c conda-forge weasyprint
    pip install pydub==0.25.1 librosa==0.10.1 ffmpeg==1.4 yt_dlp==2023.10.13
    pip install wavio==0.0.8
    pip install torchaudio soundfile==0.12.1
    pip install TTS deepspeed noisereduce pydantic==1.10.13 emoji ffmpeg-python==0.2.0 trainer pysbd coqpit
    pip install transformers==4.35.0
    pip install cutlet==0.3.0 langid==1.1.6 g2pkk==0.1.2 jamo==0.4.1 gruut[de,es,fr]==2.2.3 jieba==0.42.1
    pip uninstall -y auto-gptq
    pip install https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.4.2/auto_gptq-0.4.2+cu118-cp310-cp310-linux_x86_64.whl
    pip install optimum==1.14.1
    pip uninstall -y exllama ; pip install https://github.com/jllllll/exllama/releases/download/0.0.13/exllama-0.0.13+cu118-cp310-cp310-linux_x86_64.whl --no-cache-dir
    pip uninstall -y llama-cpp-python-cuda
    pip install https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.2.18+cu118-cp310-cp310-manylinux_2_31_x86_64.whl
    pip install flash-attn==2.3.1.post1 --no-build-isolation

## Lancement de h2ogpt

### Vérifiez si Cuda est disponible et visible 
    python
    import torch
    print(torch.cuda.is_available())

### Lancer h2ogpt via un job slurm
    spack load cuda@11.8.0%gcc@=11.3.1
    srun --gres=gpu:1 --mem=256G --reservation=M2CHPS python generate.py --share=False --gradio_offline_level=1 --base_model=TheBloke/Mistral-7B-v0.1-GGUF --score_model=None --prompt_type=human_bot

### Récupérez le port utilisé par h2ogpt et mettez en place un tunnel ssh (port forwarding) sur votre propre machine pour accéder à l'appli Web h2ogpt

    ssh -J votre_nom_user@juliet.mesonet.fr -L port_h20gpt:localhost:port_h2ogpt votre_nom_user@juliet2  -N

### Accédez à l'interface Web via votre navigateur à l'adresse localhost:port_h2ogpt