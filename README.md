# Depth & Object Detection API
API em Flask para utilizar YOLO para detecção em imagens.

Este projeto é uma **API Flask** que utiliza dois modelos de visão computacional para analisar imagens:

- **YOLOv11 (You Only Look Once)** para detecção de objetos.
- **Depth Anything V2** para estimativa de profundidade monocular.

A API é capaz de:
- Detectar objetos em uma imagem (ex: pessoas, carros, copos, etc.).
- Calcular a **distância de cada objeto** até a câmera com base no mapa de profundidade.
- Retornar uma **descrição legível** com as informações detectadas.

## Como executar localmente
1. Clone o repositótio:
```bash
git clone https://github.com/amandazeved/api-depth-object-detection.git
cd api-flask-yolo
```

2. Crie um ambiente virtual (opcional):
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Execute o servidor Flask:
```bash
flask run
```

## Créditos

Este projeto reutiliza partes do código do repositório:

- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)  
  Licenciado sob [MIT License](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/LICENSE)  
  © Alibaba Group Holding Limited

## Licença

Este projeto está licenciado sob os termos da licença MIT.  
Veja o arquivo [LICENSE](./LICENSE) para mais detalhes.
