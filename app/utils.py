import torch
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from collections import defaultdict
from Depth_Anything_V2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

CLASS_TRANSLATIONS = {
    0: "pessoa",
    1: "bicicleta",
    2: "carro",
    3: "moto",
    4: "avião",
    5: "ônibus",
    6: "trem",
    7: "caminhão",
    8: "barco",
    9: "semáforo",
    10: "hidrante",
    11: "placa de pare",
    12: "parquímetro",
    13: "banco",
    14: "pássaro",
    15: "gato",
    16: "cachorro",
    17: "cavalo",
    18: "ovelha",
    19: "vaca",
    20: "elefante",
    21: "urso",
    22: "zebra",
    23: "girafa",
    24: "mochila",
    25: "guarda-chuva",
    26: "bolsa",
    27: "gravata",
    28: "mala",
    29: "frisbee",
    30: "esquis",
    31: "snowboard",
    32: "bola esportiva",
    33: "pipa",
    34: "taco de beisebol",
    35: "luva de beisebol",
    36: "skate",
    37: "prancha de surfe",
    38: "raquete de tênis",
    39: "garrafa",
    40: "taça de vinho",
    41: "copo",
    42: "garfo",
    43: "faca",
    44: "colher",
    45: "tigela",
    46: "banana",
    47: "maçã",
    48: "sanduíche",
    49: "laranja",
    50: "brócolis",
    51: "cenoura",
    52: "cachorro-quente",
    53: "pizza",
    54: "donut",
    55: "bolo",
    56: "cadeira",
    57: "sofá",
    58: "planta em vaso",
    59: "cama",
    60: "mesa de jantar",
    61: "vaso sanitário",
    62: "televisão",
    63: "notebook",
    64: "mouse",
    65: "controle remoto",
    66: "teclado",
    67: "celular",
    68: "micro-ondas",
    69: "forno",
    70: "torradeira",
    71: "pia",
    72: "geladeira",
    73: "livro",
    74: "relógio",
    75: "vaso",
    76: "tesoura",
    77: "urso de pelúcia",
    78: "secador de cabelo",
    79: "escova de dentes"
}

def pluralize(word):
    """Função auxiliar na formatação da descrição de imagem"""
    if word.endswith("l"):
        return word[:-1] + "is"
    elif word.endswith("m"):
        return word[:-1] + "ns"
    elif word.endswith("ão"):
        return word[:-2] + "ões"
    elif word.endswith(("z", "r", "s")):
        return word + "es"
    else:
        return word + "s"

def format_description(detections, image_width):
    """Função para formatar descrição de imagem"""
    if not detections:
        return "Não foi identificado nenhum objeto na imagem."
    
    grouped_objects = defaultdict(list)

    for obj in detections:
        label = obj['class']
        distance = round(obj['distance'])
        x1,_,x2,_ = obj['box']

        third = image_width / 3
        # Calcular quanto da largura do objeto está em cada terço da imagem
        left_overlap = max(0, min(x2, third) - x1)
        center_overlap = max(0, min(x2, 2 * third) - max(x1, third))
        right_overlap = max(0, x2 - max(x1, 2 * third))

        overlaps = {
            "à sua esquerda": left_overlap,
            "na sua frente": center_overlap,
            "à sua direita": right_overlap
        }

        # A direção será aquela onde o objeto tem maior sobreposição
        direction = max(overlaps, key=overlaps.get)

        key = (distance, direction)
        grouped_objects[key].append(label)

    phrases = []

    for (distance, direction), labels in grouped_objects.items():
        count = len(labels)

        if count == 1:
            object_str = f"um {labels[0]}"
        else:
            label_counts = defaultdict(int)
            for label in labels:
                label_counts[label] += 1

            parts = []
            for label, qty in label_counts.items():
                if qty == 1:
                    parts.append(f"um {label}")
                else:
                    parts.append(f"{qty} {pluralize(label)}")
            object_str = " e ".join(parts)

        phrases.append(f"{object_str} a {distance} metros {direction}")

    intro = "Foi identificado na imagem " if len(detections) == 1 else "Foram identificados na imagem "
    return intro + ", ".join(phrases) + "."

def detect_objects(model, image):
    """Função para detectar objetos na imagem usando o modelo YOLO passado"""
    # Fazer inferencia com YOLO
    results = model(image)

    detections = []
    for result in results:
        for box in result.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0].item())
            # class_name = model.names[class_id]
            class_name = CLASS_TRANSLATIONS[class_id]

            detections.append({"class":class_name, "box": [x1,y1,x2,y2]})
    
    return detections


def load_depth_anything():
    """Função para carregar o modelo Depth Anything V2 e os checkpoints"""
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    dataset = 'hypersim' # 'hypersim' ou 'vkitti'
    encoder = 'vitb' # 'vitl', 'vitb' ou 'vits'

    try:
        model = DepthAnythingV2(**{**model_configs[encoder]})
        print("Modelo Depth Anything V2 carregado com sucesso.")
    except Exception as e:
        print("Erro ao carregar modelo Depth Anything V2", e)
        return None

    try:
        model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
        print('Checkpoints carregado com sucesso.')
    except Exception as e:
        print("Erro ao carregar os checkpoints do modelo DepthAnythingV2", e)
        return None

    model.eval()

    return model

def generate_depth_map(model, image):
    """"Função para gerar o mapa de profundidade da imagem usando o modelo passado"""
    # image = cv2.imread(image)  
    image_np = np.array(image)
    depth_map = model.infer_image(image_np) # HxW depth map in meters in numpy

    return depth_map

def calculate_object_distances(detections, depth_map):
    """Função que retorna as detecções com o calculo da distancias usando o mapa de profundidade."""
    for obj in detections:
        x1,y1,x2,y2 = obj['box']
        object_depth = depth_map[y1:y2, x1:x2]  # Recortar a região da bounding box
        median_depth = float(np.median(object_depth))  # Usamos a mediana para evitar ruídos
        
        obj['distance'] =  median_depth

    return detections