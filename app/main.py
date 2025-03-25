import base64
from flask import Blueprint, request, jsonify
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

main_bp = Blueprint('main', __name__)

try:
    model = YOLO("yolo11x.pt")
except Exception as e:
    print("Erro ao carregar modelo YOLO", e)

def decode_base64_image(image_base64):
    """Função para converter imagem base64 para objeto de imagem."""
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]  # Remove cabeçalho "data:image/jpeg;base64,"
    
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        raise ValueError(f"Erro ao decodificar imagem: {str(e)}")

@main_bp.route('/')
def home():
    return "Hello world"

@main_bp.route("/upload", methods=['POST'])
def upload_file():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Nenhum dado JSON recebido"}), 400

        image_base64 = data.get("image")
        if not image_base64:
            return jsonify({"error": "Nenhuma imagem recebida"}), 400

        print("Imagem base64 recebida com sucesso")
        
        # Converte Base64 para imagem
        try:
            image = decode_base64_image(image_base64)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

        print("Imagem decodificada com sucesso, processando...")

        # Fazer inferencia com YOLO
        results = model(image)

        # Processar os resultados
        classes = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                class_name = model.names[class_id]

                classes.append(class_name)
                
        return jsonify({"class": classes})
    
    except Exception as e:
        return jsonify({"error": f"Erro inesperado: {str(e)}"}), 500