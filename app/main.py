import time
from ultralytics import YOLO
from flask import Blueprint, request, jsonify
from app.utils import (
    decode_base64_image, 
    format_description, 
    detect_objects, 
    load_depth_anything, 
    generate_depth_map, 
    calculate_object_distances
)

main_bp = Blueprint('main', __name__)
depth_model = load_depth_anything() # carrega modelo de profundidade

try:
    yolo_model = YOLO("yolo11n.pt")
except Exception as e:
    yolo_model = None
    print("Erro ao carregar modelo YOLO", e)

@main_bp.route('/')
def home():
    return "Hello, world"

@main_bp.route("/process_image", methods=['POST'])
def process_image():
    try:
        if not yolo_model or not depth_model:
            return jsonify({"error": "Modelo YOLO ou Depth Anything não foi carregado corretamente."}), 400

        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Nenhuma imagem enviada no corpo da requisição."}), 400

        image_base64 = data.get("image")
        print("Imagem recebida com sucesso.")
        
        # Decodifica imagem base64 para formato OpenCV
        try:
            image = decode_base64_image(image_base64)
            image_width, _ = image.size
        except Exception as e:
            return jsonify({"error": f"Erro ao decodificar image: {str(e)}"}), 400
        
        print("Imagem recebida e decodificada. Iniciando detecção...")
        
        t_start = time.perf_counter()

        detections = detect_objects(yolo_model, image)
        if len(detections) == 0:
            return jsonify({"error": "Nenhum objeto detectado na imagem."}), 400
        
        depth_map = generate_depth_map(depth_model, image)
        results = calculate_object_distances(detections, depth_map)
        description  = format_description(results, image_width)

        t3 = time.perf_counter()
        print(f"Processamento finalizado em {(t3 - t_start):.2f} segundos.")

        return jsonify({"descricao": description , "resultados": results}), 200
    
    except Exception as e:
        print(f"Erro interno: {str(e)}")
        return jsonify({"error": f"Erro inesperado: {str(e)}"}), 500
