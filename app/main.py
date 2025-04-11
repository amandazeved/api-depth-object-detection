import time
import os
import gc
from PIL import Image
from ultralytics import YOLO
from flask import Blueprint, request, jsonify
from app.utils import ( 
    format_description, 
    detect_objects, 
    load_depth_anything, 
    generate_depth_map, 
    calculate_object_distances
)

main_bp = Blueprint('main', __name__)

UPLOAD_FOLDER = "tmp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@main_bp.route('/')
def home():
    return "Hello, world"

@main_bp.route("/process_image", methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada."}), 400
    
    try:
        depth_model = load_depth_anything() # carrega modelo de profundidade
        try:
            yolo_model = YOLO("yolo11n.pt")
        except Exception as e:
            yolo_model = None
            print("Erro ao carregar modelo YOLO", e)
        
        if not yolo_model or not depth_model:
            return jsonify({"error": "Modelo YOLO ou Depth Anything não foi carregado corretamente."}), 400

        file = request.files['image']
        image = Image.open(file.stream)
        image_width, _ = image.size
        
        print("Imagem recebida. Iniciando detecção...")
        
        t_start = time.perf_counter()

        detections = detect_objects(yolo_model, image)
        if len(detections) == 0:
            return jsonify({"error": "Nenhum objeto detectado na imagem."}), 400
        
        depth_map = generate_depth_map(depth_model, image)
        results = calculate_object_distances(detections, depth_map)
        description  = format_description(results, image_width)

        t3 = time.perf_counter()
        print(f"Processamento finalizado em {(t3 - t_start):.2f} segundos.")
        print(description)
        print(results)

        return jsonify({"descricao": description , "resultados": results}), 200
    
    except Exception as e:
        print(f"Erro interno: {str(e)}")
        return jsonify({"error": f"Erro inesperado: {str(e)}"}), 500


@main_bp.route("/detect_objects", methods=['POST'])
def detect_objects_route():
    if 'image' not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada."}), 400
    
    try:
        try:
            yolo_model = YOLO("yolo11n.pt")
            print("Carregado modelo YOLO.")
        except Exception as e:
            yolo_model = None
            print("Erro ao carregar modelo YOLO", e)
        
        if not yolo_model:
            return jsonify({"error": "Modelo YOLO não foi carregado corretamente."}), 400
        
        file = request.files['image']
        image = Image.open(file.stream)
        
        print("Imagem recebida. Iniciando detecção...")

        detections = detect_objects(yolo_model, image)
        if len(detections) == 0:
            return jsonify({"error": "Nenhum objeto detectado na imagem."}), 400

        return jsonify({"detections": detections}), 200

    except Exception as e:
        print(f"Erro interno: {str(e)}")
        return jsonify({"error": f"Erro inesperado: {str(e)}"}), 500
    finally:
        # Limpa memória
        del image, yolo_model, detections
        gc.collect()

@main_bp.route("/calculate_distance", methods=['POST'])
def calculate_distance_route():
    if 'image' not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada."}), 400
    
    try:
        data = request.get_json()
        detections = data.get("detections")

        if not detections:
            return jsonify({"error": "Faltando 'detections'."}), 400
    
        
        depth_model = load_depth_anything() # carrega modelo de profundidade
        
        if not depth_model:
            return jsonify({"error": "Modelo Depth Anything V2 não foi carregado corretamente."}), 400
        
        file = request.files['image']
        image = Image.open(file.stream)
        image = image.resize((640, 480))
        image_width, _ = image.size
        
        print("Imagem recebida. Gerando mapa de profundidade e calculando distâncias de objetos...")

        depth_map = generate_depth_map(depth_model, image)
        results = calculate_object_distances(detections, depth_map)
        description = format_description(results, image_width)

        return jsonify({"descricao": description, "resultados": results}), 200

    except Exception as e:
        print(f"Erro interno: {str(e)}")
        return jsonify({"error": f"Erro inesperado: {str(e)}"}), 500
    finally:
        # Limpa memória
        del image, depth_map, detections, results, description
        gc.collect()
