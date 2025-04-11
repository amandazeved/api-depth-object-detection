import time
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
