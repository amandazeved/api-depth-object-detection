FROM python:3.11.9-slim

# Instala dependências de sistema (libGL para OpenCV)
RUN apt-get update && apt-get install -y libgl1 && apt-get clean

# Define diretório de trabalho
WORKDIR /app

# Copia os arquivos do projeto
COPY . .

# Instala dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta usada
EXPOSE 8080

# Comando para iniciar a aplicação
CMD ["gunicorn", "app:create_app()", "--bind", "0.0.0.0:8080"]