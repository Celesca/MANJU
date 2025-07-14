#!/bin/bash

# MANJU Voice Chatbot Docker Setup Script
echo "🐳 MANJU Voice Chatbot Docker Setup"
echo "=================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first:"
    echo "   https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first:"
    echo "   https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker and Docker Compose are available"
echo ""

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p audio_uploads results temp

# Build and start services
echo "🔨 Building and starting services..."
echo "This may take several minutes on first run..."

# Build the application
docker-compose build

# Start services
docker-compose up -d

# Wait for services to be ready
echo ""
echo "⏳ Waiting for services to start..."
sleep 10

# Check service status
echo ""
echo "📊 Service Status:"
docker-compose ps

# Wait for Ollama to download models
echo ""
echo "📥 Waiting for Ollama to download phi3 model..."
echo "This may take several minutes depending on your internet connection..."

# Monitor Ollama setup
timeout=300  # 5 minutes
counter=0
while [ $counter -lt $timeout ]; do
    if docker-compose logs ollama-setup 2>/dev/null | grep -q "Phi3 model downloaded successfully"; then
        echo "✅ Phi3 model downloaded successfully!"
        break
    fi
    echo "   Still downloading... ($counter/$timeout seconds)"
    sleep 10
    counter=$((counter + 10))
done

if [ $counter -ge $timeout ]; then
    echo "⚠️  Model download timeout. You can check manually with:"
    echo "   docker-compose logs ollama-setup"
fi

echo ""
echo "🎉 Setup Complete!"
echo ""
echo "📋 Access Information:"
echo "   🌐 Voice Chatbot: http://localhost:8501"
echo "   🧠 Ollama API: http://localhost:11434"
echo ""
echo "🔧 Useful Commands:"
echo "   📊 Check status: docker-compose ps"
echo "   📜 View logs: docker-compose logs -f"
echo "   🛑 Stop services: docker-compose down"
echo "   🔄 Restart: docker-compose restart"
echo "   🗑️  Remove all: docker-compose down -v"
echo ""
echo "💡 Troubleshooting:"
echo "   - If Ollama is not responding, wait a few minutes for model download"
echo "   - Check logs with: docker-compose logs ollama"
echo "   - Restart Ollama: docker-compose restart ollama"
echo ""
echo "🎤 Ready to use! Open http://localhost:8501 in your browser"
