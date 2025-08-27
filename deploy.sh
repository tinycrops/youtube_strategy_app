#!/bin/bash

# YouTube Strategy Analyzer Deployment Script
# This script handles deployment to various platforms

set -e

echo "🚀 YouTube Strategy Analyzer Deployment Script"
echo "=============================================="

# Check if .env file exists, create if not
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating template..."
    cat > .env << EOL
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
EOL
    echo "✅ Created .env template. Please fill in your API keys."
fi

# Function to deploy locally
deploy_local() {
    echo "📦 Deploying locally..."
    
    # Install requirements
    echo "Installing Python requirements..."
    pip install -r requirements.txt
    
    # Create necessary directories
    mkdir -p static/videos static/clips Channel_analysis/outputs/channels
    
    # Copy sample data if available
    if [ -d "Channel_analysis/outputs/channels" ] && [ "$(ls -A Channel_analysis/outputs/channels)" ]; then
        echo "✅ Sample channel data found"
    else
        echo "⚠️  No sample data found. You can add your own CSV files to Channel_analysis/outputs/channels/"
    fi
    
    echo "🎉 Local setup complete!"
    echo "Run: streamlit run youtube_strategy_app.py"
}

# Function to deploy with Docker
deploy_docker() {
    echo "🐳 Deploying with Docker..."
    
    # Build and run with Docker Compose
    docker-compose build
    docker-compose up -d
    
    echo "🎉 Docker deployment complete!"
    echo "Access your app at: http://localhost:8501"
}

# Function to deploy to Streamlit Cloud
deploy_streamlit_cloud() {
    echo "☁️  Preparing for Streamlit Cloud deployment..."
    
    # Check if git repo exists
    if [ ! -d ".git" ]; then
        echo "Initializing git repository..."
        git init
        git add .
        git commit -m "Initial commit for YouTube Strategy Analyzer"
    fi
    
    echo "📝 Instructions for Streamlit Cloud deployment:"
    echo "1. Push this repository to GitHub"
    echo "2. Go to https://streamlit.io/cloud"
    echo "3. Connect your GitHub repo"
    echo "4. Set main file as: youtube_strategy_app.py"
    echo "5. Add your API keys in the secrets section"
    echo "6. Deploy!"
    
    echo "🔧 Required secrets for Streamlit Cloud:"
    echo "   GEMINI_API_KEY = 'your_api_key'"
    echo "   OPENAI_API_KEY = 'your_api_key'"
    echo "   ANTHROPIC_API_KEY = 'your_api_key'"
}

# Function to deploy to Heroku
deploy_heroku() {
    echo "🟣 Preparing for Heroku deployment..."
    
    # Create Procfile
    echo "web: streamlit run youtube_strategy_app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile
    
    # Create runtime.txt
    echo "python-3.11.7" > runtime.txt
    
    # Create setup.sh for Streamlit configuration
    cat > setup.sh << 'EOL'
#!/bin/bash
mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = \$PORT\n\
" > ~/.streamlit/config.toml
EOL
    
    chmod +x setup.sh
    
    echo "📝 Heroku deployment files created!"
    echo "Next steps:"
    echo "1. Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli"
    echo "2. heroku login"
    echo "3. heroku create your-app-name"
    echo "4. heroku config:set GEMINI_API_KEY=your_api_key"
    echo "5. git add . && git commit -m 'Deploy to Heroku'"
    echo "6. git push heroku main"
}

# Function to deploy to Railway
deploy_railway() {
    echo "🚂 Preparing for Railway deployment..."
    
    # Create railway.json
    cat > railway.json << 'EOL'
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE"
  },
  "deploy": {
    "startCommand": "streamlit run youtube_strategy_app.py --server.port=$PORT --server.address=0.0.0.0",
    "healthcheckPath": "/",
    "healthcheckTimeout": 100
  }
}
EOL
    
    echo "📝 Railway deployment file created!"
    echo "Next steps:"
    echo "1. Install Railway CLI: https://docs.railway.app/quick-start"
    echo "2. railway login"
    echo "3. railway init"
    echo "4. railway add --database postgresql (optional)"
    echo "5. Set environment variables in Railway dashboard"
    echo "6. railway up"
}

# Main deployment menu
echo "Please choose your deployment option:"
echo "1) Local development"
echo "2) Docker (local)"
echo "3) Streamlit Cloud"
echo "4) Heroku"
echo "5) Railway"
echo "6) Exit"

read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        deploy_local
        ;;
    2)
        deploy_docker
        ;;
    3)
        deploy_streamlit_cloud
        ;;
    4)
        deploy_heroku
        ;;
    5)
        deploy_railway
        ;;
    6)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "🎯 Deployment Summary:"
echo "======================"
echo "✅ YouTube Strategy Analyzer is ready for deployment"
echo "📊 Features include:"
echo "   • Channel performance analysis"
echo "   • Content strategy recommendations"
echo "   • Thumbnail analysis (AI-powered)"
echo "   • Video content pattern detection"
echo "   • Interactive visualizations"
echo ""
echo "🔗 For updates and support, visit:"
echo "   https://github.com/your-repo/youtube-strategy-analyzer"
echo ""
echo "Happy analyzing! 🎉"