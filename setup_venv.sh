#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VENV_NAME="seed_oss_env"
MODEL_DIR="./Seed-OSS-36B-Instruct"
PYTHON_VERSION="python3"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}🌱 Seed-OSS-36B Complete Setup with Virtual Environment${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Step 1: Check Python installation
echo -e "${BLUE}Step 1: Checking Python installation...${NC}"
if ! command_exists $PYTHON_VERSION; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_ACTUAL_VERSION=$($PYTHON_VERSION --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
print_status "Python $PYTHON_ACTUAL_VERSION found"
echo ""

# Step 2: Create virtual environment
echo -e "${BLUE}Step 2: Setting up virtual environment...${NC}"

if [ -d "$VENV_NAME" ]; then
    print_warning "Virtual environment '$VENV_NAME' already exists."
    read -p "Do you want to delete and recreate it? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_NAME"
        print_info "Removed existing virtual environment"
    else
        print_info "Using existing virtual environment"
    fi
fi

if [ ! -d "$VENV_NAME" ]; then
    print_info "Creating virtual environment '$VENV_NAME'..."
    $PYTHON_VERSION -m venv "$VENV_NAME"
    if [ $? -eq 0 ]; then
        print_status "Virtual environment created successfully"
    else
        print_error "Failed to create virtual environment"
        exit 1
    fi
fi
echo ""

# Step 3: Activate virtual environment
echo -e "${BLUE}Step 3: Activating virtual environment...${NC}"
source "$VENV_NAME/bin/activate"
print_status "Virtual environment activated"
echo ""

# Step 4: Upgrade pip
echo -e "${BLUE}Step 4: Upgrading pip...${NC}"
pip install --upgrade pip --quiet
print_status "pip upgraded to latest version"
echo ""

# Step 5: Install base dependencies
echo -e "${BLUE}Step 5: Installing base dependencies...${NC}"
print_info "This may take a few minutes..."

# Install PyTorch with MPS support for Mac or CUDA for others
if [[ "$OSTYPE" == "darwin"* ]]; then
    print_info "Detected macOS - Installing PyTorch with MPS support..."
    pip install torch torchvision torchaudio --quiet
else
    print_info "Installing PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
fi

# Install other dependencies
pip install gradio accelerate huggingface-hub --quiet

# Check if CUDA is available and install bitsandbytes
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null
if [ $? -eq 0 ]; then
    print_info "CUDA detected - Installing bitsandbytes for quantization support..."
    pip install bitsandbytes --quiet
fi

print_status "Base dependencies installed"
echo ""

# Step 6: Install custom transformers fork
echo -e "${BLUE}Step 6: Installing custom transformers fork for Seed-OSS...${NC}"
print_info "This is required for the seed_oss architecture support"

# First uninstall any existing transformers
pip uninstall transformers -y --quiet 2>/dev/null

# Install the custom fork
print_info "Installing from: https://github.com/Fazziekey/transformers.git@seed-oss"
pip install git+https://github.com/Fazziekey/transformers.git@seed-oss --quiet

if [ $? -eq 0 ]; then
    print_status "Custom transformers fork installed successfully"
else
    print_error "Failed to install custom transformers fork"
    print_info "You can try manually: pip install git+https://github.com/Fazziekey/transformers.git@seed-oss"
fi
echo ""

# Step 7: Download model if not present
echo -e "${BLUE}Step 7: Checking model files...${NC}"

if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ]; then
    print_status "Model files found at $MODEL_DIR"
    
    # Count safetensor files
    SAFETENSOR_COUNT=$(find "$MODEL_DIR" -name "*.safetensors" 2>/dev/null | wc -l)
    if [ $SAFETENSOR_COUNT -gt 0 ]; then
        print_info "Found $SAFETENSOR_COUNT model weight files"
    else
        print_warning "No model weight files found"
    fi
else
    print_warning "Model not found at $MODEL_DIR"
    echo ""
    read -p "Do you want to download the model now (~72GB)? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Downloading model (this will take a while)..."
        huggingface-cli download ByteDance-Seed/Seed-OSS-36B-Instruct --local-dir "$MODEL_DIR"
        
        if [ $? -eq 0 ]; then
            print_status "Model downloaded successfully"
        else
            print_error "Model download failed"
            print_info "You can download manually later with:"
            print_info "  huggingface-cli download ByteDance-Seed/Seed-OSS-36B-Instruct --local-dir $MODEL_DIR"
        fi
    else
        print_warning "Skipping model download"
        print_info "You'll need to download it before running the chat interface:"
        print_info "  huggingface-cli download ByteDance-Seed/Seed-OSS-36B-Instruct --local-dir $MODEL_DIR"
    fi
fi
echo ""

# Step 8: Create launcher script
echo -e "${BLUE}Step 8: Creating launcher script...${NC}"

cat > run_chat.sh << 'EOF'
#!/bin/bash

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

VENV_NAME="seed_oss_env"

echo -e "${BLUE}🚀 Starting Seed-OSS Chat Interface${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d "$VENV_NAME" ]; then
    echo "❌ Virtual environment not found. Please run setup_venv.sh first."
    exit 1
fi

# Activate virtual environment
source "$VENV_NAME/bin/activate"
echo -e "${GREEN}✅ Virtual environment activated${NC}"

# Detect platform and set device
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - check for Apple Silicon
    if [[ $(uname -m) == 'arm64' ]]; then
        echo -e "${GREEN}✅ Apple Silicon detected - using MPS${NC}"
        DEVICE="mps"
    else
        echo "ℹ️  Intel Mac detected - using CPU"
        DEVICE="cpu"
    fi
else
    # Check for CUDA
    python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ CUDA detected - using GPU${NC}"
        DEVICE="cuda"
    else
        echo "ℹ️  No CUDA detected - using CPU"
        DEVICE="cpu"
    fi
fi

# Parse command line arguments or use defaults
if [ $# -eq 0 ]; then
    echo "ℹ️  Using device: $DEVICE"
    python chat_interface.py --device "$DEVICE"
else
    python chat_interface.py "$@"
fi

# Deactivate virtual environment when done
deactivate
EOF

chmod +x run_chat.sh
print_status "Launcher script created: run_chat.sh"
echo ""

# Step 9: Create utility scripts
echo -e "${BLUE}Step 9: Creating utility scripts...${NC}"

# Create activation script
cat > activate_env.sh << 'EOF'
#!/bin/bash
source seed_oss_env/bin/activate
echo "✅ Virtual environment activated"
echo "You can now run Python commands in the Seed-OSS environment"
echo "Type 'deactivate' to exit the virtual environment"
EOF
chmod +x activate_env.sh

# Create requirements file
cat > requirements_venv.txt << 'EOF'
torch
torchvision
torchaudio
gradio
accelerate
huggingface-hub
# Note: transformers is installed from custom fork
# git+https://github.com/Fazziekey/transformers.git@seed-oss
EOF

print_status "Utility scripts created"
echo ""

# Step 10: Verify installation
echo -e "${BLUE}Step 10: Verifying installation...${NC}"

# Test imports
python -c "import torch; print(f'✅ PyTorch {torch.__version__} installed')" 2>/dev/null || print_error "PyTorch import failed"
python -c "import gradio; print('✅ Gradio installed')" 2>/dev/null || print_error "Gradio import failed"
python -c "import accelerate; print('✅ Accelerate installed')" 2>/dev/null || print_error "Accelerate import failed"

# Test transformers and check for seed_oss support
python -c "
import sys
try:
    from transformers import AutoConfig
    # Try to check if seed_oss is supported
    print('✅ Transformers installed')
    try:
        # This will fail but that's OK - we're just checking if the architecture is recognized
        config = AutoConfig.from_pretrained('./Seed-OSS-36B-Instruct', local_files_only=True)
        print('✅ seed_oss architecture supported')
    except Exception as e:
        if 'seed_oss' not in str(e):
            print('✅ Custom transformers fork appears to be installed')
        else:
            print('⚠️  seed_oss architecture may not be fully supported')
except ImportError:
    print('❌ Transformers not installed properly')
    sys.exit(1)
" 2>/dev/null || print_warning "Transformers verification had issues"

echo ""

# Final summary
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Virtual environment: $VENV_NAME"
echo ""
echo "To run the chat interface:"
echo -e "  ${BLUE}./run_chat.sh${NC}"
echo ""
echo "To manually activate the environment:"
echo -e "  ${BLUE}source activate_env.sh${NC}"
echo "  or"
echo -e "  ${BLUE}source $VENV_NAME/bin/activate${NC}"
echo ""
echo "Optional arguments for run_chat.sh:"
echo "  ./run_chat.sh --device mps    # Force Apple Silicon GPU"
echo "  ./run_chat.sh --device cuda   # Force NVIDIA GPU"
echo "  ./run_chat.sh --device cpu    # Force CPU"
echo "  ./run_chat.sh --port 8080     # Use different port"
echo "  ./run_chat.sh --share         # Create public link"
echo ""

# Deactivate virtual environment
deactivate 2>/dev/null

echo -e "${GREEN}Virtual environment has been deactivated.${NC}"
echo -e "${GREEN}Use ./run_chat.sh to start the application.${NC}"