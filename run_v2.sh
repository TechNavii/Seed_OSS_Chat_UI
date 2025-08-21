#!/bin/bash

# Run the improved v2 chat interface

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Virtual environment name (matches setup_venv.sh)
VENV_NAME="seed_oss_env"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}🚀 Launching Seed-OSS Chat Interface v2${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Check if virtual environment exists
if [ -d "$VENV_NAME" ]; then
    echo -e "${GREEN}✅ Activating virtual environment: $VENV_NAME${NC}"
    source "$VENV_NAME/bin/activate"
    
    # Verify transformers fork is installed
    python -c "
import sys
try:
    from transformers import AutoConfig
    print('✅ Transformers installed')
    # Try to check for seed_oss support
    import transformers
    print(f'   Version: {transformers.__version__}')
except ImportError:
    print('❌ Transformers not found in virtual environment')
    sys.exit(1)
" || {
        echo -e "${YELLOW}⚠️  Issue with transformers installation${NC}"
        echo "   Reinstalling custom fork..."
        pip uninstall transformers -y
        pip install git+https://github.com/Fazziekey/transformers.git@seed-oss
    }
else
    echo -e "${YELLOW}⚠️  Virtual environment '$VENV_NAME' not found${NC}"
    echo "   Run ./setup_venv.sh first to set up the environment"
    echo ""
    exit 1
fi

# Check if model exists
if [ ! -d "./Seed-OSS-36B-Instruct" ]; then
    echo -e "${YELLOW}⚠️  Model directory not found${NC}"
    echo "   Run: python setup_model.py"
    echo ""
    exit 1
fi

# Parse arguments
DEVICE="auto"
PORT="7860"
SHARE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --share)
            SHARE="--share"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo -e "${BLUE}Configuration:${NC}"
echo "  Device: $DEVICE"
echo "  Port: $PORT"
echo ""

# Run the v2 interface
echo -e "${GREEN}Starting enhanced interface with improved generation...${NC}"
python chat_interface_v2.py --device "$DEVICE" --port "$PORT" $SHARE