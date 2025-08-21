# 🌱 Seed-OSS-36B Chat Interface

A high-performance chat interface for ByteDance's **Seed-OSS-36B** model with real-time telemetry, thinking visualization, and platform-optimized inference.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Features

- 🧠 **Controllable Reasoning**: Visualize the model's thinking process with adjustable thinking budget
- 📊 **Real-time Telemetry**: Live metrics including tokens/sec, time-to-first-token, and token counts
- 🚀 **Platform Optimized**: Automatic detection and optimization for CUDA, Apple Silicon (MPS), and CPU
- 💬 **Streaming Interface**: Real-time response generation with Gradio UI
- 🎯 **Smart Token Management**: Separate tracking of thinking vs response tokens

## 📸 UI

The interface shows both the model's thinking process and final response with detailed performance metrics:

```
🧠 Thinking Process:
[Model's reasoning shown here]

💬 Response:
[Model's answer shown here]

📊 Generation Metrics:
- ⏱️ Total time: 5.23s
- 🚀 Time to first token: 0.234s
- 📝 Total tokens: 523 (thinking: 145, response: 378)
- ⚡ Speed: 100.2 tokens/sec (avg)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 32GB+ RAM (model requires ~35GB) -> Depends on the quantization
- ~75GB disk space for model files
- (Optional) NVIDIA GPU with 24GB+ VRAM or Apple Silicon Mac

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/TechNavii/seed-oss-chat.git
cd seed-oss-chat
```

2. **Run the setup script:**
```bash
./setup_venv.sh
```

This will:
- Create a virtual environment
- Install all dependencies
- Install the custom transformers fork (required for Seed-OSS)
- Prompt to download the model (~72GB)

3. **Launch the interface:**
```bash
./run_v2.sh
```

The interface will open at `http://localhost:7860`

## 🛠️ Advanced Usage

### Command Line Options

```bash
# Specify device
./run_v2.sh --device mps    # Apple Silicon
./run_v2.sh --device cuda   # NVIDIA GPU
./run_v2.sh --device cpu    # CPU only

# Custom port
./run_v2.sh --port 8080

# Create public share link
./run_v2.sh --share
```


## 🏗️ Architecture

The application uses a modular architecture:

- **`DeviceManager`**: Handles platform detection and optimization
- **`ModelManager`**: Manages model loading with fallback support
- **`ResponseParser`**: Parses thinking and response sections
- **`ImprovedChatStreamer`**: Enhanced streaming with telemetry
- **`ChatApplication`**: Main application orchestration

## 🔧 Troubleshooting

### Common Issues

**1. "seed_oss architecture not recognized"**
- The custom transformers fork is not installed
- Run: `pip install git+https://github.com/Fazziekey/transformers.git@seed-oss`

**2. Out of Memory**
- Ensure you have 35GB+ free RAM
- Try CPU inference if GPU runs out of VRAM
- Use `--device cpu` flag

**3. Slow Generation**
- This is a 36B parameter model, generation will be slower than smaller models
- Use GPU acceleration when possible
- Adjust thinking_budget to reduce thinking time

## 📚 About Seed-OSS

Seed-OSS-36B is a large language model developed by ByteDance with unique features:

- **Controllable Reasoning**: Adjustable thinking process via `thinking_budget`
- **512K Context Window**: Handle very long conversations
- **Custom Architecture**: Requires special transformers fork for `seed_oss` support

Learn more: [ByteDance-Seed/Seed-OSS-36B-Instruct](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- ByteDance for the Seed-OSS model
- Fazziekey for the custom transformers fork
- The Hugging Face team for the transformers library

## ⚠️ Important Notes

1. **Model Size**: The model files are ~72GB and are NOT included in this repository
2. **Custom Fork Required**: Standard transformers won't work; you need the custom fork
3. **Resource Intensive**: This is a large model requiring significant computational resources

## 📞 Support

If you encounter issues:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Ensure you're using the custom transformers fork
3. Open an issue with your system details and error messages
