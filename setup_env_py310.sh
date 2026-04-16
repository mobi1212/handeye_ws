#!/usr/bin/env bash
# ============================================================
# setup_env_py310.sh
# 建立 Python 3.10 統一環境：整合 anygrasp + message.txt 套件
#
# 使用方式：
#   bash setup_env_py310.sh
#
# 完成後啟用：
#   conda activate grasp-py310
# ============================================================

set -e
ENV_NAME="grasp-py310"

echo "======================================================"
echo "  建立 conda 環境：$ENV_NAME (Python 3.10.20)"
echo "======================================================"

# ── 步驟 1：建立環境 ──────────────────────────────────────
conda create -n $ENV_NAME python=3.10.20 -y

# ── 步驟 2：安裝 PyTorch（CUDA 12.4，與 message.txt 一致）──
echo ""
echo "[2/8] 安裝 PyTorch 2.4.1 + CUDA 12.4..."
conda run -n $ENV_NAME pip install \
    torch==2.4.1+cu124 \
    torchvision==0.19.1+cu124 \
    torchaudio==2.4.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# ── 步驟 3：Gemini + VLM 新功能套件（來自 message.txt）──
echo ""
echo "[3/8] 安裝 Gemini / VLM 套件..."
conda run -n $ENV_NAME pip install \
    "google-generativeai==0.8.6" \
    "python-dotenv==1.2.2" \
    "sam-2==1.0" \
    "lang-sam==0.2.1" \
    "supervision==0.27.0.post2"

# ── 步驟 4：HuggingFace 生態（OWL-v2 + SAM v1）──────────
echo ""
echo "[4/8] 安裝 HuggingFace / transformers..."
conda run -n $ENV_NAME pip install \
    "transformers==4.46.3" \
    "tokenizers==0.20.3" \
    "safetensors==0.5.3" \
    "huggingface-hub==0.36.2" \
    "accelerate==1.0.1" \
    "datasets"

# ── 步驟 5：電腦視覺與數值計算 ───────────────────────────
# numpy: 使用 1.26.4（不用 2.x，避免 MinkowskiEngine / open3d 相容問題）
echo ""
echo "[5/8] 安裝 CV / 數值計算套件..."
conda run -n $ENV_NAME pip install \
    "numpy==1.26.4" \
    "opencv-python==4.13.0.92" \
    "pillow==12.1.1" \
    "scipy==1.15.3" \
    "scikit-learn==1.3.2" \
    "scikit-image==0.21.0" \
    "matplotlib==3.10.8" \
    "pandas==2.3.3" \
    "imageio==2.35.1" \
    "tifffile==2023.7.10" \
    "PyWavelets" \
    "trimesh" \
    "transforms3d"

# ── 步驟 6：機器人控制與通訊 ─────────────────────────────
echo ""
echo "[6/8] 安裝機器人通訊套件..."
conda run -n $ENV_NAME pip install \
    "pyzmq==27.1.0" \
    "pyrealsense2==2.55.1.6486" \
    "pyserial==3.5"

# ── 步驟 7：其他工具套件 ─────────────────────────────────
echo ""
echo "[7/8] 安裝其他工具套件..."
conda run -n $ENV_NAME pip install \
    "open3d==0.19.0" \
    "ultralytics==8.3.228" \
    "tqdm==4.67.3" \
    "pyyaml==6.0.3" \
    "requests==2.32.5" \
    "click==8.1.8" \
    "rich==14.3.3" \
    "plotly==6.5.2" \
    "dash==4.0.0" \
    "Flask==3.0.3" \
    "ipython==8.12.3" \
    "sympy==1.14.0" \
    "networkx==3.4.2" \
    "filelock==3.25.2" \
    "packaging==26.0" \
    "psutil==7.2.2" \
    "h5py==3.11.0" \
    "dill==0.4.0" \
    "addict==2.4.0" \
    "pyquaternion==0.9.9" \
    "ruamel.yaml" \
    "ConfigArgParse" \
    "retrying"

# ── 步驟 8：MinkowskiEngine（需從原始碼編譯）────────────
echo ""
echo "[8/8] MinkowskiEngine..."
echo "  ⚠️  MinkowskiEngine 需從原始碼編譯，跳過自動安裝。"
echo "  完成後請手動執行（需要 CUDA toolkit）："
echo ""
echo "    conda activate $ENV_NAME"
echo "    pip install ninja"
echo "    pip install -U git+https://github.com/NVIDIA/MinkowskiEngine \\"
echo "        --install-option='--blas_include_dirs=\${CONDA_PREFIX}/include' \\"
echo "        --install-option='--blas=openblas'"
echo ""
echo "  若不需要 AnyGrasp 點雲處理功能可以跳過此步驟。"

echo ""
echo "======================================================"
echo "  環境建立完成！"
echo ""
echo "  ✅ Python 3.10.20"
echo "  ✅ torch 2.4.1+cu124"
echo "  ✅ google-generativeai 0.8.6"
echo "  ✅ transformers 4.46.3"
echo "  ✅ numpy 1.26.4  ← 使用 1.x 確保 open3d/MinkowskiEngine 相容"
echo "  ✅ sam-2 + lang-sam + supervision（message.txt 新功能）"
echo "  ⚠️  MinkowskiEngine：需手動編譯（見上方說明）"
echo "  ⚠️  ROS 套件（rospy/moveit）：由系統 /opt/ros/noetic 提供"
echo "       腳本中已有 sys.path 處理，無需在此環境安裝"
echo ""
echo "  啟用環境："
echo "    conda activate $ENV_NAME"
echo "======================================================"
