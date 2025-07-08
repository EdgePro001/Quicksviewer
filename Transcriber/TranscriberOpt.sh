#!/bin/bash

# ============================================================================
# Enhanced Whisper Processing System Shell Script
# High-Performance A100-Optimized Video Transcription Pipeline
# GPU 0-3 Configuration: 4 GPUs, 6 Whisper models per GPU
# ============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Script metadata
SCRIPT_VERSION="2.0"
SCRIPT_NAME="Enhanced Whisper Processing System"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/TranscriberOpt.py"

# Default paths (can be overridden by environment variables)
DEFAULT_MODEL_PATH="/user/zhangyizhe/whisper-large-v3/my_whisper_model"
DEFAULT_DATA_PATH="/thunlp_train/datasets/external-data-youtube/video"
DEFAULT_OUTPUT_DIR="/user/majunxian/SFTPipeline/Pool"
DEFAULT_TARGET_FILE="/user/zhangyizhe/whisper_module/target.txt"

# Environment variables with defaults
MODEL_PATH="${WHISPER_MODEL_PATH:-$DEFAULT_MODEL_PATH}"
BASIC_DATA_PATH="${WHISPER_DATA_PATH:-$DEFAULT_DATA_PATH}"
OUTPUT_DIR="${WHISPER_OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
TARGET_FILE_PATH="${WHISPER_TARGET_FILE:-$DEFAULT_TARGET_FILE}"

# ============================================================================
# GPU CONFIGURATION - EXPLICITLY SET FOR GPU 0-3
# ============================================================================

# Hardware configuration - 明确指定使用GPU 0-3
NUM_GPUS="${WHISPER_NUM_GPUS:-4}"                    # 使用4个GPU
MODELS_PER_GPU="${WHISPER_MODELS_PER_GPU:-6}"        # 每个GPU运行6个模型
GPU_IDS="${WHISPER_GPU_IDS:-0,1,2,3}"               # 明确指定GPU ID
BATCH_SIZE="${WHISPER_BATCH_SIZE:-32}"               # 批处理大小

# 计算总模型数量
TOTAL_MODELS=$((NUM_GPUS * MODELS_PER_GPU))          # 总共24个模型实例

# Performance settings
ENABLE_MONITORING="${WHISPER_ENABLE_MONITORING:-true}"
CLEANUP_CACHE="${WHISPER_CLEANUP_CACHE:-true}"
GENERATE_SUMMARY="${WHISPER_GENERATE_SUMMARY:-true}"

# Logging configuration
LOG_DIR="${OUTPUT_DIR}/logs"
LOG_FILE="${LOG_DIR}/whisper_processing_$(date +%Y%m%d_%H%M%S).log"
ERROR_LOG="${LOG_DIR}/errors_$(date +%Y%m%d_%H%M%S).log"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE" "$ERROR_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${CYAN}[DEBUG]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
    fi
}

print_banner() {
    echo -e "${PURPLE}"
    cat << "EOF"
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Enhanced Whisper Processing System                        ║
║                     A100-Optimized Video Transcription                       ║
║                   GPU 0-3 Configuration (4 GPUs × 6 Models)                  ║
║                              Version 2.0                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

print_separator() {
    echo -e "${BLUE}$( printf '=%.0s' {1..80} )${NC}"
}

# ============================================================================
# SYSTEM CHECKS AND VALIDATION
# ============================================================================

check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check if running on Linux
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        log_error "This script is designed for Linux systems. Current OS: $OSTYPE"
        exit 1
    fi
    
    # Check available memory (建议至少128GB for 24 models)
    local available_memory_gb=$(free -g | awk 'NR==2{printf "%.1f", $7}')
    log_info "Available system memory: ${available_memory_gb}GB"
    
    if (( $(echo "$available_memory_gb < 100" | bc -l) )); then
        log_warn "Low system memory detected for running 24 Whisper models. Recommend at least 128GB."
    fi
    
    # Check disk space
    local available_space_gb=$(df "$OUTPUT_DIR" 2>/dev/null | awk 'NR==2 {printf "%.1f", $4/1024/1024}' || echo "0")
    log_info "Available disk space: ${available_space_gb}GB"
    
    if (( $(echo "$available_space_gb < 200" | bc -l) )); then
        log_warn "Low disk space detected. Consider freeing up space in output directory."
    fi
}

check_cuda_environment() {
    log_info "Checking CUDA environment for GPU 0-3..."
    
    # Check if nvidia-smi is available
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. CUDA environment may not be properly set up."
        exit 1
    fi
    
    # 验证指定的GPU是否可用
    IFS=',' read -ra REQUESTED_GPUS <<< "$GPU_IDS"
    log_info "Requested GPUs: ${REQUESTED_GPUS[*]}"
    
    # Check each requested GPU
    for gpu_id in "${REQUESTED_GPUS[@]}"; do
        if ! nvidia-smi -i "$gpu_id" &> /dev/null; then
            log_error "GPU $gpu_id is not available or not found"
            exit 1
        fi
        
        # Get GPU info
        local gpu_name=$(nvidia-smi -i "$gpu_id" --query-gpu=name --format=csv,noheader)
        local gpu_memory=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.total --format=csv,noheader)
        log_info "GPU $gpu_id: $gpu_name - Total Memory: $gpu_memory"
        
        # Check for A100 or high-end GPUs
        if [[ "$gpu_name" == *"A100"* ]]; then
            log_success "GPU $gpu_id: A100 detected - optimal for Whisper processing"
        elif [[ "$gpu_name" == *"V100"* ]] || [[ "$gpu_name" == *"RTX"* ]]; then
            log_info "GPU $gpu_id: Compatible GPU detected"
        else
            log_warn "GPU $gpu_id: Performance may vary with this GPU type"
        fi
    done
    
    # Verify we have the expected number of GPUs
    if [[ ${#REQUESTED_GPUS[@]} -ne $NUM_GPUS ]]; then
        log_error "Mismatch between NUM_GPUS ($NUM_GPUS) and GPU_IDS count (${#REQUESTED_GPUS[@]})"
        exit 1
    fi
    
    log_success "All requested GPUs (${REQUESTED_GPUS[*]}) are available and ready"
}

check_python_environment() {
    log_info "Checking Python environment..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 not found. Please install Python 3.8 or higher."
        exit 1
    fi
    
    local python_version=$(python3 --version | cut -d' ' -f2)
    log_info "Python version: $python_version"
    
    # Auto-activate virtual environment if not already in one
    local venv_path="${WHISPER_VENV_PATH:-/user/zhangyizhe/venv_whisper}"
    if [[ -z "${VIRTUAL_ENV:-}" && -f "$venv_path/bin/activate" ]]; then
        log_info "Activating virtual environment: $venv_path"
        source "$venv_path/bin/activate"
        log_success "Virtual environment activated successfully"
    elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
        log_info "Virtual environment: $VIRTUAL_ENV"
    else
        log_warn "No virtual environment detected and none found at $venv_path"
    fi
    
    # Check required Python packages with proper import names
    local required_packages=(
        "torch"
        "whisper_timestamped:whisper"
        "librosa" 
        "cv2:opencv-python"
        "tqdm"
        "psutil"
        "numpy"
        "json"
        "multiprocessing"
        "concurrent.futures"
        "threading"
        "queue"
        "time"
        "gc"
        "pathlib"
        "functools"
        "subprocess"
        "warnings"
        "os"
        "sys"
    )
    
    local missing_packages=()
    local missing_pip_packages=()
    
    for package_spec in "${required_packages[@]}"; do
        # Split package_spec into import_name:pip_name if colon exists
        if [[ "$package_spec" == *":"* ]]; then
            local import_name="${package_spec%%:*}"
            local pip_name="${package_spec##*:}"
        else
            local import_name="$package_spec"
            local pip_name="$package_spec"
        fi
        
        # Skip built-in modules
        if [[ "$import_name" =~ ^(json|multiprocessing|concurrent\.futures|threading|queue|time|gc|pathlib|functools|subprocess|warnings|os|sys)$ ]]; then
            continue
        fi
        
        if ! python3 -c "import $import_name" &> /dev/null; then
            missing_packages+=("$import_name")
            missing_pip_packages+=("$pip_name")
        fi
    done
    
    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        log_error "Missing required packages: ${missing_packages[*]}"
        log_info "Install missing packages with: pip install ${missing_pip_packages[*]}"
        
        # Offer to auto-install
        if [[ "${AUTO_INSTALL_DEPS:-false}" == "true" ]] || [[ "${INSTALL_MISSING:-}" == "true" ]]; then
            log_info "Auto-installing missing packages..."
            if pip install "${missing_pip_packages[@]}"; then
                log_success "Missing packages installed successfully"
            else
                log_error "Failed to install missing packages automatically"
                exit 1
            fi
        else
            log_info "Re-run with --install-deps to auto-install missing packages"
            exit 1
        fi
    fi
    
    # Special check for whisper_timestamped
    if ! python3 -c "import whisper_timestamped" &> /dev/null; then
        log_warn "whisper_timestamped not found. Installing..."
        if pip install git+https://github.com/linto-ai/whisper-timestamped; then
            log_success "whisper_timestamped installed successfully"
        else
            log_error "Failed to install whisper_timestamped"
            exit 1
        fi
    fi
    
    log_success "All required Python packages are available"
}

check_dependencies() {
    log_info "Checking system dependencies..."
    
    # Check FFmpeg
    if ! command -v ffmpeg &> /dev/null; then
        log_error "FFmpeg not found. Please install FFmpeg."
        log_info "Install with: sudo apt-get install ffmpeg"
        exit 1
    fi
    
    local ffmpeg_version=$(ffmpeg -version 2>/dev/null | head -n1 | cut -d' ' -f3)
    log_info "FFmpeg version: $ffmpeg_version"
    
    # Check other utilities
    local required_commands=("bc" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_warn "$cmd not found. Some features may not work properly."
        fi
    done
}

validate_paths() {
    log_info "Validating paths and files..."
    
    # Check if Python script exists
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        log_error "Python script not found: $PYTHON_SCRIPT"
        exit 1
    fi
    
    # Check target file
    if [[ ! -f "$TARGET_FILE_PATH" ]]; then
        log_error "Target file not found: $TARGET_FILE_PATH"
        exit 1
    fi
    
    # Count videos in target file
    local video_count=$(grep -c "^/" "$TARGET_FILE_PATH" 2>/dev/null || echo "0")
    log_info "Videos to process: $video_count"
    
    if [[ $video_count -eq 0 ]]; then
        log_error "No valid video paths found in target file"
        exit 1
    fi
    
    # Check model path (optional)
    if [[ -n "$MODEL_PATH" && ! -d "$MODEL_PATH" ]]; then
        log_warn "Custom model path not found: $MODEL_PATH (will use default model)"
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
    log_info "Output directory: $OUTPUT_DIR"
    log_info "Log directory: $LOG_DIR"
}

# ============================================================================
# MONITORING AND PERFORMANCE FUNCTIONS
# ============================================================================

start_system_monitoring() {
    if [[ "$ENABLE_MONITORING" != "true" ]]; then
        return
    fi
    
    log_info "Starting system monitoring for GPU ${GPU_IDS}..."
    
    local monitor_log="${LOG_DIR}/system_monitor_$(date +%Y%m%d_%H%M%S).log"
    
    # Start GPU monitoring in background - 只监控指定的GPU
    {
        echo "timestamp,gpu_id,gpu_util,mem_util,mem_used,mem_total,temp,power" > "${monitor_log}"
        while true; do
            timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            # 只监控指定的GPU
            nvidia-smi -i "$GPU_IDS" --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | \
            while IFS=, read -r gpu_id gpu_util mem_util mem_used mem_total temp power; do
                echo "$timestamp,$gpu_id,$gpu_util,$mem_util,$mem_used,$mem_total,$temp,$power" >> "$monitor_log"
            done
            sleep 10
        done
    } &
    
    MONITOR_PID=$!
    log_debug "System monitoring started with PID: $MONITOR_PID"
}

stop_system_monitoring() {
    if [[ -n "${MONITOR_PID:-}" ]]; then
        log_info "Stopping system monitoring..."
        kill $MONITOR_PID 2>/dev/null || true
        wait $MONITOR_PID 2>/dev/null || true
        log_debug "System monitoring stopped"
    fi
}

generate_performance_report() {
    log_info "Generating performance report..."
    
    local report_file="${LOG_DIR}/performance_report_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" << EOF
================================================================
WHISPER PROCESSING PERFORMANCE REPORT - GPU 0-3 Configuration
================================================================
Generated: $(date)
Script Version: $SCRIPT_VERSION

Configuration:
- GPUs Used: $GPU_IDS
- Number of GPUs: $NUM_GPUS
- Models per GPU: $MODELS_PER_GPU
- Total Models: $TOTAL_MODELS
- Batch Size: $BATCH_SIZE
- Output Directory: $OUTPUT_DIR

System Information:
- OS: $(uname -s) $(uname -r)
- CPU: $(lscpu | grep "Model name" | cut -d':' -f2 | xargs)
- Memory: $(free -h | grep "^Mem:" | awk '{print $2}')
- Python: $(python3 --version)

GPU Information (Selected GPUs):
$(nvidia-smi -i "$GPU_IDS" --query-gpu=index,name,memory.total --format=csv,noheader)

Processing Results:
$(if [[ -f "${OUTPUT_DIR}/processing_summary.txt" ]]; then cat "${OUTPUT_DIR}/processing_summary.txt"; else echo "Summary not available"; fi)

================================================================
EOF
    
    log_success "Performance report saved to: $report_file"
}

# ============================================================================
# PROCESS MANAGEMENT
# ============================================================================

cleanup_processes() {
    log_info "Cleaning up processes..."
    
    # Stop monitoring
    stop_system_monitoring
    
    # Kill any remaining Python processes from this script
    pkill -f "TranscriberOpt.py" 2>/dev/null || true
    
    # Clean GPU memory on specified GPUs only
    if command -v nvidia-smi &> /dev/null; then
        log_info "Clearing GPU memory on GPUs ${GPU_IDS}..."
        IFS=',' read -ra CLEANUP_GPUS <<< "$GPU_IDS"
        for gpu_id in "${CLEANUP_GPUS[@]}"; do
            nvidia-smi -i "$gpu_id" --gpu-reset 2>/dev/null || log_debug "GPU $gpu_id reset not supported (normal for A100 multi-instance)"
        done
    fi
}

handle_interrupt() {
    log_warn "Received interrupt signal. Cleaning up..."
    cleanup_processes
    log_info "Cleanup completed. Exiting."
    exit 130
}

# Set up signal handlers
trap handle_interrupt SIGINT SIGTERM

# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

prepare_environment() {
    log_info "Preparing processing environment for GPU ${GPU_IDS}..."
    
    # Set environment variables for optimal performance
    export TOKENIZERS_PARALLELISM=false
    export CUDA_LAUNCH_BLOCKING=0
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True"
    export OMP_NUM_THREADS=4
    
    # 明确设置可见的GPU为指定的GPU
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    
    # Set Python environment variables
    export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
    export PYTHONUNBUFFERED=1
    
    # 将配置参数传递给Python脚本
    export WHISPER_NUM_GPUS="$NUM_GPUS"
    export WHISPER_MODELS_PER_GPU="$MODELS_PER_GPU"
    export WHISPER_TOTAL_MODELS="$TOTAL_MODELS"
    export WHISPER_BATCH_SIZE="$BATCH_SIZE"
    
    # Log environment setup
    log_debug "Environment variables set:"
    log_debug "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    log_debug "  WHISPER_NUM_GPUS=$WHISPER_NUM_GPUS"
    log_debug "  WHISPER_MODELS_PER_GPU=$WHISPER_MODELS_PER_GPU"
    log_debug "  WHISPER_TOTAL_MODELS=$WHISPER_TOTAL_MODELS"
    log_debug "  TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM"
    log_debug "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
    
    log_success "Environment prepared for $TOTAL_MODELS Whisper models across $NUM_GPUS GPUs"
}

run_whisper_processing() {
    log_info "Starting Whisper processing with $TOTAL_MODELS models on GPU ${GPU_IDS}..."
    
    local start_time=$(date +%s)
    local python_args=(
        "$PYTHON_SCRIPT"
    )
    
    # Start system monitoring
    start_system_monitoring
    
    # Run the Python script with proper logging
    log_info "Executing: python3 ${python_args[*]}"
    log_info "Configuration: $NUM_GPUS GPUs × $MODELS_PER_GPU models = $TOTAL_MODELS total models"
    
    if python3 "${python_args[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_success "Whisper processing completed successfully in ${duration} seconds"
        return 0
    else
        local exit_code=$?
        log_error "Whisper processing failed with exit code: $exit_code"
        return $exit_code
    fi
}

post_process_cleanup() {
    if [[ "$CLEANUP_CACHE" == "true" ]]; then
        log_info "Performing post-processing cleanup..."
        
        # Clean up temporary audio files
        find "$OUTPUT_DIR" -name "*.wav" -path "*/audio_cache_parallel/*" -delete 2>/dev/null || true
        
        # Clean up empty directories
        find "$OUTPUT_DIR" -type d -empty -delete 2>/dev/null || true
        
        log_success "Post-processing cleanup completed"
    fi
}

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

show_help() {
    cat << EOF
$SCRIPT_NAME v$SCRIPT_VERSION - GPU 0-3 Configuration

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -v, --verbose           Enable verbose output
    -d, --debug             Enable debug mode
    --dry-run               Validate setup without running processing
    --install-deps          Auto-install missing Python dependencies
    --no-monitoring         Disable system monitoring
    --no-cleanup            Skip post-processing cleanup
    --no-summary            Skip summary generation
    
    --model-path PATH       Override model path
    --data-path PATH        Override data path
    --output-dir PATH       Override output directory
    --target-file PATH      Override target file path
    --venv-path PATH        Override virtual environment path
    
    --gpu-ids "0,1,2,3"    GPU IDs to use (default: $GPU_IDS)
    --num-gpus N           Number of GPUs to use (default: $NUM_GPUS)
    --models-per-gpu N     Models per GPU (default: $MODELS_PER_GPU)
    --batch-size N         Batch size (default: $BATCH_SIZE)

CURRENT CONFIGURATION:
    - GPUs: $GPU_IDS (Total: $NUM_GPUS GPUs)
    - Models per GPU: $MODELS_PER_GPU
    - Total Models: $TOTAL_MODELS
    - Batch Size: $BATCH_SIZE

ENVIRONMENT VARIABLES:
    WHISPER_GPU_IDS         GPU IDs to use (e.g., "0,1,2,3")
    WHISPER_MODEL_PATH      Path to Whisper model
    WHISPER_DATA_PATH       Path to video data
    WHISPER_OUTPUT_DIR      Output directory
    WHISPER_TARGET_FILE     Target file with video paths
    WHISPER_VENV_PATH       Virtual environment path (default: /user/zhangyizhe/venv_whisper)
    WHISPER_NUM_GPUS        Number of GPUs
    WHISPER_MODELS_PER_GPU  Models per GPU
    WHISPER_BATCH_SIZE      Batch size
    WHISPER_ENABLE_MONITORING    Enable monitoring (true/false)
    WHISPER_CLEANUP_CACHE   Enable cache cleanup (true/false)
    WHISPER_GENERATE_SUMMARY     Generate summary (true/false)
    AUTO_INSTALL_DEPS       Auto-install missing dependencies (true/false)

EXAMPLES:
    # Basic usage with default GPU 0-3 configuration
    $0
    
    # Use different GPUs
    $0 --gpu-ids "0,1,2,3"
    
    # Custom configuration
    $0 --gpu-ids "0,1,2,3" --models-per-gpu 8 --batch-size 16
    
    # Auto-install dependencies and run
    $0 --install-deps
    
    # Dry run to check setup
    $0 --dry-run
    
    # Debug mode
    $0 --debug --verbose

PERFORMANCE NOTES:
    - Each GPU will run $MODELS_PER_GPU Whisper models simultaneously
    - Total of $TOTAL_MODELS models will be processing in parallel
    - Recommended system memory: 128GB+ for optimal performance
    - A100 GPUs are recommended for best performance

EOF
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                set -x
                ;;
            -d|--debug)
                export DEBUG=true
                ;;
            --dry-run)
                DRY_RUN=true
                ;;
            --install-deps)
                INSTALL_MISSING=true
                ;;
            --no-monitoring)
                ENABLE_MONITORING=false
                ;;
            --no-cleanup)
                CLEANUP_CACHE=false
                ;;
            --no-summary)
                GENERATE_SUMMARY=false
                ;;
            --model-path)
                MODEL_PATH="$2"
                shift
                ;;
            --data-path)
                BASIC_DATA_PATH="$2"
                shift
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift
                ;;
            --target-file)
                TARGET_FILE_PATH="$2"
                shift
                ;;
            --venv-path)
                WHISPER_VENV_PATH="$2"
                shift
                ;;
            --gpu-ids)
                GPU_IDS="$2"
                # 重新计算GPU数量
                IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
                NUM_GPUS=${#GPU_ARRAY[@]}
                TOTAL_MODELS=$((NUM_GPUS * MODELS_PER_GPU))
                shift
                ;;
            --num-gpus)
                NUM_GPUS="$2"
                TOTAL_MODELS=$((NUM_GPUS * MODELS_PER_GPU))
                shift
                ;;
            --models-per-gpu)
                MODELS_PER_GPU="$2"
                TOTAL_MODELS=$((NUM_GPUS * MODELS_PER_GPU))
                shift
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
        shift
    done
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    # Initialize logging directory
    mkdir -p "$LOG_DIR"
    
    # Print banner
    print_banner
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Log script start
    log_info "Starting $SCRIPT_NAME v$SCRIPT_VERSION"
    log_info "Script directory: $SCRIPT_DIR"
    log_info "Log file: $LOG_FILE"
    log_info "Error log: $ERROR_LOG"
    
    print_separator
    
    # System validation
    check_system_requirements
    check_cuda_environment
    check_python_environment
    check_dependencies
    validate_paths
    
    print_separator
    
    # Configuration summary
    log_info "GPU Configuration Summary:"
    log_info "  Target GPUs: $GPU_IDS"
    log_info "  Number of GPUs: $NUM_GPUS"
    log_info "  Models per GPU: $MODELS_PER_GPU"
    log_info "  Total Models: $TOTAL_MODELS"
    log_info "  Batch Size: $BATCH_SIZE"
    log_info ""
    log_info "Path Configuration:"
    log_info "  Model Path: $MODEL_PATH"
    log_info "  Data Path: $BASIC_DATA_PATH"
    log_info "  Output Directory: $OUTPUT_DIR"
    log_info "  Target File: $TARGET_FILE_PATH"
    log_info "  Virtual Environment: ${WHISPER_VENV_PATH:-/user/zhangyizhe/venv_whisper}"
    log_info ""
    log_info "Processing Options:"
    log_info "  Monitoring: $ENABLE_MONITORING"
    log_info "  Cleanup: $CLEANUP_CACHE"
    log_info "  Summary: $GENERATE_SUMMARY"
    log_info "  Auto-install Dependencies: ${INSTALL_MISSING:-false}"
    
    # Dry run mode
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        log_info "Dry run completed successfully. All checks passed."
        log_info "Ready to process with $TOTAL_MODELS Whisper models on GPU ${GPU_IDS}"
        print_separator
        exit 0
    fi
    
    print_separator
    
    # Main processing
    prepare_environment
    
    local overall_start_time=$(date +%s)
    
    if run_whisper_processing; then
        local overall_end_time=$(date +%s)
        local overall_duration=$((overall_end_time - overall_start_time))
        
        log_success "Overall processing completed in ${overall_duration} seconds"
        
        # Post-processing
        post_process_cleanup
        
        # Generate reports
        if [[ "$GENERATE_SUMMARY" == "true" ]]; then
            generate_performance_report
        fi
        
        print_separator
        log_success "All operations completed successfully!"
        log_info "Configuration: $NUM_GPUS GPUs (${GPU_IDS}) × $MODELS_PER_GPU models = $TOTAL_MODELS total"
        log_info "Results available in: $OUTPUT_DIR"
        log_info "Logs available in: $LOG_DIR"
        
    else
        log_error "Processing failed. Check logs for details."
        print_separator
        exit 1
    fi
}

# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

# Ensure cleanup on exit
trap cleanup_processes EXIT

# Run main function with all arguments
main "$@"