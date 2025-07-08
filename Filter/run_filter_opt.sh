#!/bin/bash

# Qwen2.5-VL-72B on GPU 4-7 Only with Continuous Monitoring
# Specialized for processing Whisper results from Pool directory with real-time monitoring

echo "Qwen2.5-VL-72B Continuous Monitor on GPU 4-7"
echo "=============================================="
echo "Purpose: Continuously process Whisper results from Pool directory"

# Force restrict to GPU 4-7 only
export CUDA_VISIBLE_DEVICES=4,5,6,7

LOCAL_MODEL_PATH="/user/majunxian/.cache/modelscope/qwen/Qwen2.5-VL-72B-Instruct"

# Validate model path
if [ ! -d "$LOCAL_MODEL_PATH" ]; then
    echo "ERROR: Model directory does not exist: $LOCAL_MODEL_PATH"
    exit 1
fi

if [ ! -f "$LOCAL_MODEL_PATH/config.json" ]; then
    echo "ERROR: Missing config.json"
    exit 1
fi

echo "Model validation passed"
echo "Model path: $LOCAL_MODEL_PATH"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES (Physical GPU 4-7)"

# Check GPU 4-7 status
echo ""
echo "GPU 4-7 Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader,nounits | sed -n '5,8p'

# Clean environment - only clean port 8001 related processes
echo ""
echo "Cleaning environment for GPU 4-7 task..."
# Clean possible existing 8001 port services
lsof -ti:8001 2>/dev/null | xargs kill -9 2>/dev/null || true
# Clean previous Qwen filtering processes
pkill -f "QwenFilterOpt" 2>/dev/null || true
sleep 2

# GPU 4-7 memory cleanup
echo "Clearing GPU 4-7 memory..."
python3 -c "
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
    print('GPU 4-7 memory cleared')
else:
    print('CUDA not available')
" 2>/dev/null || echo "GPU cleanup completed"

# Configuration parameters - optimized for 4 GPUs
TENSOR_PARALLEL_SIZE=4  # Use 4 GPUs instead of 8
MAX_MODEL_LEN=6144
MAX_NUM_SEQS=64         # Reduce sequence count since GPU count is halved
GPU_UTIL=0.85           # Slightly increase GPU utilization
PORT=8001               # Use different port to avoid conflicts

echo ""
echo "GPU 4-7 Startup Configuration:"
echo "  - Physical GPUs: 4,5,6,7"
echo "  - Logical GPUs: 0,1,2,3 (within CUDA_VISIBLE_DEVICES)"
echo "  - Tensor parallel: ${TENSOR_PARALLEL_SIZE}"
echo "  - Max length: ${MAX_MODEL_LEN}"
echo "  - Max sequences: ${MAX_NUM_SEQS}"
echo "  - GPU utilization: ${GPU_UTIL}"
echo "  - Service port: ${PORT}"

# Start vLLM server
echo ""
echo "Starting vLLM server on GPU 4-7..."

vllm serve "$LOCAL_MODEL_PATH" \
    --host 0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --gpu-memory-utilization $GPU_UTIL \
    --trust-remote-code \
    --disable-log-requests \
    --max-num-batched-tokens 3072 \
    --enable-chunked-prefill \
    > vllm_gpu4_7.log 2>&1 &

VLLM_PID=$!
echo "vLLM process ID: $VLLM_PID"
echo "Log file: vllm_gpu4_7.log"

# Wait for startup
echo ""
echo "Waiting for startup on GPU 4-7 (estimated 5-8 minutes)..."

STARTUP_SUCCESS=false
for ((i=1; i<=80; i++)); do  # 13 minute timeout
    sleep 10
    
    # Check process
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: Process exited, checking logs:"
        tail -20 vllm_gpu4_7.log
        exit 1
    fi
    
    # Test connection
    if curl -s --connect-timeout 5 http://localhost:$PORT/v1/models >/dev/null 2>&1; then
        echo "SUCCESS: GPU 4-7 startup completed! (time: $((i*10)) seconds)"
        STARTUP_SUCCESS=true
        break
    fi
    
    # Show progress
    if [ $((i % 6)) -eq 0 ]; then
        echo "[$((i*10))s] Starting on GPU 4-7..."
        
        # Show GPU 4-7 usage
        gpu4_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4)
        gpu7_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7)
        echo "GPU4 memory: ${gpu4_used}MB, GPU7 memory: ${gpu7_used}MB"
        
        # Show latest log
        latest_log=$(tail -1 vllm_gpu4_7.log 2>/dev/null | grep -v "^$")
        if [ -n "$latest_log" ]; then
            echo "Status: $latest_log"
        fi
        echo "---"
    fi
done

if [ "$STARTUP_SUCCESS" = false ]; then
    echo "ERROR: Startup timeout on GPU 4-7"
    echo "Check full logs:"
    tail -30 vllm_gpu4_7.log
    exit 1
fi

# Verify service
echo ""
echo "Service verification on port $PORT..."
curl -s http://localhost:$PORT/v1/models | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'data' in data:
        print('Model service on GPU 4-7 normal')
        for model in data['data']:
            print(f'Model: {model.get(\"id\", \"Unknown\")}')
    else:
        print('Response format abnormal')
except:
    print('Response parsing failed')
" 2>/dev/null

# Show GPU 4-7 usage
echo ""
echo "Post-startup GPU 4-7 status:"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits | sed -n '5,8p'

# Inference test
echo ""
echo "Inference test on GPU 4-7..."
curl -s -X POST http://localhost:$PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'$LOCAL_MODEL_PATH'",
        "prompt": "Hello",
        "max_tokens": 5,
        "temperature": 0
    }' | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'choices' in data:
        print('GPU 4-7 inference normal')
    else:
        print('GPU 4-7 inference abnormal')
except:
    print('GPU 4-7 inference test failed')
" 2>/dev/null

# Prepare filtering task
echo ""
echo "Preparing Whisper results continuous monitoring..."

# Whisper results directory - Pool folder
INPUT_DIR="/user/majunxian/SFTPipeline/Pool"
OUTPUT_DIR="/user/majunxian/SFTPipeline/FiltedOutput"

echo "Whisper results continuous monitoring configuration:"
echo "  - Input (Whisper Pool): $INPUT_DIR"
echo "  - Output: $OUTPUT_DIR"
echo "  - vLLM API: http://localhost:$PORT/v1"
echo "  - Using GPU: 4-7"
echo "  - Threshold: 0.80"

if [ ! -d "$INPUT_DIR" ]; then
    echo "WARNING: Input directory does not exist: $INPUT_DIR"
    echo "Please ensure Whisper processing pipeline is running"
    read -p "Enter correct Whisper Pool path: " INPUT_DIR
fi

# Check if there are JSON files
json_count=$(find "$INPUT_DIR" -name "*.json" 2>/dev/null | wc -l)
echo "Currently found $json_count JSON files in Whisper Pool"

# Check QwenFilterOpt.py
if [ ! -f "QwenFilterOpt.py" ]; then
    echo "ERROR: Cannot find QwenFilterOpt.py"
    echo "Current directory: $(pwd)"
    echo "Please ensure QwenFilterOpt.py is in the current directory"
    exit 1
fi

echo ""
echo "=== Continuous Monitoring Options ==="
echo "1. Start continuous monitoring (recommended for real-time processing)"
echo "2. Process current files only (single batch)"
echo "3. Custom monitoring settings"
echo "4. Check current status and exit"

read -p "Select option (1-4): " monitoring_option

case $monitoring_option in
    1)
        # Default continuous monitoring
        CONTINUOUS="--continuous"
        CHECK_INTERVAL="30"
        MAX_IDLE_TIME="1800"
        echo "Using default continuous monitoring settings:"
        echo "  - Check interval: ${CHECK_INTERVAL}s"
        echo "  - Max idle time: ${MAX_IDLE_TIME}s (30 minutes)"
        ;;
    2)
        # Single batch processing
        CONTINUOUS=""
        CHECK_INTERVAL=""
        MAX_IDLE_TIME=""
        echo "Single batch processing mode selected"
        ;;
    3)
        # Custom settings
        CONTINUOUS="--continuous"
        read -p "Check interval in seconds (default 30): " CHECK_INTERVAL
        CHECK_INTERVAL=${CHECK_INTERVAL:-30}
        read -p "Max idle time in seconds (default 1800): " MAX_IDLE_TIME
        MAX_IDLE_TIME=${MAX_IDLE_TIME:-1800}
        echo "Custom continuous monitoring settings:"
        echo "  - Check interval: ${CHECK_INTERVAL}s"
        echo "  - Max idle time: ${MAX_IDLE_TIME}s"
        ;;
    4)
        echo "Current system status:"
        echo "  - vLLM server PID: $VLLM_PID"
        echo "  - API endpoint: http://localhost:$PORT/v1"
        echo "  - JSON files in Pool: $json_count"
        echo "  - Output directory: $OUTPUT_DIR"
        echo ""
        echo "To manually start processing:"
        echo "python3 QwenFilterOpt.py --input_dir \"$INPUT_DIR\" --output_dir \"$OUTPUT_DIR\" --continuous"
        exit 0
        ;;
    *)
        echo "Invalid option, using default continuous monitoring"
        CONTINUOUS="--continuous"
        CHECK_INTERVAL="30"
        MAX_IDLE_TIME="1800"
        ;;
esac

echo ""
echo "Ready to start Whisper results processing using GPU 4-7"
if [ -n "$CONTINUOUS" ]; then
    echo "Mode: Continuous monitoring"
    echo "The system will:"
    echo "  - Check for new JSON files every ${CHECK_INTERVAL}s"
    echo "  - Process new files automatically"
    echo "  - Stop after ${MAX_IDLE_TIME}s of no new files"
    echo "  - Can be stopped with Ctrl+C"
else
    echo "Mode: Single batch processing"
fi

read -p "Start processing? (press enter to continue, 'n' to exit): " start_processing

if [[ $start_processing =~ ^[Nn]$ ]]; then
    echo "Processing cancelled. vLLM server still running on GPU 4-7"
    echo "Server PID: $VLLM_PID"
    echo "To stop: kill $VLLM_PID"
    exit 0
fi

echo ""
echo "Starting Whisper results processing on GPU 4-7..."
echo "Start time: $(date)"

# Build command
CMD="python3 QwenFilterOpt.py \
    --input_dir \"$INPUT_DIR\" \
    --output_dir \"$OUTPUT_DIR\" \
    --threshold 0.80 \
    --api_base \"http://localhost:$PORT/v1\" \
    --model_name \"$LOCAL_MODEL_PATH\" \
    --max_workers 8 \
    --max_concurrent_requests 16"

if [ -n "$CONTINUOUS" ]; then
    CMD="$CMD $CONTINUOUS --check_interval $CHECK_INTERVAL --max_idle_time $MAX_IDLE_TIME"
fi

# Execute processing
eval $CMD 2>&1 | tee processing_gpu4_7.log

PROCESS_EXIT_CODE=$?

echo ""
echo "Completion time: $(date)"

if [ $PROCESS_EXIT_CODE -eq 0 ]; then
    echo "Processing completed successfully on GPU 4-7!"
    
    if [ -d "$OUTPUT_DIR" ]; then
        output_count=$(find "$OUTPUT_DIR" -name "filtered_*.json" 2>/dev/null | wc -l)
        echo "Filtered output files: $output_count JSON files"
        
        # Show processing statistics
        echo ""
        echo "Processing summary:"
        if [ -f "processing_gpu4_7.log" ]; then
            if [ -n "$CONTINUOUS" ]; then
                # Continuous monitoring stats
                total_batches=$(grep "Total batches processed:" processing_gpu4_7.log | tail -1 | cut -d: -f2 | xargs)
                total_files=$(grep "Total files processed:" processing_gpu4_7.log | tail -1 | cut -d: -f2 | xargs)
                retention_rate=$(grep "Overall retention rate:" processing_gpu4_7.log | tail -1 | cut -d: -f2 | xargs)
                
                if [ -n "$total_batches" ]; then
                    echo "  - Total monitoring batches: $total_batches"
                    echo "  - Total files processed: $total_files"
                    echo "  - Overall retention rate: $retention_rate"
                fi
            else
                # Single batch stats
                processed_files=$(grep "Processed files:" processing_gpu4_7.log | tail -1 | cut -d: -f2 | xargs)
                retention_rate=$(grep "retention rate:" processing_gpu4_7.log | tail -1 | cut -d: -f2 | xargs)
                
                if [ -n "$processed_files" ]; then
                    echo "  - Processed files: $processed_files"
                    echo "  - Retention rate: $retention_rate"
                fi
            fi
        fi
        
        # Show monitoring statistics if available
        if [ -f "$OUTPUT_DIR/monitoring_stats.json" ]; then
            echo "  - Detailed statistics: $OUTPUT_DIR/monitoring_stats.json"
        fi
        
        # Show processed files log
        if [ -f "$OUTPUT_DIR/processed_files.json" ]; then
            echo "  - Processed files log: $OUTPUT_DIR/processed_files.json"
        fi
    fi
else
    echo "ERROR: Processing failed on GPU 4-7"
    echo "Last 10 lines of log:"
    tail -10 processing_gpu4_7.log
fi

echo ""
echo "Results summary:"
echo "  - Whisper input: $INPUT_DIR"
echo "  - Filtered output: $OUTPUT_DIR"
echo "  - GPU 4-7 vLLM log: vllm_gpu4_7.log"
echo "  - Processing log: processing_gpu4_7.log"

echo ""
echo "GPU 4-7 Server management:"
echo "  - PID: $VLLM_PID"
echo "  - Port: $PORT"
echo "  - API: http://localhost:$PORT/v1"

# Check if we should restart monitoring
if [ -n "$CONTINUOUS" ] && [ $PROCESS_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Continuous monitoring completed normally."
    echo "Options:"
    echo "1. Restart monitoring with same settings"
    echo "2. Restart monitoring with different settings"
    echo "3. Stop and clean up"
    
    read -p "Select option (1-3): " restart_option
    
    case $restart_option in
        1)
            echo "Restarting monitoring with same settings..."
            exec "$0"  # Restart the script
            ;;
        2)
            echo "Restarting script for new configuration..."
            # Keep vLLM server running but restart script for new settings
            echo "vLLM server (PID: $VLLM_PID) will continue running"
            exec "$0"
            ;;rh
        3)
            echo "Stopping and cleaning up..."
            ;;
    esac
fi

read -p "Stop GPU 4-7 server? (y/n): " stop_server
if [[ $stop_server =~ ^[Yy]$ ]]; then
    kill $VLLM_PID 2>/dev/null
    echo "GPU 4-7 server stopped"
    
    # Clean GPU 4-7 memory
    echo "Cleaning GPU 4-7 memory..."
    python3 -c "
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
    print('GPU 4-7 memory cleared')
" 2>/dev/null
else
    echo "GPU 4-7 server still running at http://localhost:$PORT"
    echo ""
    echo "Manual commands for continued processing:"
    echo ""
    echo "# Single batch processing:"
    echo "python3 QwenFilterOpt.py --input_dir \"$INPUT_DIR\" --output_dir \"$OUTPUT_DIR\" --threshold 0.80"
    echo ""
    echo "# Continuous monitoring:"
    echo "python3 QwenFilterOpt.py --input_dir \"$INPUT_DIR\" --output_dir \"$OUTPUT_DIR\" --threshold 0.80 --continuous --check_interval 30 --max_idle_time 1800"
    echo ""
    echo "# Custom monitoring intervals:"
    echo "python3 QwenFilterOpt.py --input_dir \"$INPUT_DIR\" --output_dir \"$OUTPUT_DIR\" --threshold 0.80 --continuous --check_interval 60 --max_idle_time 3600"
fi

echo ""
echo "GPU 4-7 processing script completed!"

# Show final status
if kill -0 $VLLM_PID 2>/dev/null; then
    echo "vLLM server is still running (PID: $VLLM_PID)"
    echo "API available at: http://localhost:$PORT/v1"
else
    echo "vLLM server has been stopped"
fi

echo "All logs and results are saved in the current directory and output folder"