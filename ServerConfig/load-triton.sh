docker run --rm --name triton_py \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /home/linux1/model-repository:/models \
  icr.io/ibmz/ibmz-accelerated-for-nvidia-triton-inference-server:1.4.1 \
  tritonserver --model-repository=/models