# The served TFT bundle as a tiny OCI image. The API pod's init container copies the two
# files from /model into an emptyDir at /models/tft (TFT_BUNDLE_DIR). Deployments reference
# this image by digest, so the pod template names the exact model that serves, and
# `kubectl rollout undo` restores the previous one together with its pin.
#
# Build context = a bundle folder, never the repo root:
#   docker build -f deploy/k8s/model/Dockerfile -t gridcast-model:fixture deploy/k8s/fixtures/tft-tiny
#   docker build -f deploy/k8s/model/Dockerfile -t <registry>/model:<tag> data/models/tft
FROM busybox:1.36
COPY tft.onnx tft.json /model/
