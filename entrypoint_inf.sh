#!/bin/bash
set -e

if [ -n "$SA_KEY" ]; then
    echo "Writing SA_KEY to $GOOGLE_APPLICATION_CREDENTIALS..."
    mkdir -p "$(dirname "$GOOGLE_APPLICATION_CREDENTIALS")"
    echo "$SA_KEY" > "$GOOGLE_APPLICATION_CREDENTIALS"
    chmod 600 "$GOOGLE_APPLICATION_CREDENTIALS"
else
    echo "SA_KEY is not set. Exiting."
    exit 1
fi    

echo "Checking for W&B API key..."
if [ -n "$WANDB_API_KEY" ]; then
    echo "Logging into W&B..."
    wandb login "$WANDB_API_KEY"
else
    echo "WANDB_API_KEY is not set. Skipping W&B login."
fi

echo "Pulling the model..."
python gr00t/data/gcs_utils.py \
    --bucket_name="${BUCKET_NAME}" \
    --action=pull \
    --content_type=model \
    --identifiers="${MODEL_ID}" \
    --base_dir=model

exec "$@"