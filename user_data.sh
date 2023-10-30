#!/bin/bash
# Assumes AWS CLI is installed and has permissions to describe instances

# Fetch all instances in the ASG
INSTANCES=$(aws ec2 describe-instances --filters "Name=tag:aws:autoscaling:groupName,Values=ECC-EC2-Single-ASG" --query 'Reservations[*].Instances[*].PrivateIpAddress' --output text)

# Convert the list into a format suitable for TF_CONFIG
WORKERS=($(echo $INSTANCES | tr " " "\n"))

# Identify the index of this worker
INDEX=0
for i in "${!WORKERS[@]}"; do
   if [[ "${WORKERS[$i]}" = "$(curl -s http://169.254.169.254/latest/meta-data/local-ipv4)" ]]; then
       INDEX=$i
       break
   fi
done

# Dynamically build the worker part of the TF_CONFIG
WORKER_JSON=""
for i in "${WORKERS[@]}"; do
    if [ -z "$WORKER_JSON" ]; then
        WORKER_JSON="\"$i:2222\""
    else
        WORKER_JSON="$WORKER_JSON,\"$i:2222\""
    fi
done

# Create the TF_CONFIG JSON
TF_CONFIG=$(echo "{
    \"cluster\": {
        \"worker\": [$WORKER_JSON]
    },
    \"task\": {\"type\": \"worker\", \"index\": $INDEX}
}" | jq .)

# Export it so your TensorFlow script can see it
export TF_CONFIG

# Now, run your script via Docker
docker pull 808556460535.dkr.ecr.us-east-1.amazonaws.com/image-classification:latest
docker run -it -e TF_CONFIG="$TF_CONFIG" 808556460535.dkr.ecr.us-east-1.amazonaws.com/image-classification:latest