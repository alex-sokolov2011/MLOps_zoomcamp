FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

# Copy script and reqs.txt for docker to working directory
COPY calc_mean_pred_for_docker.py .
COPY requirements_for_docker.txt .

# Install necessary libraries
RUN pip install --upgrade pip
RUN pip install -r requirements_for_docker.txt

# Run script for docker with parameters for May 2023
CMD ["python", "calc_mean_pred_for_docker.py", "--year", "2023", "--month", "5"]
