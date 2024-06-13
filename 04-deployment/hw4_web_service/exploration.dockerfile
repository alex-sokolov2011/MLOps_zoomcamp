FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Show installed Python packages
RUN pip install --upgrade pip
RUN pip list

# Set the default command to bash
CMD ["/bin/bash"]