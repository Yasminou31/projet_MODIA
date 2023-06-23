# Use the Jupyter Data Science image as the base image
FROM jupyter/datascience-notebook

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt ./

# Install the required Python packages # --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

# Copy the necessary files to run the main script to the working directory
COPY weights.pth user2id.pkl recipe2id.pkl test_script.csv model.py main.py ./

# Copy the necessary files to run the recommender app to the working directory
COPY sentiment_model.pkl recommender_app.py ./

# Expose port 8080 for the application
EXPOSE 8080

# Define a Docker argument for the script name
ARG SCRIPT_NAME

# Set the script name as an environment variable
ENV SCRIPT_NAME=${SCRIPT_NAME}

# Set the command to run the application using the provided script name
CMD python $SCRIPT_NAME