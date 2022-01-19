FROM tensorflow/tensorflow:latest-gpu
LABEL maintainer="code@adamltyson.com"
RUN pip install cellfinder
CMD ["bash"]
