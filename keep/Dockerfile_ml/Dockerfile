FROM nsml/ml:cuda9.0-cudnn7-tf-1.11torch0.4keras2.2

RUN pip3 install --no-cache-dir tensorflow-gpu==1.4 matplotlib scikit-image scipy

RUN wget https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip \
&& unzip protoc-3.3.0-linux-x86_64.zip \
&& PATH_TO_PROTOC=`pwd`

# install tensorflow-slim nets
RUN git clone https://github.com/tensorflow/models \
&& cd models/research/slim/ \
&& pip3 install -e . \
&& cd .. \
&& export PYTHONPATH=$PYTHONPATH:`pwd`

RUN ${PATH_TO_PROTOC?}/bin/protoc delf/protos/*.proto --python_out=. \
&& pip3 install -e

RUN ln -sf /models/research

ENV LANG C.UTF-8

