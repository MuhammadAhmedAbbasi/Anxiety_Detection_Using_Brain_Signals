FROM python:3.12
WORKDIR /ANXIETY DETECTION USING BRAIN SIGNAL
COPY . .

RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

EXPOSE 8090
EXPOSE 8091
