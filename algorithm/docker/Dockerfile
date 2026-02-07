FROM myfedml:mpi

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends iperf3 iproute2  iputils-ping net-tools && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt
