FROM ubuntu:20.04

#ENV http_proxy 'http://wwwcache.fmi.fi:8080'
#ENV https_proxy 'http://wwwcache.fmi.fi:8080'

# Install conda
RUN apt-get -qq update && apt-get -qq -y install curl bzip2 \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && apt-get -qq -y remove curl bzip2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

# Workdir and input/output/log dir
WORKDIR .
RUN mkdir input output log

# Create conda environment
COPY environment.yml .
ENV PYTHONDONTWRITEBYTECODE=true

# RUN conda env create -f environment.yml -n radargaugemerging
RUN conda install -c conda-forge mamba && \
    mamba env create -f environment.yml -n radargaugemerging && \
    mamba clean --all -f -y

# Copy code directory
COPY . /

# Run
WORKDIR .
ENV config hulehenri
ENV timestamp 202007071130
ENV mfb_state_file /mfb_state_{config}.dat
ENTRYPOINT conda run -n radargaugemerging python run_radargaugemerging.py --timestamp=$timestamp --config=$config --mfb_state_file=$mfb_state_file
